-- Citation-graph schema: AGE projection.
--
-- The relational tables in pg_schema_citation.sql are the durable truth; the
-- AGE graph 'citation_graph' this file projects them into is a derived
-- projection, never the other way round (project_graph() below is the only
-- writer of the graph, and it is safe to rerun at any time).
-- Reason: apache/age issue #2503 (open) -- after pg_dump/restore, Cypher
-- queries against a restored graph fail because ag_graph.graphid stores the
-- bare oid of the ORIGINAL database, which the restore target does not
-- share. kb artifacts are built with pg_dump (see deploy/), so a graph that
-- were itself the source of truth would come back broken on every restore.
-- A relational table survives pg_dump/restore unmodified and rebuilds the
-- graph from scratch on the other side by calling project_graph() again.
--
-- The graph is named 'citation_graph', not 'citation': AGE's create_graph()
-- creates its own schema named after the graph, and that schema must not
-- collide with the citation schema pg_schema_citation.sql defines.
--
-- Applied after pg_schema_citation.sql: project_graph() reads
-- citation.work/citation.cites, which that file creates.

-- Escapes a plain string for safe use inside a *Cypher* single-quoted string
-- literal (backslash-style escaping, like Cypher/JSON -- NOT SQL's
-- quote-doubling). This is the ONLY thing standing between a title/key
-- containing a quote or backslash and a broken (or, for a hostile source
-- feed, injected) Cypher command built by string concatenation below.
-- Order matters: backslashes must be doubled first, or a later inserted
-- backslash (e.g. from escaping a quote) would itself get doubled.
CREATE OR REPLACE FUNCTION citation.cypher_literal(raw TEXT) RETURNS TEXT AS $$
    SELECT replace(replace(replace(replace(replace(
        coalesce(raw, ''),
        '\', '\\'),
        '''', '\'''),
        '"', '\"'),
        E'\n', '\n'),
        E'\r', '\r');
$$ LANGUAGE sql IMMUTABLE;

-- Rebuilds the AGE graph 'citation_graph' from citation.work/citation.cites.
--
-- Contract the caller MUST satisfy: `LOAD 'age';` and
-- `SET search_path = ag_catalog, "$user", public;` before calling this --
-- AGE's LOAD is a session-local library load and does not take effect from
-- inside a function body (see deploy/pg/README.md "Activation"). Every
-- ag_catalog identifier below is schema-qualified anyway, so search_path is
-- only needed for the agtype column type in this function's own callers.
--
-- Full reprojection, not an incremental MERGE: drop_graph()+create_graph()
-- guarantees no orphaned vertex/edge label data survives a row that
-- disappeared from citation.work/cites between two runs. An incremental
-- MERGE would need to diff two representations to find deletions and is
-- simply more code for the same result.
--
-- create_vlabel/create_elabel run unconditionally, even when work/cites are
-- empty, so the label tables citation_graph."Work"/"CITES" always exist for
-- the count(*) at the end -- AGE only auto-creates a label table on first
-- use inside a CREATE clause, and an empty citation.work would otherwise
-- leave "Work" missing rather than merely empty.
--
-- BULK LOAD, not one Cypher command per row. A label table is an ordinary
-- table (vertex: id graphid, properties agtype; edge: + start_id, end_id),
-- and AGE's Cypher reads whatever is in it, so two INSERT ... SELECT
-- statements fill the whole graph in one pass each. The row-by-row Cypher
-- form this replaced cost O(V*E): AGE indexes no vertex property by
-- itself, so every edge's `MATCH (a:Work {key: ...}), (b:Work {key: ...})`
-- sequentially scanned the entire vertex label twice, on top of a fresh
-- parse+plan per row. Measured on this instance (AGE 1.7.0, PostgreSQL
-- 17.11), same data, identical resulting graph both ways:
--
--     438 works / 2 425 edges (live)      row-by-row 0.70 s   bulk 0.08 s
--   10 000 works / 50 235 edges (synth)   row-by-row  221 s   bulk 0.26 s
--
-- A btree index on the key property
-- (agtype_access_operator(properties, '"key"')) does NOT rescue the
-- row-by-row form -- measured 254 s on the same 10k/50k data, i.e. slower
-- than no index at all: AGE 1.7.0 does not plan a property MATCH through
-- such an index, so the scan stays and the index maintenance is added to
-- it. The cost is in the shape, not in a missing index.
--
-- The vertex graphid's entry id IS citation.work.id, so an edge's endpoints
-- are arithmetic (_graphid(<Work label>, ci.citing)) rather than a lookup
-- by key -- the per-edge join disappears with the per-edge command. Both
-- label sequences are then advanced past the ids just used, so a Cypher
-- CREATE issued against the projection afterwards cannot collide with one.
--
-- Properties are built as jsonb and cast to agtype: Postgres's own JSON
-- writer escapes quotes, backslashes and newlines in a third-party title,
-- and no command text is assembled here at all -- with it goes the
-- $CYPHERQ$ delimiter hazard the string-building form had to guard
-- against. citation.cypher_literal() above stays: the READ side
-- (pg_graph_cypher.py) still splices a key into a Cypher command.
-- jsonb_strip_nulls keeps a missing year/title out of the property map
-- entirely rather than storing a null one, exactly as the Cypher form did.
--
-- The INSERTs go through EXECUTE, not as static statements: drop_graph()
-- destroys and create_vlabel() recreates citation_graph."Work"/"CITES" on
-- every call, and a plpgsql statement referencing them by name would carry
-- a cached plan for a relation this very function has already dropped.
CREATE OR REPLACE FUNCTION citation.project_graph()
RETURNS TABLE(vertices BIGINT, edges BIGINT) AS $$
DECLARE
    graph_oid   OID;
    work_label  INTEGER;
    cites_label INTEGER;
    work_seq    TEXT;
    cites_seq   TEXT;
    max_work_id BIGINT;
    n_cites     BIGINT;
BEGIN
    IF EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'citation_graph') THEN
        PERFORM ag_catalog.drop_graph('citation_graph', true);
    END IF;
    PERFORM ag_catalog.create_graph('citation_graph');
    PERFORM ag_catalog.create_vlabel('citation_graph', 'Work');
    PERFORM ag_catalog.create_elabel('citation_graph', 'CITES');

    SELECT g.graphid INTO graph_oid
      FROM ag_catalog.ag_graph g WHERE g.name = 'citation_graph';
    SELECT l.id, format('citation_graph.%I', l.seq_name) INTO work_label, work_seq
      FROM ag_catalog.ag_label l WHERE l.graph = graph_oid AND l.name = 'Work';
    SELECT l.id, format('citation_graph.%I', l.seq_name) INTO cites_label, cites_seq
      FROM ag_catalog.ag_label l WHERE l.graph = graph_oid AND l.name = 'CITES';

    EXECUTE '
        INSERT INTO citation_graph."Work" (id, properties)
        SELECT ag_catalog._graphid($1, w.id),
               jsonb_strip_nulls(jsonb_build_object(
                   ''key'', w.key, ''kind'', w.kind,
                   ''year'', w.year, ''title'', w.title))::text::ag_catalog.agtype
        FROM citation.work w'
    USING work_label;

    EXECUTE '
        INSERT INTO citation_graph."CITES" (id, start_id, end_id, properties)
        SELECT ag_catalog._graphid(
                   $2, row_number() OVER (ORDER BY ci.citing, ci.cited, ci.source)),
               ag_catalog._graphid($1, ci.citing),
               ag_catalog._graphid($1, ci.cited),
               jsonb_build_object(''source'', ci.source)::text::ag_catalog.agtype
        FROM citation.cites ci'
    USING work_label, cites_label;

    SELECT coalesce(max(w.id), 0) INTO max_work_id FROM citation.work w;
    SELECT count(*) INTO n_cites FROM citation.cites;
    PERFORM setval(work_seq::regclass, GREATEST(max_work_id, 1), max_work_id > 0);
    PERFORM setval(cites_seq::regclass, GREATEST(n_cites, 1), n_cites > 0);

    EXECUTE 'SELECT count(*) FROM citation_graph."Work"' INTO vertices;
    EXECUTE 'SELECT count(*) FROM citation_graph."CITES"' INTO edges;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
