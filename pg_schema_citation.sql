-- Citation-graph schema (plan 038).
--
-- The relational tables are the durable truth; the AGE graph is a derived
-- projection of them, never the other way round (project_graph() below is
-- the only writer of the graph, and it is safe to rerun at any time).
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
-- collide with the citation schema defined here.

CREATE SCHEMA IF NOT EXISTS citation;

-- work: one row per bibliographic item the crawl has ever seen, including
-- ones we chose not to keep (kind = 'excluded' -- see exclusion_reason).
-- Dropping excluded rows outright would let the crawl rediscover and
-- re-fetch the same dead end on every run; keeping them with a reason is
-- how crawl_step's journal stays honest about what "already looked at"
-- means.
--
-- id is a surrogate: pg_embed.py (TARGETS) addresses rows by integer id the
-- same way it does for corpus.pages and measurements.run, and `key` (the
-- source's own canonical identifier, e.g. 'https://openalex.org/W123') is
-- not guaranteed to be a convenient primary key across sources (zbMATH,
-- OpenAlex and S2 all shape theirs differently).
CREATE TABLE IF NOT EXISTS citation.work (
    id                BIGSERIAL PRIMARY KEY,
    key               TEXT NOT NULL UNIQUE,
    doi               TEXT,
    title             TEXT,
    abstract          TEXT,
    year              INTEGER,
    authors           JSONB,
    external_ids      JSONB,
    source            TEXT NOT NULL,
    -- our-document: already in corpus.documents (document_id set).
    -- external-skeleton: a citation-graph node we have metadata for but no
    --   full text -- most of the graph, by construction.
    -- indexed: an external-skeleton item promoted after being read in full
    --   (kept distinct from our-document, which means "one of the 68/4 IIS
    --   works", not "we happened to read this one too").
    -- excluded: considered and rejected by the crawl filter.
    kind              TEXT NOT NULL CHECK (kind IN ('our-document', 'external-skeleton', 'indexed', 'excluded')),
    document_id       TEXT REFERENCES corpus.documents(id) ON DELETE SET NULL,
    exclusion_reason  TEXT,
    -- Where the fields above came from: the raw source record (or enough of
    -- it to re-derive a verdict without re-fetching), not prose.
    evidence          JSONB,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding         vector(1024),
    -- A document already in our own corpus must be traceable to it; an
    -- excluded item must carry the reason it was excluded for -- either
    -- omission would make the corresponding column decorative rather than
    -- load-bearing.
    CHECK (kind <> 'our-document' OR document_id IS NOT NULL),
    CHECK (kind <> 'excluded' OR exclusion_reason IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS work_document_id_idx ON citation.work (document_id);
CREATE INDEX IF NOT EXISTS work_embedding_hnsw ON citation.work
    USING hnsw (embedding vector_cosine_ops);
-- external_ids carries per-source keys (openalex, s2, zbmath, doi, ...) for
-- an item first seen through one source and later matched from another;
-- GIN makes "does any source already know this id" a lookup, not a scan.
CREATE INDEX IF NOT EXISTS work_external_ids_gin ON citation.work USING GIN (external_ids);

-- cites: a directed edge, itself sourced (the same pair can be attested by
-- more than one crawl source, each independently, hence the PK includes
-- source rather than deduplicating across sources on ingest).
CREATE TABLE IF NOT EXISTS citation.cites (
    citing      BIGINT NOT NULL REFERENCES citation.work(id) ON DELETE CASCADE,
    cited       BIGINT NOT NULL REFERENCES citation.work(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    evidence    JSONB,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (citing, cited, source),
    CHECK (citing <> cited)
);

CREATE INDEX IF NOT EXISTS cites_cited_idx ON citation.cites (cited);

-- crawl_step: the crawl's own journal (BFS snowball, task 038.5), one row
-- per decision it made -- seed accepted, candidate fetched, kept or
-- dropped by the filter, or errored. Without this, "why is X in the graph"
-- or "why isn't Y" is unanswerable after the fact; run 53 / the
-- literature_sweep_query precedent is exactly this kind of forgotten
-- decision trail.
CREATE TABLE IF NOT EXISTS citation.crawl_step (
    id            BIGSERIAL PRIMARY KEY,
    crawl_id      TEXT NOT NULL,
    depth         INTEGER NOT NULL,
    frontier_key  TEXT,
    candidate_key TEXT,
    action        TEXT NOT NULL CHECK (action IN ('seed', 'seed-missing', 'fetch', 'keep', 'drop', 'error')),
    n_found       INTEGER,
    n_kept        INTEGER,
    reason        TEXT,
    at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS crawl_step_crawl_depth_idx ON citation.crawl_step (crawl_id, depth);

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
-- disappeared from citation.work/cites between two runs, at a cost that is
-- negligible at the graph's actual size (~10k nodes). An incremental MERGE
-- would need to diff two representations to find deletions and is simply
-- more code for the same result at this scale.
--
-- create_vlabel/create_elabel run unconditionally, even when work/cites are
-- empty, so the label tables citation_graph."Work"/"CITES" always exist for
-- the count(*) at the end -- AGE only auto-creates a label table on first
-- use inside a CREATE clause, and an empty citation.work would otherwise
-- leave "Work" missing rather than merely empty.
--
-- ag_catalog.cypher()'s second argument MUST be a dollar-quoted string
-- constant, not merely a text-valued expression: AGE rewrites the query at
-- parse-analysis time by locating a dollar-quoted literal in the parse
-- tree, and a plain quoted (or E'') literal -- what format('%L', ...) would
-- produce -- fails with "a dollar-quoted string constant is expected"
-- (confirmed against 1.7.0 on this instance). EXECUTE still works: it
-- reparses the assembled SQL text fresh each time, so a dynamically built
-- string that happens to use $CYPHERQ$...$CYPHERQ$ delimiters is, by the
-- time AGE's hook sees it, indistinguishable from one written by hand in a
-- static query. CYPHERQ is checked against the command text first because
-- nothing stops a title/key from containing that exact tag; if it ever did,
-- silently using it as a delimiter would truncate the command instead of
-- raising.
CREATE OR REPLACE FUNCTION citation.project_graph()
RETURNS TABLE(vertices BIGINT, edges BIGINT) AS $$
DECLARE
    w RECORD;
    c RECORD;
    cyp TEXT;
BEGIN
    IF EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'citation_graph') THEN
        PERFORM ag_catalog.drop_graph('citation_graph', true);
    END IF;
    PERFORM ag_catalog.create_graph('citation_graph');
    PERFORM ag_catalog.create_vlabel('citation_graph', 'Work');
    PERFORM ag_catalog.create_elabel('citation_graph', 'CITES');

    FOR w IN SELECT key, kind, year, title FROM citation.work LOOP
        cyp := format(
            'CREATE (:Work {key: ''%s'', kind: ''%s''%s%s})',
            citation.cypher_literal(w.key),
            citation.cypher_literal(w.kind),
            CASE WHEN w.year IS NULL THEN '' ELSE format(', year: %s', w.year) END,
            CASE WHEN w.title IS NULL THEN '' ELSE format(', title: ''%s''', citation.cypher_literal(w.title)) END
        );
        IF cyp LIKE '%$CYPHERQ$%' THEN
            RAISE EXCEPTION 'citation.work.key=%: command text collides with the $CYPHERQ$ delimiter', w.key;
        END IF;
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''citation_graph'', $CYPHERQ$%s$CYPHERQ$) AS (v ag_catalog.agtype)', cyp);
    END LOOP;

    FOR c IN
        SELECT wa.key AS citing_key, wb.key AS cited_key, ci.source AS src
        FROM citation.cites ci
        JOIN citation.work wa ON wa.id = ci.citing
        JOIN citation.work wb ON wb.id = ci.cited
    LOOP
        cyp := format(
            'MATCH (a:Work {key: ''%s''}), (b:Work {key: ''%s''}) CREATE (a)-[:CITES {source: ''%s''}]->(b)',
            citation.cypher_literal(c.citing_key),
            citation.cypher_literal(c.cited_key),
            citation.cypher_literal(c.src)
        );
        IF cyp LIKE '%$CYPHERQ$%' THEN
            RAISE EXCEPTION 'citation.cites %->% (%): command text collides with the $CYPHERQ$ delimiter', c.citing_key, c.cited_key, c.src;
        END IF;
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''citation_graph'', $CYPHERQ$%s$CYPHERQ$) AS (v ag_catalog.agtype)', cyp);
    END LOOP;

    EXECUTE 'SELECT count(*) FROM citation_graph."Work"' INTO vertices;
    EXECUTE 'SELECT count(*) FROM citation_graph."CITES"' INTO edges;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
