-- Citation-graph schema.
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

-- crawl_step: the crawl's own journal (BFS snowball), one row
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
    action        TEXT NOT NULL,
    n_found       INTEGER,
    n_kept        INTEGER,
    reason        TEXT,
    at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The decision's machine-readable part, in columns of its own: which graph
-- node the candidate resolved to, the score it was measured at, and the
-- threshold it was measured against. Added separately, ADD COLUMN IF NOT
-- EXISTS, for the same reason the action CHECK is applied separately -- the
-- table already exists on this instance and on every artifact restored from
-- an earlier dump.
--
-- These three used to live inside `reason` as "score=... tau=... node=...",
-- and three consumers parsed them back out (the score distribution query in
-- citations/crawl.py, the depth-1 node set in citations/hub_report.py, the
-- public artifact's journal cut in deploy/citation_profile.py). A number the
-- pipeline reads is not prose: `reason` keeps only what a human reads.
--
-- node_key is "the node this decision resolved to": for a kept candidate the
-- registry node it was merged into (two OpenAlex records of one work share a
-- node), for a twin promotion the seed work the candidate turned out to be.
ALTER TABLE citation.crawl_step ADD COLUMN IF NOT EXISTS node_key TEXT;
ALTER TABLE citation.crawl_step ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;
ALTER TABLE citation.crawl_step ADD COLUMN IF NOT EXISTS tau DOUBLE PRECISION;

-- The action vocabulary lives in a NAMED constraint applied separately, not
-- inline in CREATE TABLE: the crawl grows new kinds of decision (hub-skip
-- arrived when depth-2 turned out to pull >51k citers through a handful of
-- heavily-cited classics), and an inline CHECK cannot be widened on an
-- instance that already has the table. DROP IF EXISTS + ADD is idempotent on
-- both a fresh database and one created before the value existed.
--
-- hub-skip: the node was NOT expanded upward because its citer count is past
-- the cap. It is a decision, not an error -- without a row saying so, "why is
-- this node a dead end" is unanswerable after the fact, which is the whole
-- reason crawl_step exists.
ALTER TABLE citation.crawl_step DROP CONSTRAINT IF EXISTS crawl_step_action_check;
ALTER TABLE citation.crawl_step ADD CONSTRAINT crawl_step_action_check
    CHECK (action IN ('seed', 'seed-missing', 'fetch', 'keep', 'drop', 'hub-skip', 'error'));

CREATE INDEX IF NOT EXISTS crawl_step_crawl_depth_idx ON citation.crawl_step (crawl_id, depth);

-- The public artifact's journal cut asks which rows name a document or a
-- work the package leaves behind (deploy/citation_profile.py's
-- crawl_step_cut_ctes). It asks that as three separate branches, one per
-- column carrying a name, precisely so each can be an index lookup driven
-- from the tiny set of cut names -- an OR of the three inside one subquery
-- is a single non-sargable qualifier and reaches no index at all.
--
-- All three verified in use, EXPLAIN (ANALYZE) of the real COPY select on
-- the live instance, over a 100k-row depth-2-sized journal inserted inside
-- a rolled-back transaction: three nested loops, one per index, each driven
-- from the tiny cut-names side (10 names, 10 loops each), 21 ms for the
-- whole statement. At today's 604 rows the planner hashes the table instead
-- and is right to -- the indexes are there for the size the crawl grows to.
-- The third branch used to be strpos() over `reason` and could only ever
-- scan; with node_key a column it is an index lookup like the other two.
CREATE INDEX IF NOT EXISTS crawl_step_frontier_key_idx ON citation.crawl_step (frontier_key);
CREATE INDEX IF NOT EXISTS crawl_step_candidate_key_idx ON citation.crawl_step (candidate_key);
CREATE INDEX IF NOT EXISTS crawl_step_node_key_idx ON citation.crawl_step (node_key);

-- One-time backfill of the three columns for journal rows written before
-- they existed, parsed out of the prose that carried them. Idempotent in
-- both directions: it only ever fills a NULL, and a row whose reason never
-- carried the value keeps its NULL (a twin promotion has no score, a hub
-- skip no node). Kept in the schema file rather than run by hand once,
-- because every instance this schema is applied to -- this one, a restored
-- artifact, a developer's fresh database -- meets the same old rows.
UPDATE citation.crawl_step SET
    node_key = coalesce(node_key,
                        nullif(substring(reason from 'node=([^ ]+)'), ''),
                        nullif(substring(reason from 'seed=([^ ]+)'), '')),
    score = coalesce(score, substring(reason from 'score=(-?[0-9.]+)')::double precision),
    tau = coalesce(tau, substring(reason from 'tau=(-?[0-9.]+)')::double precision)
WHERE reason IS NOT NULL
  AND (node_key IS NULL OR score IS NULL OR tau IS NULL)
  AND reason ~ '(node=|seed=|score=|tau=)';

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
-- (pg_graph_queries.py) still splices a key into a Cypher command.
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

-- Public-artifact policy for the WHOLE citation schema, one row (id = 1)
-- decided once by the corpus owner -- the same DATA-not-code discipline as
-- corpus.documents.legal_class/public_distribution (kb/CLAUDE.md
-- LEGAL_IS_DATA), applied here to the crawl's own record rather than to an
-- individual document (see deploy/citation_profile.py for what each mode
-- means to a public build, and deploy/citation_dump.py for how it is
-- applied to the dump).
--
-- No default row is inserted here, and none may be inserted anywhere but by
-- the owner: absence of a row is the same UNCLASSIFIED_FAILS_BUILD refusal
-- corpus.documents applies per-document (deploy/legal_profile.py) -- a
-- public build over a citation schema with no policy decided MUST fail with
-- an explicit message ("citation schema not classified for a public
-- artifact"), not silently ship or silently strip a crawl record that
-- names third-party titles, abstracts and citation edges.
CREATE TABLE IF NOT EXISTS citation.public_policy (
    id          SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    mode        TEXT NOT NULL CHECK (mode IN ('full-skeleton', 'topology-only', 'none')),
    note        TEXT NOT NULL,
    decided_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE citation.public_policy IS
    'Row written by the corpus owner, never by code: decides whether/how the '
    'citation schema ships in the public artifact (full-skeleton | '
    'topology-only | none). No row = the public build refuses '
    '(deploy/citation_profile.py), the same way an unclassified '
    'corpus.documents row refuses deploy/legal_profile.py.';
