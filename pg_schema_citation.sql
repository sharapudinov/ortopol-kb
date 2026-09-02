-- Citation-graph schema: relational data definition.
--
-- The tables below (citation.work / citation.cites / citation.crawl_step /
-- citation.public_policy) are the durable truth of the citation graph.
-- The AGE graph projected from them is a separate, derived structure,
-- defined in pg_schema_citation_graph.sql -- see that file's header for why
-- the graph is never itself the source of truth. The journal's one-time
-- prose-to-column backfill is likewise separate, in
-- pg_schema_citation_backfill.sql: it depends on the columns this file
-- creates and must be applied after them.

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
    -- The same four values are declared once on the Python side, in
    -- citation_vocab.WorkKind; tests/test_citation_vocab.py compares this
    -- CHECK against them in both directions, so an extra, missing or
    -- renamed value fails there rather than at a COPY.
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
-- The добор pg_embed.py works does: rows the crawl left without a vector
-- (a title edited by hand, a node written before the embedder was reachable).
-- Its loop re-asks "the next 16 rows where embedding IS NULL" until none are
-- left, and the crawl writes the vector inline, so almost every row is a
-- non-NULL row to walk past. Partial and on id alone: the index holds only
-- the pending rows, so it is empty in the normal case and costs the crawl's
-- own writes nothing. Verified on this instance (438 works + one NULL row,
-- inside a rolled-back transaction): Seq Scan + Sort becomes Index Scan
-- using work_pending_embedding_idx, without ANALYZE and without the
-- planner needing a bigger table to prefer it.
CREATE INDEX IF NOT EXISTS work_pending_embedding_idx ON citation.work (id)
    WHERE embedding IS NULL;
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

-- Two more of the same kind, promoted for the same reason.
--
-- relation ('cites' | 'referenced') is HOW the candidate reached the
-- frontier, and the crawl acts on it: only a node reached by 'cites'
-- expands at depth >= 2 (kb/CLAUDE.md SNOWBALL_FRONTIER), and the hub
-- measurement groups its whole verdict by it. It lived in the prose, so
-- citations/hub_report.py had to re-derive it from citation.work.evidence
-- with a coalesce(..., 'unknown') -- reading a decision off a blob shaped
-- by registry.Node.absorb instead of off the decision that made it.
--
-- cited_by_count is the measured quantity a hub-skip turned on. The cap it
-- was compared against stays in the prose: it is the run's own --hub-cap,
-- and nothing queries it. No index on either -- both are grouped over, not
-- searched by; an index arrives with the query that needs one.
ALTER TABLE citation.crawl_step ADD COLUMN IF NOT EXISTS relation TEXT;
ALTER TABLE citation.crawl_step ADD COLUMN IF NOT EXISTS cited_by_count BIGINT;

-- The action vocabulary lives in a NAMED constraint applied separately, not
-- inline in CREATE TABLE: the crawl grows new kinds of decision (hub-skip
-- arrived when depth-2 turned out to pull >51k citers through a handful of
-- heavily-cited classics), and an inline CHECK cannot be widened on an
-- instance that already has the table.
--
-- hub-skip: the node was NOT expanded upward because its citer count is past
-- the cap. It is a decision, not an error -- without a row saying so, "why is
-- this node a dead end" is unanswerable after the fact, which is the whole
-- reason crawl_step exists.
--
-- Replaced only when the vocabulary actually differs, for the same reason
-- the reason-parse backfill below carries a registry: ADD CONSTRAINT
-- validates the new CHECK against every existing row under an ACCESS
-- EXCLUSIVE lock, and this schema is applied on every `pg_graph.py init`
-- AND every non-dry-run crawl, against an append-only journal that grows by
-- ~100k rows per depth-2 crawl. Value-idempotent DROP+ADD was never the
-- gap; the gap was paying a full validation scan to arrive at the
-- constraint that was already there.
--
-- The same seven values are declared once on the Python side, in
-- citation_vocab.CrawlAction, which is where the crawl reads them from;
-- tests/test_citation_vocab.py compares the two in both directions.
--
-- Compared as the VOCABULARY, not as text: pg_get_constraintdef() renders
-- the same CHECK as `action = ANY (ARRAY[...])`, and its exact spelling is
-- the server's business (and its version's). The literals inside it are
-- ours, so they are what is compared -- an extra value, a missing one or a
-- renamed one all differ, and nothing else does.
DO $action_check$
DECLARE
    wanted     CONSTANT text[] := ARRAY['seed', 'seed-missing', 'fetch', 'keep',
                                        'drop', 'hub-skip', 'error'];
    definition text;
    current_vocabulary text[];
BEGIN
    SELECT pg_get_constraintdef(c.oid) INTO definition
    FROM pg_constraint c
    WHERE c.conrelid = 'citation.crawl_step'::regclass
      AND c.conname = 'crawl_step_action_check';

    IF definition IS NOT NULL THEN
        SELECT array_agg(m[1] ORDER BY m[1]) INTO current_vocabulary
        FROM regexp_matches(definition, '''([^'']*)''', 'g') AS m;
    END IF;

    IF current_vocabulary IS NOT DISTINCT FROM
       (SELECT array_agg(value ORDER BY value) FROM unnest(wanted) AS value) THEN
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE citation.crawl_step '
            'DROP CONSTRAINT IF EXISTS crawl_step_action_check';
    EXECUTE format(
        'ALTER TABLE citation.crawl_step ADD CONSTRAINT crawl_step_action_check '
        'CHECK (action IN (%s))',
        (SELECT string_agg(quote_literal(value), ', ') FROM unnest(wanted) AS value));
END
$action_check$;

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
