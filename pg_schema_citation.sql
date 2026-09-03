-- Citation-graph schema: relational data definition.
--
-- The tables below (citation.work / citation.cites / citation.crawl_step /
-- citation.public_policy) are the durable truth of the citation graph.
-- The AGE graph projected from them is a separate, derived structure,
-- defined in pg_schema_citation_graph.sql -- see that file's header for why
-- the graph is never itself the source of truth. Two more files
-- complete the definition and are applied after this one:
-- pg_schema_citation_constraints.sql carries what a CREATE TABLE IF NOT
-- EXISTS cannot establish on an instance that already has the table -- the
-- closed-vocabulary CHECKs and the document FK's referential action -- and
-- pg_schema_citation_backfill.sql the journal's one-time prose-to-column
-- backfill, which depends on the columns this file adds.

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
    -- citation_vocab.WorkKind. The CHECK that mirrors them is NOT here: an
    -- inline one cannot be widened on a table that already exists, and
    -- CREATE TABLE IF NOT EXISTS makes every existing instance exactly that
    -- case. It is applied as the named work_kind_check by
    -- pg_schema_citation_constraints.sql, which compares before it replaces.
    kind              TEXT NOT NULL,
    -- ON DELETE: nothing, deliberately -- NOT the SET NULL this column
    -- carried. Under SET NULL the two constraints on this table could not
    -- both hold: deleting a corpus.documents row UPDATEs the referencing
    -- work row to document_id = NULL, and
    -- work_our_document_has_document_check (an our-document row must name
    -- its document) rejects exactly that, so the delete aborted with a
    -- CHECK violation naming a row the deleter never touched. The loaders DO delete documents -- pg_load_djvu.py and
    -- pg_load_metadata.py re-insert each of theirs, pg_load.py and
    -- pg_load_external.py drop what vanished from a manifest -- and the IIS
    -- documents they delete are precisely the crawl's seeds. So the refusal
    -- is a live path, and it must at least be the error it really is: a
    -- referential one, naming this constraint. Demoting the node first is
    -- the deleter's step, and EXTENDING.md procedure A carries it.
    document_id       TEXT REFERENCES corpus.documents(id),
    exclusion_reason  TEXT,
    -- Where the fields above came from: the raw source record (or enough of
    -- it to re-derive a verdict without re-fetching), not prose.
    evidence          JSONB,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding         vector(1024)
    -- A document already in our own corpus must be traceable to it; an
    -- excluded item must carry the reason it was excluded for -- either
    -- omission would make the corresponding column decorative rather than
    -- load-bearing. Both are stated in
    -- pg_schema_citation_constraints.sql, as
    -- work_our_document_has_document_check and
    -- work_excluded_has_reason_check: each consumes a value of the closed
    -- kind vocabulary, and a value spelled inline here can never be
    -- corrected -- CREATE TABLE IF NOT EXISTS is a no-op on every instance
    -- that already carries the table, which is every instance. Anonymous
    -- besides, so nothing could even have found them by name to migrate.
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
-- external_ids carries per-source keys (openalex, s2, zbmath, doi, ...) plus
-- the titles and years a node is known by. It was indexed with GIN for
-- "does any source already know this id", but nothing asks that in SQL: the
-- identity question is answered in Python, in citations/registry.WorkRegistry
-- by an in-memory index over the ids the crawl has seen, and every SQL reader
-- of the column extracts a field (->>'titles', jsonb_array_length) -- a shape
-- default jsonb_ops cannot serve. So the index was write amplification on the
-- crawl's bulkiest path: one GIN entry per key AND per value of that JSONB,
-- for thousands of upserts per level, plus the size and restore time it adds
-- to the public artifact, which ships the column.
-- Dropped rather than left in place: an index nothing reads is not free.
-- A SQL-side identity lookup, if one is ever written, arrives WITH its query
-- and in the shape that query needs (jsonb_path_ops for @>, or a btree
-- expression index) -- not as this one, kept on the chance of it.
DROP INDEX IF EXISTS citation.work_external_ids_gin;

-- No index on `kind` either, and that too is measured rather than
-- forgotten. Its two per-kind readers are in citations/twin_pass.py:
-- seed_titles() (kind = 'our-document' AND document_id IS NOT NULL) and
-- skeleton_nodes() (kind = 'external-skeleton', ORDER BY key). EXPLAIN
-- (ANALYZE, BUFFERS) on the live instance with the table grown to
-- depth-2 size (60438 rows, inside a rolled-back transaction):
--
--   seed_titles already reaches its 75 rows through work_document_id_idx
--   (Bitmap Index Scan on document_id IS NOT NULL, 30 buffers, 0.41 ms),
--   because every non-seed row has a NULL there; adding ON citation.work
--   (key) WHERE kind = 'our-document' turns that into a BitmapAnd over two
--   indexes and measures 31 buffers, 0.42 ms -- worse, not better;
--   skeleton_nodes returns 60363 rows of 60438, i.e. the table itself, in
--   key order through work_key_key. No selective index exists for a query
--   whose answer IS the table.
--
-- So the index would be one more write on every upsert of the crawl's
-- bulkiest path and would serve neither reader: the same verdict the GIN
-- index got, reached the same way.

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
-- relation is HOW the candidate reached the frontier, and the crawl acts
-- on it: only a node reached as a citer expands at depth >= 2 (kb/CLAUDE.md
-- SNOWBALL_FRONTIER), and the hub measurement groups its whole verdict by
-- it. Its two values are as closed a vocabulary as action's, and are
-- CHECKed beside it in pg_schema_citation_constraints.sql. It lived in the
-- prose, so
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
-- The three values are declared once on the Python side, in
-- citation_vocab.PublicPolicyMode -- the same one-declaration discipline
-- work.kind and crawl_step.action follow, and the same live comparison
-- against pg_get_constraintdef() holds the two halves together
-- (tests/test_citation_vocab.py). deploy/manifest_contract.CitationMode
-- extends that class rather than restating its values. The CHECK itself is
-- the named public_policy_mode_check applied by
-- pg_schema_citation_constraints.sql, for the reason work.kind's is: this
-- table exists on every instance, so an inline CHECK could never be
-- widened again.
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
    mode        TEXT NOT NULL,
    note        TEXT NOT NULL,
    decided_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE citation.public_policy IS
    'Row written by the corpus owner, never by code: decides whether/how the '
    'citation schema ships in the public artifact (full-skeleton | '
    'topology-only | none). No row = the public build refuses '
    '(deploy/citation_profile.py), the same way an unclassified '
    'corpus.documents row refuses deploy/legal_profile.py.';
