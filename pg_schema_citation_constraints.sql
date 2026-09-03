-- Citation-graph schema: the constraints CREATE TABLE cannot establish.
--
-- pg_schema_citation.sql declares the tables with CREATE TABLE IF NOT
-- EXISTS, which changes nothing on an instance that already carries them --
-- and every instance does: the schema is applied on every `pg_graph.py
-- init` AND at the start of every non-dry-run crawl. So a constraint that
-- CHANGES (a vocabulary gaining a value, a referential action decided
-- differently) can only arrive here, as an idempotent migration.
--
-- Idempotent means more than "safe to re-run": ALTER TABLE ... ADD
-- CONSTRAINT validates the new constraint against every existing row under
-- an ACCESS EXCLUSIVE lock, and citation.crawl_step grows by ~100k rows per
-- depth-2 crawl. Everything below therefore READS what the instance
-- currently has and does nothing when it already agrees.
--
-- Applied after the data definition and before the AGE projection
-- (pg_graph_common.SCHEMA_PATHS).
--
-- The migrator itself is not here and not in this schema: it is
-- public.ensure_vocabulary_check (pg_schema_vocabulary.sql, applied first),
-- because a measurements table declares its vocabulary through it too and
-- `citation` is a schema the packager ships in three modes including none.
-- Each vocabulary below is one call: a NAMED constraint, compared before it
-- is replaced.

-- action: what kind of decision a journal row is. hub-skip means the node
-- was NOT expanded upward because its citer count is past the cap -- a
-- decision, not an error, and without a row saying so "why is this node a
-- dead end" is unanswerable after the fact, which is the whole reason
-- crawl_step exists.
--
-- The same seven values are declared once on the Python side, in
-- citation_vocab.CrawlAction, which is where the crawl reads them from;
-- tests/test_citation_vocab.py compares the two in both directions, against
-- this file and against pg_get_constraintdef().
DO $action_check$ BEGIN PERFORM public.ensure_vocabulary_check(
    'citation.crawl_step', 'action', 'crawl_step_action_check',
    ARRAY['seed', 'seed-missing', 'fetch', 'keep', 'drop', 'hub-skip', 'error']);
END $action_check$;

-- relation: how the candidate reached the frontier. As closed a vocabulary
-- as action and as load-bearing -- only a node reached as a citer expands
-- at depth >= 2 (kb/CLAUDE.md SNOWBALL_FRONTIER) -- so it gets the same
-- CHECK rather than staying the one promoted column nothing constrains.
-- citation_vocab.Relation is its single Python declaration, and
-- citations/threshold_store.py mirrors the same pair on the measurements
-- table that records a calibration.
--
-- NULL stays legal, and that is not laxity: only fetch/keep/drop rows are
-- ABOUT a relation at all, and seed/error/hub-skip rows have none to state.
DO $relation_check$ BEGIN PERFORM public.ensure_vocabulary_check(
    'citation.crawl_step', 'relation', 'crawl_step_relation_check',
    ARRAY['cites', 'referenced']);
END $relation_check$;

-- kind: what a node in the graph IS to this corpus. Here rather than inline
-- on the column for the reason action and relation are here: citation.work
-- exists on every instance this schema is ever applied to, so an inline
-- CHECK is a constraint that can never be corrected. Left inline it was a
-- shared mechanism deployed to half the vocabularies it exists for, and the
-- half it skipped failed in the worst place: adding a WorkKind and its SQL
-- literal would have passed every offline test (they read the schema FILE)
-- and been a silent no-op on the database the crawl and the packager run
-- against, surfacing as a CHECK violation mid-COPY -- all-or-nothing, so
-- the whole batch.
--
-- citation_vocab.WorkKind is its single Python declaration; the name is the
-- one Postgres gave the inline column CHECK, so an instance created before
-- this file is compared, found equal and left alone.
DO $kind_check$ BEGIN PERFORM public.ensure_vocabulary_check(
    'citation.work', 'kind', 'work_kind_check',
    ARRAY['our-document', 'external-skeleton', 'indexed', 'excluded']);
END $kind_check$;

-- mode: how much of the citation schema a PUBLIC artifact carries. The
-- OWNER writes this column and the packager reads it
-- (CITATION_POLICY_IS_DATA), which is exactly why the vocabulary has to be
-- correctable on a live instance: a mode nobody can record is a decision
-- nobody can take. citation_vocab.PublicPolicyMode is its single Python
-- declaration, and deploy/manifest_contract.CitationMode extends that class
-- rather than restating the values.
DO $mode_check$ BEGIN PERFORM public.ensure_vocabulary_check(
    'citation.public_policy', 'mode', 'public_policy_mode_check',
    ARRAY['full-skeleton', 'topology-only', 'none']);
END $mode_check$;

-- citation.work.document_id: ON DELETE NO ACTION, migrated onto instances
-- created while it was ON DELETE SET NULL.
--
-- SET NULL could not coexist with the table's own CHECK (an our-document
-- row must name its document): the referential UPDATE to NULL violated it
-- immediately, so deleting a corpus document that seeds the graph aborted
-- with a CHECK violation naming a row the deleter never touched. NO ACTION
-- refuses the same delete as what it is -- a referential error naming this
-- constraint -- and the demotion of the graph node is an explicit step
-- (EXTENDING.md procedure A), not a silent side effect that would leave the
-- crawl's provenance quietly nulled.
--
-- CASCADE was the other coherent choice and is the wrong one here: the work
-- row carries the crawl's own record of a node (its key, its edges, the
-- journal rows pointing at it), and losing that because a PDF was reloaded
-- is exactly the class of silent loss LOADERS_PRESERVE exists against.
--
-- 'a' is pg_constraint.confdeltype for NO ACTION. Read and compared rather
-- than dropped-and-re-added blindly: ADD CONSTRAINT on a foreign key
-- re-validates every row.
DO $document_fk$
DECLARE
    existing  text;
    on_delete "char";
BEGIN
    SELECT c.conname, c.confdeltype INTO existing, on_delete
    FROM pg_constraint c
    WHERE c.conrelid = 'citation.work'::regclass AND c.contype = 'f'
      AND c.confrelid = 'corpus.documents'::regclass;

    IF existing IS NULL OR on_delete = 'a' THEN
        RETURN;
    END IF;

    EXECUTE format('ALTER TABLE citation.work DROP CONSTRAINT %I', existing);
    EXECUTE 'ALTER TABLE citation.work ADD CONSTRAINT work_document_id_fkey '
            'FOREIGN KEY (document_id) REFERENCES corpus.documents(id)';
END
$document_fk$;
