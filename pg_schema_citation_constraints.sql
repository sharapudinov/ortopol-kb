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

-- A closed vocabulary as a NAMED constraint rather than an inline CHECK:
-- the crawl grows new kinds of decision (hub-skip arrived when depth-2
-- turned out to pull >51k citers through a handful of heavily-cited
-- classics), and an inline CHECK cannot be widened on a table that exists.
--
-- Compared as the VOCABULARY, not as text: pg_get_constraintdef() renders
-- the same CHECK as `col = ANY (ARRAY[...])`, and its exact spelling is the
-- server's business (and its version's). The literals inside it are ours,
-- so they are what is compared -- an extra value, a missing one or a
-- renamed one all differ, and nothing else does. Value-idempotent DROP+ADD
-- was never the gap; the gap was paying a full validation scan to arrive at
-- the constraint that was already there.
--
-- One function for two columns rather than one DO block each: the second
-- copy of a comparison this subtle is the place the next vocabulary gets
-- widened wrongly.
CREATE OR REPLACE FUNCTION citation.ensure_vocabulary_check(
    qualified_table text, column_name text, constraint_name text, wanted text[])
RETURNS void LANGUAGE plpgsql AS $ensure_vocabulary$
DECLARE
    definition text;
    current_vocabulary text[];
BEGIN
    SELECT pg_get_constraintdef(c.oid) INTO definition
    FROM pg_constraint c
    WHERE c.conrelid = qualified_table::regclass AND c.conname = constraint_name;

    IF definition IS NOT NULL THEN
        SELECT array_agg(m[1] ORDER BY m[1]) INTO current_vocabulary
        FROM regexp_matches(definition, '''([^'']*)''', 'g') AS m;
    END IF;

    IF current_vocabulary IS NOT DISTINCT FROM
       (SELECT array_agg(value ORDER BY value) FROM unnest(wanted) AS value) THEN
        RETURN;
    END IF;

    EXECUTE format('ALTER TABLE %s DROP CONSTRAINT IF EXISTS %I',
                   qualified_table, constraint_name);
    EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I CHECK (%I IN (%s))',
                   qualified_table, constraint_name, column_name,
                   (SELECT string_agg(quote_literal(value), ', ')
                      FROM unnest(wanted) AS value));
END
$ensure_vocabulary$;

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
DO $action_check$ BEGIN PERFORM citation.ensure_vocabulary_check(
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
DO $relation_check$ BEGIN PERFORM citation.ensure_vocabulary_check(
    'citation.crawl_step', 'relation', 'crawl_step_relation_check',
    ARRAY['cites', 'referenced']);
END $relation_check$;

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
