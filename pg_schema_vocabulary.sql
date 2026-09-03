-- The vocabulary migrator: one function, no schema of its own.
--
-- A closed vocabulary is declared to the database as a NAMED constraint,
-- compared before it is replaced -- never as an inline CHECK in CREATE TABLE
-- IF NOT EXISTS, which changes nothing on an instance that already carries
-- the table and therefore can never be widened again (kb/CLAUDE.md
-- VOCABULARY_ONE_DECLARATION). This is the function that does it, and every
-- vocabulary of the knowledge base goes through it: four on citation tables
-- (pg_schema_citation_constraints.sql) and one on a measurements table whose
-- DDL is built in Python, because a spike's own data table has no schema
-- file (citations/threshold_store.py).
--
-- In `public` -- deliberately, and applied before anything else
-- (pg_graph_common.SCHEMA_PATHS). It used to live in `citation`, which made
-- the measurements schema's DDL depend at runtime on a domain schema with
-- its own lifecycle: `citation` is one the packager ships in three modes
-- including none, and a database can legitimately not carry it at all,
-- while `measurements` is the research schema, versioned and dumped on its
-- own. A tool both of them use belongs to neither. `public` rather than a
-- schema of its own because both dumps are per-schema whitelists
-- (deploy/artifact_bundle, deploy/public_dump): a new schema would be a new
-- name every schema list, manifest and profile check has to learn, for one
-- function that ships in no artifact and is applied by `pg_graph.py init`.
--
-- Idempotent means more than "safe to re-run": ALTER TABLE ... ADD
-- CONSTRAINT validates the new constraint against every existing row under
-- an ACCESS EXCLUSIVE lock, and citation.crawl_step grows by ~100k rows per
-- depth-2 crawl. The function therefore READS what the instance currently
-- has and does nothing when it already agrees.
--
-- Compared as the VOCABULARY, not as text: pg_get_constraintdef() renders
-- the same CHECK as `col = ANY (ARRAY[...])`, and its exact spelling is the
-- server's business (and its version's). The literals inside it are ours,
-- so they are what is compared -- an extra value, a missing one or a
-- renamed one all differ, and nothing else does. Value-idempotent DROP+ADD
-- was never the gap; the gap was paying a full validation scan to arrive at
-- the constraint that was already there.
--
-- One function for all five vocabularies rather than one DO block each: the
-- second copy of a comparison this subtle is the place the next vocabulary
-- gets widened wrongly.
CREATE OR REPLACE FUNCTION public.ensure_vocabulary_check(
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

-- The instance that carries the older copy is migrated, not left with two:
-- a second definition is a second answer to "which comparison decided",
-- and the one in `citation` disappears with its schema. Dropped AFTER the
-- replacement exists, and IF EXISTS covers both the instance that never had
-- it and the one with no citation schema at all (Postgres skips a missing
-- schema here with a notice).
DROP FUNCTION IF EXISTS citation.ensure_vocabulary_check(text, text, text, text[]);
