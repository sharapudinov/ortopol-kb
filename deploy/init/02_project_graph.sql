-- Runs after the data dump is restored (docker-entrypoint-initdb.d sorts
-- init scripts by filename: 00_extensions.sql -> 01_dump.sql.gz -> this).
--
-- The citation graph (schema citation_graph, AGE) is a derived projection,
-- never itself dumped: pg_schema_citation.sql's header explains why
-- (apache/age issue #2503 -- a restored ag_graph.graphid carries the bare
-- oid of the ORIGINAL database, so Cypher against a restored graph is
-- broken). Every artifact that ships the citation schema must therefore
-- rebuild the graph here, once, right after restore.
--
-- Guarded by to_regclass('citation.work') IS NOT NULL rather than assuming
-- the schema exists: this script is bundled unconditionally into every
-- artifact (see artifact_bundle.DEPLOY_FILES), but CitationMode.NONE ships
-- no citation schema at all (deploy/citation_profile.py) and an older
-- an artifact predating the citation schema has no citation.work either -- against either,
-- this must be a silent no-op, not an error.
--
-- LOAD 'age' cannot appear as a bare statement inside a DO block: PL/pgSQL
-- has no grammar for the utility command LOAD, so it must go through
-- EXECUTE like any other dynamically-run SQL text (measured against this
-- image: a bare `LOAD 'age';` line inside DO $$ ... $$ fails to parse;
-- `EXECUTE 'LOAD ''age'''` runs citation.project_graph() successfully,
-- confirmed against the live kb-pg image 2026-09-02). set_config's third
-- argument (true) scopes the search_path change to this one transaction,
-- matching the session-local AGE contract pg_graph.graph_sql()'s clients
-- follow everywhere else in this repository.
DO $$
BEGIN
    IF to_regclass('citation.work') IS NOT NULL THEN
        EXECUTE 'LOAD ''age''';
        PERFORM set_config('search_path', 'ag_catalog,"$user",public', true);
        PERFORM citation.project_graph();
    END IF;
END $$;
