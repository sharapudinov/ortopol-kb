-- Runs before the data dump (docker-entrypoint-initdb.d sorts by filename).
-- pgvector/pgvector:pg17 bundles the extension binary but does not activate
-- it; corpus.pages.embedding is vector(1024) and the HNSW index needs the
-- vector_cosine_ops operator class, both defined only after this statement.
CREATE EXTENSION IF NOT EXISTS vector;

-- ortopol-pg:17-age1.7-pgvector (kb/deploy/pg) bundles Apache AGE the same
-- way -- binary present, not activated. This registers the extension in the
-- catalog (ag_catalog schema, persists); `LOAD 'age'` and
-- `SET search_path = ag_catalog, "$user", public` are session-local per
-- AGE's own README and are issued by each client, not here.
CREATE EXTENSION IF NOT EXISTS age;
