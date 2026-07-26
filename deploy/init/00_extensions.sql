-- Runs before the data dump (docker-entrypoint-initdb.d sorts by filename).
-- pgvector/pgvector:pg17 bundles the extension binary but does not activate
-- it; corpus.pages.embedding is vector(1024) and the HNSW index needs the
-- vector_cosine_ops operator class, both defined only after this statement.
CREATE EXTENSION IF NOT EXISTS vector;
