#!/usr/bin/env python3
"""The statements citations/store.py runs, and the encodings they expect.

Split off store.py by responsibility (and by kb/CLAUDE.md FILE_SIZE): this
module is what the database is TOLD -- the column lists, the staging DDL,
the upserts and the small encoders that turn a node field into something
COPY accepts. store.py is WHO says it: the seam the crawl writes through,
and the two implementations of it.

The staging relations are TEMP and dropped at COMMIT, and each statement
here is written to travel in ONE script with the \\copy that fills it
(pg_common.copy_csv_rows): one session, one transaction. A globally named
table in schema citation was visible to every other connection between the
three psql invocations a write used to take, so two crawls running at once
-- different crawl ids, different levels -- staged into the same relation
and each dropped the other's rows out from under it.

Every upsert ends by counting what it actually wrote (the closing SELECT
that pg_common.CopyResult.accepted() reads): the Writer contract is rows
ACCEPTED, and these statements refuse rows on
purpose -- a self-edge, an edge already known, a key the promote pass finds
no work row for.
"""
from __future__ import annotations

import json


WORK_COLUMNS = (
    "key", "doi", "title", "abstract", "year", "authors", "external_ids",
    "source", "kind", "document_id", "evidence", "embedding",
)
CITES_COLUMNS = ("citing_key", "cited_key", "source", "evidence")
STEP_COLUMNS = (
    "crawl_id", "depth", "frontier_key", "candidate_key", "node_key", "action",
    "n_found", "n_kept", "score", "tau", "relation", "cited_by_count", "reason",
)

WORK_STAGE_DDL = """
CREATE TEMP TABLE stage_work (
    key TEXT, doi TEXT, title TEXT, abstract TEXT, year INTEGER,
    authors JSONB, external_ids JSONB, source TEXT, kind TEXT,
    document_id TEXT, evidence JSONB, embedding vector(1024)) ON COMMIT DROP;
"""

WORK_UPSERT = """
WITH upserted AS (
INSERT INTO citation.work
    (key, doi, title, abstract, year, authors, external_ids, source, kind,
     document_id, evidence, embedding, fetched_at)
SELECT key, doi, title, abstract, year, authors, external_ids, source, kind,
       document_id, evidence, embedding, now()
FROM stage_work
ON CONFLICT (key) DO UPDATE SET
    doi          = COALESCE(EXCLUDED.doi, citation.work.doi),
    title        = COALESCE(EXCLUDED.title, citation.work.title),
    abstract     = COALESCE(EXCLUDED.abstract, citation.work.abstract),
    year         = COALESCE(EXCLUDED.year, citation.work.year),
    authors      = COALESCE(EXCLUDED.authors, citation.work.authors),
    external_ids = COALESCE(EXCLUDED.external_ids, citation.work.external_ids),
    source       = EXCLUDED.source,
    kind         = CASE WHEN citation.work.kind IN ('our-document', 'indexed')
                         AND EXCLUDED.kind = 'external-skeleton'
                        THEN citation.work.kind ELSE EXCLUDED.kind END,
    document_id  = COALESCE(EXCLUDED.document_id, citation.work.document_id),
    evidence     = EXCLUDED.evidence,
    embedding    = COALESCE(EXCLUDED.embedding, citation.work.embedding),
    fetched_at   = now()
RETURNING 1)
SELECT count(*) FROM upserted;
"""

CITES_STAGE_DDL = """
CREATE TEMP TABLE stage_cites (
    citing_key TEXT, cited_key TEXT, source TEXT, evidence JSONB) ON COMMIT DROP;
"""

# a.id <> b.id, not a.key <> b.key: after the twin union two OpenAlex
# records of one work share a node, and a "translation cites original" edge
# would otherwise violate the CHECK (citing <> cited) and abort the batch.
CITES_UPSERT = """
WITH inserted AS (
INSERT INTO citation.cites (citing, cited, source, evidence)
SELECT a.id, b.id, s.source, s.evidence
FROM stage_cites s
JOIN citation.work a ON a.key = s.citing_key
JOIN citation.work b ON b.key = s.cited_key
WHERE a.id <> b.id
ON CONFLICT (citing, cited, source) DO NOTHING
RETURNING 1)
SELECT count(*) FROM inserted;
"""


PROMOTE_COLUMNS = ("key", "document_id", "seed_key", "rule")

PROMOTE_STAGE_DDL = """
CREATE TEMP TABLE stage_twin (
    key TEXT, document_id TEXT, seed_key TEXT, rule TEXT) ON COMMIT DROP;
"""

PROMOTE_UPDATE = """
WITH promoted AS (
UPDATE citation.work w
SET kind = 'our-document',
    document_id = s.document_id,
    evidence = coalesce(w.evidence, '{}'::jsonb)
               || jsonb_build_object('twin_of', s.seed_key, 'twin_rule', s.rule)
FROM stage_twin s
WHERE w.key = s.key
RETURNING 1)
SELECT count(*) FROM promoted;
"""


def json_or_null(value):
    return None if value is None else json.dumps(value, ensure_ascii=False)
