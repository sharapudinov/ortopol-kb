"""Load the extracted corpus (manifest.json + text/*.txt) into Postgres.

Storage step only: build_corpus.py already produced the manifest and does
not depend on this succeeding, and can be re-run without touching Postgres
at all. Talks to Postgres exclusively through the psql CLI — see
pg_common.py for why.

Usage:
    python3 pg_load.py [--corpus-dir DIR] [--pgenv FILE] [--source-dir REL]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from encoding import Category
from paths import IIS_SOURCE_DIR, default_corpus_dir
from pg_common import (
    PostgresUnavailable,
    check_postgres_available,
    copy_csv_into,
    load_pgenv,
    run_sql,
    run_sql_file,
)
from report import extraction_state

SCHEMA_PATH = Path(__file__).resolve().parent / "pg_schema.sql"
DOCUMENT_FIELDS = ("id, filename, extraction_state, source_dir, source_tier, "
                   "pages_count, chars_extracted, note")
DOCUMENT_COLUMNS = f"corpus.documents ({DOCUMENT_FIELDS})"
PAGE_COLUMNS = "corpus.pages (document_id, page_number, body)"


def _read_manifest(corpus_dir: Path) -> dict:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}; run build_corpus.py first")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _documents_csv(manifest: dict, source_dir: str) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for doc in manifest["documents"]:
        writer.writerow(
            [
                doc["id"],
                doc["filename"],
                extraction_state(Category(doc["category"])),
                source_dir,
                "local_corpus",
                doc["pages_count"],
                doc["chars_extracted"],
                doc["note"],
            ]
        )
    return buf.getvalue()


def _pages_csv(manifest: dict, corpus_dir: Path) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for doc in manifest["documents"]:
        if not doc["text_path"]:
            continue
        text = (corpus_dir / doc["text_path"]).read_text(encoding="utf-8")
        for page_number, body in enumerate(text.split("\f"), start=1):
            writer.writerow([doc["id"], page_number, body])
    return buf.getvalue()


_STAGING_DDL = """
DROP TABLE IF EXISTS corpus.stage_documents, corpus.stage_pages;
CREATE UNLOGGED TABLE corpus.stage_documents (LIKE corpus.documents INCLUDING DEFAULTS);
CREATE UNLOGGED TABLE corpus.stage_pages (
    document_id TEXT NOT NULL, page_number INTEGER NOT NULL, body TEXT NOT NULL);
"""

# The `embedding` clause is the point of the whole upsert: keep the vector when the
# page text is byte-identical, void it when the text moved. Anything else either
# throws away a GPU pass or keeps a vector that no longer describes its page.
#
# Every statement is additionally scoped to the source_dir this run extracted:
# theory/external/ is loaded by its own pipeline (pg_load_external.py), and an
# unscoped DELETE here would wipe its documents on every ordinary reload —
# the same class of defect already paid for twice with transcribed pages.
def _upsert_sql(source_dir: str) -> str:
    return f"""
BEGIN;

INSERT INTO corpus.documents ({DOCUMENT_FIELDS})
SELECT {DOCUMENT_FIELDS}
FROM corpus.stage_documents
ON CONFLICT (id) DO UPDATE SET
    filename         = EXCLUDED.filename,
    extraction_state = EXCLUDED.extraction_state,
    source_dir       = EXCLUDED.source_dir,
    source_tier      = EXCLUDED.source_tier,
    pages_count      = EXCLUDED.pages_count,
    chars_extracted  = EXCLUDED.chars_extracted,
    note             = EXCLUDED.note
-- States NOT owned by this pipeline: transcribed (pg_load_transcribed.py),
-- ocr (pg_load_djvu.py), metadata (pg_load_metadata.py). An ordinary reload would
-- otherwise demote a transcribed document back to 'unreadable' (the extractor still
-- cannot read it) or delete djvu/metadata documents outright (they are never in the
-- extractor's manifest). Same failure class was already paid for twice.
WHERE corpus.documents.extraction_state NOT IN ('transcribed', 'ocr', 'metadata')
  AND corpus.documents.source_dir = '{source_dir}';

INSERT INTO corpus.pages (document_id, page_number, body)
SELECT document_id, page_number, body FROM corpus.stage_pages
ON CONFLICT (document_id, page_number) DO UPDATE SET
    body      = EXCLUDED.body,
    embedding = CASE WHEN corpus.pages.body IS DISTINCT FROM EXCLUDED.body
                     THEN NULL ELSE corpus.pages.embedding END;

-- Sources that vanished from the manifest must not linger as phantom rows.
-- BUT transcribed documents belong to a different pipeline (pg_load_transcribed.py):
-- their pages are read from rendered images, never appear in this manifest, and
-- deleting them here would silently undo that work on every ordinary reload —
-- exactly the failure this loader was already fixed for once, with embeddings.
DELETE FROM corpus.documents d
WHERE d.extraction_state NOT IN ('transcribed', 'ocr', 'metadata')
  AND d.source_dir = '{source_dir}'
  AND NOT EXISTS (SELECT 1 FROM corpus.stage_documents s WHERE s.id = d.id);
DELETE FROM corpus.pages p
USING corpus.documents d
WHERE d.id = p.document_id
  AND d.extraction_state NOT IN ('transcribed', 'ocr', 'metadata')
  AND d.source_dir = '{source_dir}'
  AND NOT EXISTS (SELECT 1 FROM corpus.stage_pages s
                  WHERE s.document_id = p.document_id AND s.page_number = p.page_number);

DROP TABLE corpus.stage_documents, corpus.stage_pages;
COMMIT;
"""


def load_corpus(corpus_dir: Path, pgenv_path: Path,
                source_dir: str = IIS_SOURCE_DIR) -> dict:
    manifest = _read_manifest(corpus_dir)
    env = load_pgenv(pgenv_path)
    if not check_postgres_available(env):
        raise PostgresUnavailable("could not connect to Postgres with the configured credentials")

    run_sql_file(env, SCHEMA_PATH)

    # NOT truncate-and-reload. Pages carry an embedding that costs a GPU pass to
    # rebuild, and a loader that silently discards it turns every reload — including
    # a test run — into a corpus that has quietly lost its semantic index.
    # Upsert instead, and drop the vector ONLY where the text actually changed:
    # a stale vector is worse than a missing one, an unchanged vector is free.
    run_sql(env, _STAGING_DDL)
    copy_csv_into(env, f"corpus.stage_documents ({DOCUMENT_FIELDS})",
                  _documents_csv(manifest, source_dir))
    pages_csv = _pages_csv(manifest, corpus_dir)
    if pages_csv:
        copy_csv_into(env, "corpus.stage_pages (document_id, page_number, body)", pages_csv)
    run_sql(env, _upsert_sql(source_dir))

    counts = run_sql(
        env,
        "SELECT extraction_state, count(*) FROM corpus.documents GROUP BY extraction_state ORDER BY extraction_state;",
    )
    return {"documents": len(manifest["documents"]), "state_counts": counts.stdout.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=default_corpus_dir())
    parser.add_argument("--pgenv", type=Path, default=None)
    parser.add_argument("--source-dir", default=IIS_SOURCE_DIR,
                        help="каталог источников этого прогона относительно дерева "
                             "данных; им же ограничены удаления")
    args = parser.parse_args(argv)
    pgenv_path = args.pgenv or (args.corpus_dir / ".pgenv")

    try:
        result = load_corpus(args.corpus_dir, pgenv_path, args.source_dir)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {result['documents']} documents into Postgres")
    print(result["state_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
