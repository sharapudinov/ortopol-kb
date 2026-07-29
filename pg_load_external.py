#!/usr/bin/env python3
"""Loads theory/external/ -- literature by OTHER authors -- into corpus.*.

A separate pipeline from pg_load.py, not a flag on it, for three reasons that
are all about not letting one directory's rules leak into the other's:

- the rows it writes carry a legal regime (external-literature / excluded,
  see external_registry.py) that no document of the Sharapudinov corpus has;
- the source tier comes from the registry per source ('arxiv-oa',
  'publisher-paywalled', ...), never the corpus default 'local_corpus';
- a registry row may name a .md bibliography record instead of a PDF: for a
  work whose text we do not hold, the honest holding is the citation, the
  reason it matters, and what we have NOT read.

Both loaders scope their deletes to their own source_dir, so neither can
remove the other's documents -- the failure this project already paid for
twice with transcribed pages and embeddings.

Usage:
    python3 pg_load_external.py [--external-dir DIR] [--pgenv FILE]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

from external_registry import (
    LEGAL_CLASS,
    PUBLIC_DISTRIBUTION,
    REGISTRY_DISTRIBUTION,
    REGISTRY_DOCUMENT_ID,
    REGISTRY_FILENAME,
    REGISTRY_LEGAL_CLASS,
    REGISTRY_NOTE,
    ExternalSource,
    RegistryError,
    load_registry,
    registry_problems,
)
from paths import EXTERNAL_SOURCE_DIR, default_corpus_dir, default_external_dir
from pdf_extract import extract_document
from pg_common import (
    PostgresUnavailable,
    check_postgres_available,
    copy_csv_into,
    load_pgenv,
    run_sql,
    run_sql_file,
    scalar,
)
from report import extraction_state

SCHEMA_PATH = Path(__file__).resolve().parent / "pg_schema.sql"

DOCUMENT_COLUMNS = (
    "id, filename, extraction_state, source_dir, source_tier, pages_count, "
    "chars_extracted, note, source_url, legal_class, public_distribution, legal_note"
)
PAGE_COLUMNS = "document_id, page_number, body"

# A bibliography record: our own text about somebody else's work, one page,
# no extraction involved. Same state as INDEX/THEMES, and protected from the
# ordinary loader by the same predicate.
BIBLIOGRAPHY_STATE = "metadata"

_STAGING_DDL = """
DROP TABLE IF EXISTS corpus.stage_external_documents, corpus.stage_external_pages;
CREATE UNLOGGED TABLE corpus.stage_external_documents (LIKE corpus.documents INCLUDING DEFAULTS);
CREATE UNLOGGED TABLE corpus.stage_external_pages (
    document_id TEXT NOT NULL, page_number INTEGER NOT NULL, body TEXT NOT NULL);
"""

# The embedding clause is the same bargain pg_load.py strikes: keep the vector
# while the page text is byte-identical, void it the moment the text moves.
# The delete scoping is this loader's own: source_dir = the external tree, so
# re-running it can never reach a document of the corpus proper.
_UPSERT_SQL = f"""
BEGIN;

INSERT INTO corpus.documents ({DOCUMENT_COLUMNS})
SELECT {DOCUMENT_COLUMNS} FROM corpus.stage_external_documents
ON CONFLICT (id) DO UPDATE SET
    filename            = EXCLUDED.filename,
    extraction_state    = EXCLUDED.extraction_state,
    source_dir          = EXCLUDED.source_dir,
    source_tier         = EXCLUDED.source_tier,
    pages_count         = EXCLUDED.pages_count,
    chars_extracted     = EXCLUDED.chars_extracted,
    note                = EXCLUDED.note,
    source_url          = EXCLUDED.source_url,
    legal_class         = EXCLUDED.legal_class,
    public_distribution = EXCLUDED.public_distribution,
    legal_note          = EXCLUDED.legal_note
WHERE corpus.documents.source_dir = '{EXTERNAL_SOURCE_DIR}';

INSERT INTO corpus.pages (document_id, page_number, body)
SELECT document_id, page_number, body FROM corpus.stage_external_pages
ON CONFLICT (document_id, page_number) DO UPDATE SET
    body      = EXCLUDED.body,
    embedding = CASE WHEN corpus.pages.body IS DISTINCT FROM EXCLUDED.body
                     THEN NULL ELSE corpus.pages.embedding END;

DELETE FROM corpus.documents d
WHERE d.source_dir = '{EXTERNAL_SOURCE_DIR}'
  AND NOT EXISTS (SELECT 1 FROM corpus.stage_external_documents s WHERE s.id = d.id);
DELETE FROM corpus.pages p
USING corpus.documents d
WHERE d.id = p.document_id
  AND d.source_dir = '{EXTERNAL_SOURCE_DIR}'
  AND NOT EXISTS (SELECT 1 FROM corpus.stage_external_pages s
                  WHERE s.document_id = p.document_id AND s.page_number = p.page_number);

DROP TABLE corpus.stage_external_documents, corpus.stage_external_pages;
COMMIT;
"""


def read_source(source: ExternalSource, directory: Path) -> tuple[str, list[str], str]:
    """(extraction_state, pages, note) for one registry row."""
    path = directory / source.filename
    if source.is_pdf:
        extraction = extract_document(path)
        return (extraction_state(extraction.category),
                extraction.pages,
                f"{source.note} || извлечение: {extraction.note}")
    body = path.read_text(encoding="utf-8")
    return BIBLIOGRAPHY_STATE, [body], f"{source.note} || тела работы у нас нет"


def staged_csv(sources: list[ExternalSource], directory: Path) -> tuple[str, str]:
    documents, pages = io.StringIO(), io.StringIO()
    document_writer = csv.writer(documents, lineterminator="\n")
    page_writer = csv.writer(pages, lineterminator="\n")

    # The registry itself, on the same terms as theory/iis/INDEX.md: our own
    # curated page, searchable by both keys, classified as ours.
    registry_body = (directory / REGISTRY_FILENAME).read_text(encoding="utf-8")
    document_writer.writerow([
        REGISTRY_DOCUMENT_ID, REGISTRY_FILENAME, BIBLIOGRAPHY_STATE,
        EXTERNAL_SOURCE_DIR, "local_corpus", 1, len(registry_body), REGISTRY_NOTE,
        "", REGISTRY_LEGAL_CLASS, REGISTRY_DISTRIBUTION, REGISTRY_NOTE,
    ])
    page_writer.writerow([REGISTRY_DOCUMENT_ID, 1, registry_body])

    for source in sources:
        state, bodies, note = read_source(source, directory)
        document_writer.writerow([
            source.document_id, source.filename, state, EXTERNAL_SOURCE_DIR,
            source.source_tier, len(bodies), sum(len(b) for b in bodies), note,
            source.source_url, LEGAL_CLASS, PUBLIC_DISTRIBUTION, source.legal_note,
        ])
        for page_number, body in enumerate(bodies, start=1):
            page_writer.writerow([source.document_id, page_number, body])
    return documents.getvalue(), pages.getvalue()


def refuse_id_collisions(env: dict, sources: list[ExternalSource]) -> None:
    """An external id that already names a document of another tree stops the
    load. The upsert's own WHERE would merely skip such a row, leaving the
    registry and the database disagreeing in silence.
    """
    if not sources:
        return
    listed = ", ".join(f"'{source.document_id}'" for source in sources)
    clash = scalar(env, f"SELECT string_agg(id, ', ' ORDER BY id) FROM corpus.documents "
                        f"WHERE id IN ({listed}) AND source_dir <> '{EXTERNAL_SOURCE_DIR}';")
    if clash:
        raise RegistryError(
            f"идентификаторы внешних источников заняты документами другого дерева: "
            f"{clash} — переименовать файл в {EXTERNAL_SOURCE_DIR}/")


def load_external(directory: Path, pgenv_path: Path) -> dict:
    sources = load_registry(directory)
    problems = registry_problems(directory, sources)
    if problems:
        raise RegistryError("; ".join(problems))

    env = load_pgenv(pgenv_path)
    if not check_postgres_available(env):
        raise PostgresUnavailable("could not connect to Postgres with the configured credentials")
    run_sql_file(env, SCHEMA_PATH)
    refuse_id_collisions(env, sources)

    documents_csv, pages_csv = staged_csv(sources, directory)
    run_sql(env, _STAGING_DDL)
    copy_csv_into(env, f"corpus.stage_external_documents ({DOCUMENT_COLUMNS})", documents_csv)
    if pages_csv:
        copy_csv_into(env, f"corpus.stage_external_pages ({PAGE_COLUMNS})", pages_csv)
    run_sql(env, _UPSERT_SQL)

    tiers = run_sql(
        env,
        f"SELECT source_tier, extraction_state, count(*), sum(pages_count) "
        f"FROM corpus.documents WHERE source_dir = '{EXTERNAL_SOURCE_DIR}' "
        f"GROUP BY 1, 2 ORDER BY 1, 2;",
    ).stdout.strip()
    return {"sources": len(sources), "tiers": tiers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=None)
    parser.add_argument("--pgenv", type=Path, default=None)
    args = parser.parse_args(argv)
    directory = args.external_dir or default_external_dir()

    try:
        result = load_external(directory, args.pgenv or (default_corpus_dir() / ".pgenv"))
    except (PostgresUnavailable, RegistryError) as exc:
        print(f"внешние источники не загружены: {exc}", file=sys.stderr)
        return 1

    print(f"Загружено внешних источников: {result['sources']} "
          f"({LEGAL_CLASS}, public_distribution={PUBLIC_DISTRIBUTION})")
    print(result["tiers"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
