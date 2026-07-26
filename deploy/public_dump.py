"""The PUBLIC profile's dump: same schema, legally filtered content.

Why not pg_dump: pg_dump has no row- or cell-level filter, so the only
pg_dump-shaped options were (a) build temporary tables/schema in the live
database and dump those -- which writes to the live instance and renames the
schema the whole toolchain reads (corpus -> something else), or (b) dump
everything and post-process a multi-gigabyte SQL stream with sed. Both are
worse than the third option, which this module takes:

    pg_dump --schema-only --schema=corpus   (real DDL, unmodified)
  + COPY (SELECT ...) TO STDOUT             (the rows we may ship)

The result restores into schema `corpus` exactly as the full artifact does
-- same table names, same generated columns, same indexes -- so every
consumer (pg_search.py, smoke_checks.py, AGENT_GUIDE.md's own recipes) works
against it unchanged. measurements is simply not dumped: our own research
records ship as their own artifact, not folded into a corpus package.

What the filter does, per corpus.documents.public_distribution (see
legal_profile.py -- this module contains no legal knowledge and no id list):

  full-text / internal -- every column, blob included; pages keep body
                          (and therefore tsv, which is GENERATED from it)
  metadata-only        -- the documents row WITHOUT source_blob; pages keep
                          page_number + embedding, and body is written as
                          the empty string

body='' rather than "no page row" is deliberate and load-bearing: the pages
row still exists, so the page's vector is still searchable (semantic search
finds the DOCUMENT, the reader then goes to the publisher's site for the
text), the page count still matches the source, and corpus.pages.body's NOT
NULL survives. Empty body means to_tsvector('russian','') -- an empty tsv,
i.e. no fulltext content, which profile_checks.py verifies rather than
assumes.

Column lists are read from the catalog, not typed here: when the legal
classification columns were added to corpus.documents, a hardcoded list
would have silently dropped legal_class/public_distribution/legal_note from
the public artifact -- the one artifact whose whole point is carrying that
information. Only generated columns (recomputed on restore) and pages.id
(the sequence reassigns it, see PAGES_EXCLUDED) are left out.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from artifact_bundle import DUMP_COMPRESSLEVEL  # noqa: E402
from legal_profile import FULL_CONTENT_SQL, require_classified  # noqa: E402
from pg_common import run_sql  # noqa: E402
from pg_stream import CommandFailed, stream_stdout  # noqa: E402

PUBLIC_SCHEMAS = ("corpus",)

# pages.id is a BIGSERIAL nothing references (probes address a page by
# document_id + page_number). Omitting it from the COPY lets the sequence
# assign ids on restore, in the deterministic order of the ORDER BY below,
# and leaves the sequence itself correctly positioned afterward -- no
# setval() to emit and keep in sync.
PAGES_EXCLUDED = ("id",)

_COLUMNS_SQL = """
SELECT a.attname
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'corpus' AND c.relname = '{table}'
  AND a.attnum > 0 AND NOT a.attisdropped AND a.attgenerated = ''
ORDER BY a.attnum;
"""


# One alias per dumped table, so _select_expression() never has to guess:
# an unlisted table raises KeyError instead of silently producing SQL with
# the wrong alias.
TABLE_ALIASES = {"documents": "d", "pages": "p", "embedding_model": "m"}


def table_columns(env: dict, table: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Non-generated columns of corpus.<table>, in catalog order, minus
    `exclude`. Raises when the table has none -- a typo in a table name
    would otherwise produce a silently empty COPY block.
    """
    rows = run_sql(env, _COLUMNS_SQL.format(table=table), extra_args=["-t", "-A"]).stdout
    columns = [
        line.strip() for line in rows.splitlines()
        if line.strip() and line.strip() not in exclude
    ]
    if not columns:
        raise RuntimeError(f"corpus.{table} has no dumpable columns -- wrong table name?")
    return columns


def _select_expression(table: str, column: str) -> str:
    """The expression written into the COPY select for one column: the
    column itself, except where the legal filter replaces its value.
    """
    alias = TABLE_ALIASES[table]
    if table == "documents" and column == "source_blob":
        return f"CASE WHEN {FULL_CONTENT_SQL} THEN {alias}.source_blob END AS source_blob"
    if table == "pages" and column == "body":
        return f"CASE WHEN {FULL_CONTENT_SQL} THEN {alias}.body ELSE '' END AS body"
    return f"{alias}.{column}"


def _copy_select(table: str, columns: list[str]) -> str:
    projection = ",\n       ".join(_select_expression(table, c) for c in columns)
    if table == "documents":
        source = "FROM corpus.documents d ORDER BY d.id"
    elif table == "pages":
        # The join supplies public_distribution for the body CASE above: the
        # class lives on the document, the cut applies to its pages.
        source = (
            "FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id "
            "ORDER BY p.document_id, p.page_number"
        )
    else:
        alias = TABLE_ALIASES[table]
        source = f"FROM corpus.{table} {alias} ORDER BY {alias}.id"
    return f"COPY (SELECT {projection}\n{source}) TO STDOUT"


def write_copy_block(env: dict, dst: IO[bytes], table: str, columns: list[str]) -> None:
    """Writes one `COPY corpus.<table> (cols) FROM stdin;` block, streaming
    the server-side COPY TO STDOUT output between the header and the `\\.`
    terminator -- the same shape pg_dump emits, so the result is an ordinary
    psql-restorable dump with no special-case loader.
    """
    header = f"COPY corpus.{table} ({', '.join(columns)}) FROM stdin;\n"
    dst.write(header.encode())
    argv = ["psql", "-v", "ON_ERROR_STOP=1", "--quiet", "--no-psqlrc",
            "-c", _copy_select(table, columns)]
    stream_stdout(argv, env, dst)
    dst.write(b"\\.\n\n")


def _dump_ddl(env: dict, dst: IO[bytes]) -> None:
    schema_args = [f"--schema={name}" for name in PUBLIC_SCHEMAS]
    stream_stdout(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges", "--no-tablespaces",
         *schema_args],
        env, dst,
    )


PREAMBLE = (
    "--\n"
    "-- ortopol knowledge base, PUBLIC profile.\n"
    "--\n"
    "-- Schema: real pg_dump --schema-only output for schema corpus (measurements\n"
    "-- is deliberately absent). Data: COPY blocks filtered by\n"
    "-- corpus.documents.public_distribution -- documents classified\n"
    "-- 'metadata-only' carry no source_blob and their pages carry an empty body\n"
    "-- (and therefore an empty tsv), only page_number and embedding. See\n"
    "-- manifest.json's `legal` block for the classification this build used and\n"
    "-- deploy/public_dump.py for how it was applied.\n"
    "--\n\n"
)


def dump_public(env: dict, gz_path: Path) -> None:
    """Writes the filtered dump to gz_path, gzip-streamed in one pass.

    Refuses to write anything at all while any document lacks a usable
    classification (legal_profile.require_classified): an unclassified
    document must stop the build, not be quietly assigned a default.
    """
    require_classified(env)
    documents_columns = table_columns(env, "documents")
    pages_columns = table_columns(env, "pages", exclude=PAGES_EXCLUDED)
    model_columns = table_columns(env, "embedding_model")
    try:
        with gzip.open(gz_path, "wb", compresslevel=DUMP_COMPRESSLEVEL) as dst:
            dst.write(PREAMBLE.encode())
            _dump_ddl(env, dst)
            dst.write(b"\n")
            # documents first: corpus.pages.document_id is a FK to it.
            write_copy_block(env, dst, "documents", documents_columns)
            write_copy_block(env, dst, "pages", pages_columns)
            write_copy_block(env, dst, "embedding_model", model_columns)
    except CommandFailed as exc:
        gz_path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
