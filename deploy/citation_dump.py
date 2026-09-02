"""The PUBLIC profile's citation-schema slice of the dump: the same pg_dump
--schema-only + hand-rolled COPY pattern public_dump.py uses for schema
corpus (see that module's own docstring for why), applied to schema citation
and governed by citation_profile.py's mode instead of a per-row legal class.

Split into its own module purely for size (kb/CLAUDE.md FILE_SIZE), not for
a different design: public_dump.dump_public() decides WHETHER and under
which mode to call dump_citation(); this module only knows HOW to write one
mode's worth of citation.* DDL+COPY. Like public_dump.py itself, this is
build-time-only and deliberately NOT bundled into the artifact (see
artifact_bundle.DEPLOY_FILES's own comment on legal_profile.py/
public_dump.py) -- it reads and cuts the live database, which is the
builder's job, not the recipient's.

id is preserved (never re-sequenced the way corpus.pages.id is excluded and
left to the restore-side sequence): citation.cites references
citation.work.id BY VALUE, and a column-list COPY has no id-remapping
mechanism, so both the id and the BIGSERIAL sequence position must survive
the round trip -- setval() is appended after the work/crawl_step COPY
blocks for that reason (cites/public_policy carry no serial column of their
own to fix up).
"""
from __future__ import annotations

from typing import IO

from manifest_contract import CitationMode
from pg_common import run_sql
from pg_stream import stream_stdout

CITATION_TABLES = ("work", "cites", "crawl_step", "public_policy")

_ORDER_BY = {
    "work": "id",
    "cites": "citing, cited, source",
    "crawl_step": "id",
    "public_policy": "id",
}

# Tables carrying a BIGSERIAL id whose sequence must be advanced past every
# copied value, or a crawl continued on the restored artifact could try to
# reuse an id already taken.
_SERIAL_TABLES = ("work", "crawl_step")

# Columns forced NULL under CitationMode.TOPOLOGY_ONLY, with the cast an
# untyped NULL in a COPY select's column list would otherwise need Postgres
# to guess (and some psql builds refuse to restore).
_BLANKED = {
    "work": {"abstract": "text", "evidence": "jsonb"},
    "cites": {"evidence": "jsonb"},
}

_COLUMNS_SQL = """
SELECT a.attname
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'citation' AND c.relname = '{table}'
  AND a.attnum > 0 AND NOT a.attisdropped AND a.attgenerated = ''
ORDER BY a.attnum;
"""


def table_columns(env: dict, table: str) -> list[str]:
    rows = run_sql(env, _COLUMNS_SQL.format(table=table), extra_args=["-t", "-A"]).stdout
    columns = [line.strip() for line in rows.splitlines() if line.strip()]
    if not columns:
        raise RuntimeError(f"citation.{table} has no dumpable columns -- wrong table name?")
    return columns


def _select_expression(table: str, column: str, mode: str) -> str:
    cast = _BLANKED.get(table, {}).get(column) if mode == CitationMode.TOPOLOGY_ONLY else None
    return f"NULL::{cast} AS {column}" if cast else column


def copy_select(table: str, columns: list[str], mode: str) -> str:
    projection = ",\n       ".join(_select_expression(table, c, mode) for c in columns)
    return (f"COPY (SELECT {projection}\n"
            f"FROM citation.{table} ORDER BY {_ORDER_BY[table]}) TO STDOUT")


def _setval_sql(table: str) -> bytes:
    return (
        f"SELECT setval(pg_get_serial_sequence('citation.{table}', 'id'), "
        f"coalesce((SELECT max(id) FROM citation.{table}), 1), "
        f"(SELECT max(id) FROM citation.{table}) IS NOT NULL);\n"
    ).encode()


def write_copy_block(env: dict, dst: IO[bytes], table: str, columns: list[str], mode: str) -> None:
    dst.write(f"COPY citation.{table} ({', '.join(columns)}) FROM stdin;\n".encode())
    argv = ["psql", "-v", "ON_ERROR_STOP=1", "--quiet", "--no-psqlrc",
            "-c", copy_select(table, columns, mode)]
    stream_stdout(argv, env, dst)
    dst.write(b"\\.\n")
    if table in _SERIAL_TABLES:
        dst.write(_setval_sql(table))
    dst.write(b"\n")


def dump_ddl(env: dict, dst: IO[bytes]) -> None:
    stream_stdout(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges", "--no-tablespaces",
         "--schema=citation", "--exclude-schema=citation_graph", "--exclude-schema=ag_catalog"],
        env, dst,
    )


def dump_citation(env: dict, dst: IO[bytes], mode: str) -> None:
    """Writes the citation schema's DDL + every table's COPY block for
    `mode`, or writes nothing at all under CitationMode.NONE -- the caller
    (public_dump.dump_public) decides whether to call this at all; this
    function only ever ships something when called with a shipping mode.
    """
    if mode == CitationMode.NONE:
        return
    dump_ddl(env, dst)
    dst.write(b"\n")
    for table in CITATION_TABLES:
        write_copy_block(env, dst, table, table_columns(env, table), mode)
