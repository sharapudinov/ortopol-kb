r"""Reads back what a dump actually CONTAINS, straight from the gzipped SQL.

profile_checks.py needs to answer "does this artifact carry a source blob
for document X, and page text for its pages" without a database: the whole
point of a static check is that it holds before anyone restores anything,
and that it inspects the shipped bytes rather than the intentions of the
code that produced them.

Both dump flavours are ordinary psql scripts with pg_dump-shaped data
blocks --

    COPY corpus.documents (id, filename, ...) FROM stdin;
    1997_sm280\tmzm...\t...
    \.

-- so one line-oriented scanner covers the full profile (pg_dump's own
output) and the public profile (public_dump.py's assembled equivalent)
alike. Line-oriented parsing is exact here, not a heuristic: COPY's text
format escapes every newline, tab and backslash inside a value (\n, \t,
\\), so one row is always exactly one line and the terminator line `\.`
can never be produced by data.

Nothing is decoded beyond what the checks need (presence/absence, lengths),
and no whole column value is retained: the full dump's documents block
carries every source PDF as one hex field per row, hundreds of megabytes in
total, and this must stay streaming.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
COPY_HEADER = re.compile(rf"^COPY ({_NAME})\.({_NAME}) \(([^)]*)\) FROM stdin;$")
NULL_FIELD = "\\N"


@dataclass
class TableScan:
    """What one COPY block declared and what its rows looked like."""

    schema: str
    table: str
    columns: list[str]
    rows: int = 0
    # column -> number of rows where that column was SQL NULL / empty.
    nulls: dict[str, int] = field(default_factory=dict)


@dataclass
class DumpContents:
    """What ONE pass over the dump found: the COPY blocks, and every schema
    the file names.

    Both answers come off the same decompression because both are read from
    the same lines. Asked separately they were two full gzip inflates of a
    file that carries every source PDF as hex -- the second one re-parsing
    the very COPY headers the first had already parsed.
    """

    tables: dict[str, TableScan]
    schemas: set[str]


CREATE_SCHEMA = re.compile(rf"^CREATE SCHEMA (?:IF NOT EXISTS )?({_NAME});")
QUALIFIED = re.compile(rf"^(?:CREATE TABLE|CREATE SEQUENCE|COPY|ALTER TABLE(?: ONLY)?) "
                       rf"({_NAME})\.")


def _schema_of(line: str) -> str | None:
    """The schema this DDL/COPY line names, if it names one."""
    match = CREATE_SCHEMA.match(line) or QUALIFIED.match(line)
    return match.group(1) if match else None


def _decompress_lines(dump_path: Path) -> Iterator[str]:
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def schema_names(dump_path: Path) -> set[str]:
    """Every schema the dump touches, from its CREATE SCHEMA / CREATE TABLE
    / COPY statements. Used to assert that the public artifact carries no
    measurements schema at all -- not merely no measurements rows.

    A pass of its own, for a caller that wants only this answer. The
    verification path does NOT use it: profile_checks.py reads
    DumpContents.schemas off the pass it already makes.
    """
    return {schema for schema in (_schema_of(line) for line in _decompress_lines(dump_path))
            if schema}


SETVAL = re.compile(
    rf"^SELECT setval\(pg_get_serial_sequence\('({_NAME})\.({_NAME})', '({_NAME})'\)")


def sequence_resets(dump_path: Path) -> set[str]:
    """Every "<schema>.<table>.<column>" whose sequence the dump repositions.

    A COPY block that carries a sequence-owning column, or one that omits it
    and lets the restore-side DEFAULT assign it, both leave a sequence whose
    position the recipient inherits; the setval() is what makes that position
    right, and it is the kind of statement whose absence nothing notices --
    the restore succeeds and the FIRST INSERT afterwards collides. Read here
    so the assertion is about the shipped bytes, like every other question
    this module answers.

    Only the form schema_catalog.setval_sql() writes: pg_dump emits its own,
    keyed by sequence NAME, for the profile it produces whole.
    """
    return {f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            for m in (SETVAL.match(line) for line in _decompress_lines(dump_path)) if m}


def scan(
    dump_path: Path,
    row_visitors: dict[str, Callable[[dict[str, str]], None]] | None = None,
) -> DumpContents:
    """One pass over the dump: every COPY block, and every schema it names.

    `tables` is {"<schema>.<table>": TableScan}; `schemas` is what
    schema_names() answers on a pass of its own, collected here instead
    because the caller that needs both (profile_checks.py) reads the same
    lines for both.

    row_visitors maps "<schema>.<table>" to a callable invoked with each row
    as {column: raw COPY field}. Raw on purpose: the checks care about
    "\\N vs empty vs something", which is exactly what the wire format
    distinguishes and what any un-escaping would blur. Rows of tables with
    no visitor are counted but never materialised as dicts, so scanning the
    full profile's blob-carrying documents block stays cheap.
    """
    visitors = row_visitors or {}
    scans: dict[str, TableScan] = {}
    schemas: set[str] = set()
    current: TableScan | None = None
    visitor: Callable[[dict[str, str]], None] | None = None

    for line in _decompress_lines(dump_path):
        if current is None:
            schema = _schema_of(line)
            if schema:
                schemas.add(schema)
            match = COPY_HEADER.match(line)
            if not match:
                continue
            schema, table, column_list = match.groups()
            columns = [c.strip() for c in column_list.split(",")]
            key = f"{schema}.{table}"
            current = TableScan(schema=schema, table=table, columns=columns,
                                nulls={c: 0 for c in columns})
            scans[key] = current
            visitor = visitors.get(key)
            continue
        if line == "\\.":
            current, visitor = None, None
            continue
        fields = line.split("\t")
        current.rows += 1
        if len(fields) != len(current.columns):
            raise ValueError(
                f"{current.schema}.{current.table} row {current.rows}: "
                f"{len(fields)} field(s) for {len(current.columns)} declared column(s)"
            )
        for column, value in zip(current.columns, fields):
            if value == NULL_FIELD or value == "":
                current.nulls[column] += 1
        if visitor is not None:
            visitor(dict(zip(current.columns, fields)))

    if current is not None:
        raise ValueError(
            f"dump ends inside the {current.schema}.{current.table} COPY block "
            "(truncated artifact?)"
        )
    return DumpContents(tables=scans, schemas=schemas)
