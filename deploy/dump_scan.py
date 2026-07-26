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


def _decompress_lines(dump_path: Path) -> Iterator[str]:
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def schema_names(dump_path: Path) -> set[str]:
    """Every schema the dump touches, from its CREATE SCHEMA / CREATE TABLE
    / COPY statements. Used to assert that the public artifact carries no
    measurements schema at all -- not merely no measurements rows.
    """
    found: set[str] = set()
    create_schema = re.compile(rf"^CREATE SCHEMA (?:IF NOT EXISTS )?({_NAME});")
    qualified = re.compile(rf"^(?:CREATE TABLE|CREATE SEQUENCE|COPY|ALTER TABLE(?: ONLY)?) "
                          rf"({_NAME})\.")
    for line in _decompress_lines(dump_path):
        match = create_schema.match(line)
        if match:
            found.add(match.group(1))
            continue
        match = qualified.match(line)
        if match:
            found.add(match.group(1))
    return found


def scan(
    dump_path: Path,
    row_visitors: dict[str, Callable[[dict[str, str]], None]] | None = None,
) -> dict[str, TableScan]:
    """Scans every COPY block, returning {"<schema>.<table>": TableScan}.

    row_visitors maps "<schema>.<table>" to a callable invoked with each row
    as {column: raw COPY field}. Raw on purpose: the checks care about
    "\\N vs empty vs something", which is exactly what the wire format
    distinguishes and what any un-escaping would blur. Rows of tables with
    no visitor are counted but never materialised as dicts, so scanning the
    full profile's blob-carrying documents block stays cheap.
    """
    visitors = row_visitors or {}
    scans: dict[str, TableScan] = {}
    current: TableScan | None = None
    visitor: Callable[[dict[str, str]], None] | None = None

    for line in _decompress_lines(dump_path):
        if current is None:
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
    return scans
