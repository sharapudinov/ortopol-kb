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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
COPY_HEADER = re.compile(rf"^COPY ({_NAME})\.({_NAME}) \(([^)]*)\) FROM stdin;$")
# The line that ends a COPY block. Spelled once here because the builder's
# streaming counter (copy_rows.CopyBlockCounter) recognises the same block
# structure in the bytes on their way into gzip, and the two readings of one
# dump must not be two spellings of what a block is.
COPY_TERMINATOR = "\\."
NULL_FIELD = "\\N"


@dataclass
class TableScan:
    """What one COPY block declared and what its rows looked like."""

    schema: str
    table: str
    columns: list[str]
    rows: int = 0
    # Which line the block's terminator was on, so a statement can be
    # placed before or after it. A setval that ran BEFORE its own COPY
    # block leaves the sequence exactly where it was.
    ended_at: int = 0


@dataclass
class DumpContents:
    """What ONE pass over the dump found: the COPY blocks, and everything
    the STATEMENTS between them say.

    Every answer comes off the same decompression because every one of them
    is read from the same lines. Asked separately they were a full gzip
    inflate each, of a file that carries every source PDF as hex -- each
    re-parsing the very lines the one before had already parsed.

    `sequence_columns` is every "<schema>.<table>.<column>" the dump's own
    DDL declares sequence-owned, and `sequence_resets` maps such a column to
    the line its setval sits on. Read from the file rather than from a list
    of table names on either side of the boundary: which columns own a
    sequence is a fact about the schema the artifact CARRIES, and a
    hand-kept copy of it is silent about exactly the column that was added
    after the copy was written.
    """

    tables: dict[str, TableScan]
    schemas: set[str]
    sequence_columns: set[str]
    sequence_resets: dict[str, int]


CREATE_SCHEMA = re.compile(rf"^CREATE SCHEMA (?:IF NOT EXISTS )?({_NAME});")
QUALIFIED = re.compile(rf"^(?:CREATE TABLE|CREATE SEQUENCE|COPY|ALTER TABLE(?: ONLY)?) "
                       rf"({_NAME})\.")
# Which column a sequence belongs to, in pg_dump's own words. Both dump
# flavours carry it: the public profile's DDL is real pg_dump --schema-only
# output (public_dump.py), and the full profile is pg_dump throughout.
SEQUENCE_OWNER = re.compile(
    rf"^ALTER SEQUENCE ({_NAME})\.({_NAME}) OWNED BY ({_NAME})\.({_NAME})\.({_NAME});")
# The two spellings of "reposition this sequence". schema_catalog.setval_sql()
# names the table and the column (pg_get_serial_sequence), because the
# sequence's NAME is pg_dump's business; pg_dump names the sequence, which
# SEQUENCE_OWNER above is what resolves back to a column.
SETVAL_BY_COLUMN = re.compile(
    rf"^SELECT (?:pg_catalog\.)?setval\(pg_get_serial_sequence\('({_NAME})\.({_NAME})', "
    rf"'({_NAME})'\)")
SETVAL_BY_SEQUENCE = re.compile(
    rf"^SELECT (?:pg_catalog\.)?setval\('({_NAME})\.({_NAME})'")


def _schema_of(line: str) -> str | None:
    """The schema this DDL/COPY line names, if it names one."""
    match = CREATE_SCHEMA.match(line) or QUALIFIED.match(line)
    return match.group(1) if match else None


class StatementReader:
    """What the lines BETWEEN the COPY blocks say.

    The scan used to model data blocks and nothing else, so every fact
    carried by a statement needed a pass of its own -- and the one fact that
    matters most at the recipient's end is a statement: a sequence left
    where the restore found it hands their first insert an id already taken,
    which is a failure that shows up on their side, days later, with no
    error at restore time to point at it.

    Fed one line at a time with the ordinal of that line, so a caller can
    ask not only WHETHER a sequence was repositioned but whether it happened
    after the rows arrived.
    """

    def __init__(self):
        self.schemas: set[str] = set()
        self.sequence_columns: set[str] = set()
        self.sequence_resets: dict[str, int] = {}
        self._owned_by: dict[str, str] = {}

    def read(self, line: str, ordinal: int) -> None:
        schema = _schema_of(line)
        if schema:
            self.schemas.add(schema)
        owner = SEQUENCE_OWNER.match(line)
        if owner:
            self._owned_by[".".join(owner.group(1, 2))] = ".".join(owner.group(3, 4, 5))
            self.sequence_columns.add(".".join(owner.group(3, 4, 5)))
            return
        column = self._repositioned(line)
        if column:
            # The first setval is the one that counts: a second pass over
            # the same sequence can only move it further along, and the
            # question is whether the rows were followed by one at all.
            self.sequence_resets.setdefault(column, ordinal)

    def _repositioned(self, line: str) -> str | None:
        """Which "<schema>.<table>.<column>" this line's setval moves, if
        it is one. A sequence named by name is resolved through the OWNED BY
        the dump declared earlier -- pg_dump writes the ownership in the DDL
        section and the setval in the data section, in that order.
        """
        by_column = SETVAL_BY_COLUMN.match(line)
        if by_column:
            return ".".join(by_column.group(1, 2, 3))
        by_sequence = SETVAL_BY_SEQUENCE.match(line)
        if by_sequence:
            return self._owned_by.get(".".join(by_sequence.group(1, 2)))
        return None


def _decompress_lines(dump_path: Path) -> Iterator[str]:
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def _statements(dump_path: Path) -> StatementReader:
    """The statement half of the pass, for a caller that has only that
    question.

    scan() is the whole reader and the one the verification path goes
    through. This driver runs the SAME StatementReader over the same lines
    without the COPY-block state machine, so there is still exactly one
    place that knows what a setval or a CREATE SCHEMA looks like -- and so
    a question about statements stays answerable on a file whose data
    section is truncated or is simply not the subject.
    """
    reader = StatementReader()
    for ordinal, line in enumerate(_decompress_lines(dump_path)):
        reader.read(line, ordinal)
    return reader


def schema_names(dump_path: Path) -> set[str]:
    """Every schema the dump touches, from its CREATE SCHEMA / CREATE TABLE
    / COPY statements. Used to assert that the public artifact carries no
    measurements schema at all -- not merely no measurements rows.

    One line of convenience over the shared statement reader, for a caller
    that wants only this answer and has no pass of its own. The
    verification path does NOT come through here: profile_checks.py reads
    DumpContents.schemas off the pass it already makes.
    """
    return _statements(dump_path).schemas


def sequence_resets(dump_path: Path) -> set[str]:
    """Every "<schema>.<table>.<column>" whose sequence the dump repositions.

    A COPY block that carries a sequence-owning column, or one that omits it
    and lets the restore-side DEFAULT assign it, both leave a sequence whose
    position the recipient inherits; the setval() is what makes that position
    right, and it is the kind of statement whose absence nothing notices --
    the restore succeeds and the FIRST INSERT afterwards collides.

    Both spellings, so the answer is about the dump rather than about which
    writer produced it. Through the same StatementReader the verification
    pass uses, for the same reason schema_names() is: one reader of what a
    statement says, whatever drives it over the lines.
    """
    return set(_statements(dump_path).sequence_resets)


def scan(
    dump_path: Path,
    row_visitors: dict[str, Callable[[dict[str, str]], None]] | None = None,
) -> DumpContents:
    """One pass over the dump: every COPY block, and every schema it names.

    `tables` is {"<schema>.<table>": TableScan}; everything the statements
    between the blocks say -- the schema names, which columns own a
    sequence, and where each of those sequences was repositioned -- is
    StatementReader's, collected on the same lines because the caller that
    needs them (profile_checks.py) is reading those lines anyway.

    row_visitors maps "<schema>.<table>" to a callable invoked with each row
    as {column: raw COPY field}. Raw on purpose: the checks care about
    "\\N vs empty vs something", which is exactly what the wire format
    distinguishes and what any un-escaping would blur. Rows of tables with
    no visitor are counted but never materialised as dicts, so scanning the
    full profile's blob-carrying documents block stays cheap.
    """
    visitors = row_visitors or {}
    scans: dict[str, TableScan] = {}
    statements = StatementReader()
    current: TableScan | None = None
    visitor: Callable[[dict[str, str]], None] | None = None

    for ordinal, line in enumerate(_decompress_lines(dump_path)):
        if current is None:
            statements.read(line, ordinal)
            match = COPY_HEADER.match(line)
            if not match:
                continue
            schema, table, column_list = match.groups()
            columns = [c.strip() for c in column_list.split(",")]
            key = f"{schema}.{table}"
            current = TableScan(schema=schema, table=table, columns=columns)
            scans[key] = current
            visitor = visitors.get(key)
            continue
        if line == COPY_TERMINATOR:
            current.ended_at = ordinal
            current, visitor = None, None
            continue
        fields = line.split("\t")
        current.rows += 1
        if len(fields) != len(current.columns):
            raise ValueError(
                f"{current.schema}.{current.table} row {current.rows}: "
                f"{len(fields)} field(s) for {len(current.columns)} declared column(s)"
            )
        if visitor is not None:
            visitor(dict(zip(current.columns, fields)))

    if current is not None:
        raise ValueError(
            f"dump ends inside the {current.schema}.{current.table} COPY block "
            "(truncated artifact?)"
        )
    return DumpContents(tables=scans, schemas=statements.schemas,
                        sequence_columns=statements.sequence_columns,
                        sequence_resets=statements.sequence_resets)
