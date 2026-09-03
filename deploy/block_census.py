"""The one manifest number that is not a row count: a census of ONE column
of ONE COPY block.

Split off copy_rows.py by responsibility (and by kb/CLAUDE.md FILE_SIZE):
that module answers "how many rows went past", which is a question about
every block alike; this one answers "how many of each VALUE", which is a
question about one named block and is meaningless for the rest. Both
counters there take a tally from here and neither owns one.

WHICH block and column that is stays out of both: it is
citation.work.kind, declared in deploy/citation_columns.py because the
bundled verifier re-tallies the same column off the shipped file and only
that module crosses the artifact boundary.
"""
from __future__ import annotations

from typing import NamedTuple


class FieldTally:
    """{value: rows} for ONE column of ONE COPY block, off the same bytes
    the dump is written from.

    COPY's text format escapes every tab inside a value (\\t), so splitting
    a row on tabs is exact rather than a heuristic -- the same property the
    row counter and dump_scan.py both rest on. The column's POSITION comes
    from the block's own header, never from a hand-kept order: a column
    added to the table shifts every index after it.
    """

    def __init__(self, column: str):
        self.column = column
        self.counts: dict[str, int] = {}
        self.index: int | None = None

    def start(self, columns: list[str]) -> None:
        """Binds the tally to a block's column list. A block that does not
        carry the column tallies nothing rather than tallying the wrong
        field -- and a census that stayed empty is one no manifest census
        can equal, which is the direction this has to fail in.
        """
        self.index = columns.index(self.column) if self.column in columns else None

    def line(self, line: bytes) -> None:
        if self.index is None:
            return
        fields = line.split(b"\t")
        if self.index < len(fields):
            value = fields[self.index].decode("utf-8", "replace")
            self.counts[value] = self.counts.get(value, 0) + 1


class BlockCensus(NamedTuple):
    """WHICH block is tallied and WHAT tallies it, as one argument.

    Two parameters would let a caller supply half an answer: a tally with no
    block name counts nothing, a block name with no tally tallies nowhere,
    and both are an empty census stamped into the manifest as fact. `table`
    is qualified the way a COPY header spells it ("<schema>.<table>").
    """

    table: str
    tally: FieldTally
