r"""COPY's text format, as the artifact-side reader meets it: the NULL
marker, the line that ends a block, and one data line read by column.

Split out of dump_scan.py for module size (kb/CLAUDE.md FILE_SIZE) along
the responsibility the two have between them: that module says what a DUMP
is (which statements matter, which blocks it holds, how the pass runs),
this one says what a LINE is. The wire format is also the part other
modules need on its own -- copy_rows.py recognises the same terminator in
the bytes on their way into gzip, citation_content_checks reads the same
NULL marker -- and the module that owns a name is the module they import it
from.

Nothing here decodes or unescapes: the checks care about "\N vs empty vs
something", which is exactly what this format distinguishes and what any
un-escaping would blur.
"""
from __future__ import annotations

# The line that ends a COPY block. Spelled once, here, because the
# builder's streaming counter (copy_rows.CopyBlockCounter) recognises the
# same block structure in the bytes on their way into gzip, and the two
# readings of one dump must not be two spellings of what a block is.
COPY_TERMINATOR = "\\."
NULL_FIELD = "\\N"


def line_end(line: str) -> int:
    """Where the line's data stops: before its newline, if it has one.

    Returned as an index rather than applied with rstrip(), because on the
    full profile's documents block that strip is a copy of a
    hundreds-of-megabytes line for the sake of one character.
    """
    return len(line) - 1 if line.endswith("\n") else len(line)


class Row:
    r"""One COPY line, read by column WITHOUT copying the line.

    What a visitor gets. `row[column]` still hands back the raw field
    (`\N` and empty distinguished, as the wire format has them), but the
    field is sliced only when it is actually asked for, and `is_blank()`
    answers "\N, empty or absent" from the tab OFFSETS alone -- no slice at
    all, which is the whole point on a documents row carrying a source PDF
    as hex.

    Before, scan() built `line.split("\t")` for every row of every visited
    block and handed the visitor a dict of the pieces, on top of the
    rstrip'd copy of the line the reader had already made. Both checks that
    read a content column only ever ask whether it holds anything
    (corpus_content_checks._carries_content, citation_content_checks'
    content hunt), so the copies bought nothing and cost, at the peak, two
    to three times the largest row -- while the producer's counter
    (copy_rows.CopyBlockCounter) caps retained line prefixes precisely so
    it never rebuilds such a line at all.

    Measured with tracemalloc over one synthetic 50 MB single-field row
    (tests/test_dump_scan.py::PeakMemoryTests, which re-measures all three
    on every run): decompressing the file and touching no row at all peaks
    at 105.3 MB -- 2.01x the field, the reader's decode buffer plus the
    line, and the floor nothing here can go under. The old shape peaked at
    157.4 MB (3.00x). This one peaks at 105.3 MB: the floor exactly, i.e.
    the pass over a visited blob block now costs no more than reading it.

    The trailing newline is not stripped (that alone was a full copy of the
    line): it is excluded by the offsets instead.
    """

    __slots__ = ("columns", "_line", "_end", "_offsets")

    def __init__(self, columns: list[str], line: str, end: int):
        self.columns = columns
        self._line = line
        self._end = end
        self._offsets: list[int] | None = None

    def _bounds(self, column: str) -> tuple[int, int] | None:
        """(start, stop) of `column`'s field, or None when the block does
        not carry that column at all."""
        try:
            index = self.columns.index(column)
        except ValueError:
            return None
        if self._offsets is None:
            offsets, at = [0], 0
            while True:
                cut = self._line.find("\t", at, self._end)
                if cut < 0:
                    break
                offsets.append(cut + 1)
                at = cut + 1
            self._offsets = offsets
        if index >= len(self._offsets):
            return None
        start = self._offsets[index]
        stop = (self._offsets[index + 1] - 1 if index + 1 < len(self._offsets)
                else self._end)
        return start, stop

    def __contains__(self, column: str) -> bool:
        return self._bounds(column) is not None

    def __getitem__(self, column: str) -> str:
        bounds = self._bounds(column)
        if bounds is None:
            raise KeyError(column)
        return self._line[bounds[0]:bounds[1]]

    def get(self, column: str, default=None):
        bounds = self._bounds(column)
        return default if bounds is None else self._line[bounds[0]:bounds[1]]

    def is_blank(self, column: str) -> bool:
        r"""Does this row carry NOTHING in `column`: \N, the empty string,
        or no such column in this block.

        Answered from the offsets, so the value is never materialised --
        which is what makes a presence check over corpus.documents.
        source_blob free rather than a copy of a whole PDF in hex.
        """
        bounds = self._bounds(column)
        if bounds is None:
            return True
        start, stop = bounds
        if start == stop:
            return True
        return stop - start == len(NULL_FIELD) and self._line[start:stop] == NULL_FIELD
