"""How many rows a dump ACTUALLY wrote, counted as it was written.

MANIFEST_DESCRIBES_ARTIFACT: every number in manifest.json is about the
package. The counts got there by asking the live database -- one psql
process for the cut row sets before the file was opened (the public
profile), one for the whole schema after pg_dump finished (the full one) --
and the recipient's gate then demands exact equality with the COPY blocks
the file turns out to contain. Those are two readings of one database with
no shared snapshot between them: every read is its own connection and its
own implicit transaction, and the crawl this repository runs adds ~100k
journal rows per pass against the same live instance the packager reads.
A write landing between the count and the COPY produced a build that
reported success and an artifact that fails its own bundled certification.

The fix is not a bigger transaction, it is not asking twice: the block's
own output IS the count. RowCounter sits between the psql child and the
gzip file and counts the rows going past. COPY's text format escapes every
newline inside a value (\\n, \\t, \\\\), so one row is exactly one line --
the same property dump_scan.py's line-oriented reader rests on -- which
makes the count exact rather than an estimate.

For the full profile there is no per-block seam to sit in: pg_dump writes
the whole file itself, header and terminator included. CopyBlockCounter
sits in the ONE seam that profile does have -- the bytes on their way from
pg_dump's pipe into gzip -- and reads the block structure out of them as
they go past, which is the same polarity and the same exactness, with no
second reading of anything. Read back off the finished file instead
(DumpedRows.from_contents over dump_scan.scan), the same answer costs a
full inflate of a dump that carries every source PDF as hex; that reading
stays, as the independent one verification asks for, and is no longer on
the build's path.
"""
from __future__ import annotations

from typing import IO, NamedTuple

import dump_scan

CORPUS_SCHEMA = "corpus"
CITATION_SCHEMA = "citation"


class RowCounter:
    """A write-through file wrapper that counts the rows passing through it.

    Only write() is needed: pg_stream.stream_stdout copies into its
    destination with shutil.copyfileobj, which calls nothing else.
    """

    def __init__(self, dst: IO[bytes]):
        self.dst = dst
        self.rows = 0

    def write(self, data: bytes) -> int:
        self.rows += data.count(b"\n")
        return self.dst.write(data)


# How much of a line CopyBlockCounter keeps while it is still arriving.
# A COPY header is a table name and its column list -- a few hundred bytes
# -- and a terminator is two, so nothing the state machine has to RECOGNISE
# comes anywhere near this. What does is the data: the full profile's
# documents block carries every source PDF as one hex field on one line,
# hundreds of megabytes of it, and a counter that accumulated each line to
# decide what it was would rebuild the whole dump in memory (and copy the
# growing buffer once per 64KB chunk). Past the cap the rest of the line is
# counted and dropped.
LINE_PREFIX = 8192

COPY_TERMINATOR = dump_scan.COPY_TERMINATOR.encode()


class CopyBlockCounter:
    """A write-through file wrapper that tallies EVERY COPY block passing
    through it: {"<schema>.<table>": rows}.

    RowCounter's answer for a dump nobody assembles block by block. The
    recognition is dump_scan's -- the same COPY_HEADER and the same `\\.`
    terminator the artifact-side reader holds the shipped file to -- so the
    count the manifest is stamped from and the count the recipient's gate
    re-derives are one state machine, run twice over the same bytes rather
    than two readings that agree by habit.

    Only write() is needed while the dump streams (pg_stream.stream_stdout
    copies with shutil.copyfileobj, which calls nothing else); finish()
    closes a last line the child left without a newline.
    """

    def __init__(self, dst: IO[bytes]):
        self.dst = dst
        self.tables: dict[str, int] = {}
        self._prefix = b""
        self._current: str | None = None

    def write(self, data: bytes) -> int:
        written = self.dst.write(data)
        start = 0
        while True:
            cut = data.find(b"\n", start)
            if cut < 0:
                self._extend(data[start:])
                return written
            self._extend(data[start:cut])
            self._line()
            start = cut + 1

    def finish(self) -> None:
        """Ends a trailing line the stream never terminated. pg_dump's own
        output always does; a truncated child's does not, and the rows it
        did write are still rows the file carries.
        """
        if self._prefix:
            self._line()

    def _extend(self, piece: bytes) -> None:
        room = LINE_PREFIX - len(self._prefix)
        if room > 0:
            self._prefix += piece[:room]

    def _line(self) -> None:
        line, self._prefix = self._prefix, b""
        if self._current is None:
            match = dump_scan.COPY_HEADER.match(line.decode("utf-8", "replace"))
            if match:
                schema, table, _columns = match.groups()
                self._current = f"{schema}.{table}"
                self.tables.setdefault(self._current, 0)
            return
        if line == COPY_TERMINATOR:
            self._current = None
            return
        self.tables[self._current] += 1


class DumpedRows(NamedTuple):
    """{table: rows} per schema, as the dump wrote them.

    Two schemas because two parts of the manifest are stamped from it:
    documents_count/pages_count at the top level, and the citation block's
    table_rows with its two headline totals. One shape for both profiles,
    so build_package.py stamps the manifest the same way whichever writer
    produced the file.
    """

    corpus: dict[str, int]
    citation: dict[str, int]

    @classmethod
    def from_blocks(cls, blocks: dict[str, int]) -> "DumpedRows":
        """The same tally out of {"<schema>.<table>": rows} -- what
        CopyBlockCounter holds when the dump has finished streaming.
        """
        per_schema: dict[str, dict[str, int]] = {CORPUS_SCHEMA: {}, CITATION_SCHEMA: {}}
        for qualified, rows in blocks.items():
            schema, _, table = qualified.partition(".")
            if schema in per_schema:
                per_schema[schema][table] = rows
        return cls(corpus=per_schema[CORPUS_SCHEMA], citation=per_schema[CITATION_SCHEMA])

    @classmethod
    def from_contents(cls, contents) -> "DumpedRows":
        """The same tally read back off a FINISHED dump (dump_scan.scan).

        The independent reading: nothing on the build's path calls it any
        more, and that is the point -- it is what a test or a verifier uses
        to ask the file the question the counter answered from the stream.
        """
        return cls.from_blocks({f"{scan.schema}.{scan.table}": scan.rows
                                for scan in contents.tables.values()})
