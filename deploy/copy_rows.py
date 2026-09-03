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

The one number here that is not a row count -- the census of one column of
one named block -- is block_census.py: both counters take a tally, neither
owns one, and which block it belongs to is the schema's own knowledge.
"""
from __future__ import annotations

from typing import IO, NamedTuple

import copy_row
import dump_scan
from block_census import BlockCensus, FieldTally

CORPUS_SCHEMA = "corpus"
CITATION_SCHEMA = "citation"

# WHICH block and column a census is taken of is NOT declared here. This
# module is the schema-agnostic streaming seam -- it counts whatever goes
# past -- and the one census the manifest carries is citation.work.kind,
# whose name has to be the same on both sides of the artifact boundary
# (deploy/citation_columns.CENSUS_TABLE / CENSUS_COLUMN, the module that
# travels). Callers hand the qualified block name in with the tally.


class RowCounter:
    """A write-through file wrapper that counts the rows passing through it,
    and -- when a caller asks for one -- tallies one column as they go.

    Only write() is needed: pg_stream.stream_stdout copies into its
    destination with shutil.copyfileobj, which calls nothing else.

    Without a tally nothing is split: the journal streams ~100k rows per
    depth-2 crawl through this wrapper, and counting newlines in a chunk is
    the whole cost. With one, the chunk is walked line by line and a partial
    line is carried to the next write.

    A row longer than one write() is rebuilt from a LIST of its chunks,
    joined once when the newline arrives. `bytes` concatenation allocates a
    fresh object every time, so `partial += chunk` down a row spanning N
    chunks copies the growing buffer N times -- quadratic in the row's
    length, on the seam every byte of the dump passes through. A
    citation.work row carries `evidence` (a source record per re-sighting)
    and its 1024-float vector as text, so the length is bounded by nothing.
    """

    def __init__(self, dst: IO[bytes], tally: FieldTally | None = None):
        self.dst = dst
        self.rows = 0
        self.tally = tally
        self._partial: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.rows += data.count(b"\n")
        if self.tally is not None:
            self._tally(data)
        return self.dst.write(data)

    def _tally(self, data: bytes) -> None:
        """Hands every completed line to the tally -- unless the block has
        no such column at all.

        A tally whose column the block's header did not carry answers
        nothing (FieldTally.line returns at once), so keeping its rows would
        be a buffer nobody reads: a block of hundreds-of-megabytes rows is
        exactly the one a census does not apply to, and rebuilding its lines
        is the whole cost this wrapper is meant not to pay.
        """
        if self.tally.index is None:
            return
        start = 0
        while True:
            cut = data.find(b"\n", start)
            if cut < 0:
                if start < len(data):
                    self._partial.append(data[start:])
                return
            line = data[start:cut]
            if self._partial:
                self._partial.append(line)
                line = b"".join(self._partial)
                self._partial = []
            self.tally.line(line)
            start = cut + 1


# How much of a line CopyBlockCounter keeps while it is still arriving.
# A COPY header is a table name and its column list -- a few hundred bytes
# -- and a terminator is two, so nothing the state machine has to RECOGNISE
# comes anywhere near this. What does is the data: the full profile's
# documents block carries every source PDF as one hex field on one line,
# hundreds of megabytes of it, and a counter that accumulated each line to
# decide what it was would rebuild the whole dump in memory. Past the cap
# the rest of the line is counted and dropped.
LINE_PREFIX = 8192

COPY_TERMINATOR = copy_row.COPY_TERMINATOR.encode()


class CopyBlockCounter:
    """A write-through file wrapper that tallies EVERY COPY block passing
    through it: {"<schema>.<table>": rows}, and one column of one of them.

    RowCounter's answer for a dump nobody assembles block by block. The
    recognition is dump_scan's -- the same COPY_HEADER and the same `\\.`
    terminator the artifact-side reader holds the shipped file to -- so the
    count the manifest is stamped from and the count the recipient's gate
    re-derives are one state machine, run twice over the same bytes rather
    than two readings that agree by habit.

    `census` is a BlockCensus -- the caller's block name and tally -- in the
    one seam this profile has: the block is recognised by name off the same
    COPY header, and its column list comes from that header.

    Only write() is needed while the dump streams (pg_stream.stream_stdout
    copies with shutil.copyfileobj, which calls nothing else); finish()
    closes a last line the child left without a newline.
    """

    def __init__(self, dst: IO[bytes], census: BlockCensus | None = None):
        self.dst = dst
        self.tables: dict[str, int] = {}
        self.census = census
        self._counting = False
        self._prefix: list[bytes] = []
        self._kept = 0
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
        """Keeps the line while it arrives, capped -- except inside the
        census block, which is kept whole.

        The cap is there for the documents block, one source PDF as hex per
        row; a work row is a bibliographic record and its vector, tens of
        kilobytes at worst, and one of them at a time. Capped, a row whose
        abstract runs past LINE_PREFIX would lose the tab the census field
        is counted from -- a tally silently short by however many rows had
        long abstracts, stamped into the manifest as fact.

        The pieces are KEPT as a list and joined once per line: an
        uncapped `bytes +=` per 64KiB chunk recopies the whole line every
        time, which is quadratic in the length of exactly the rows the cap
        is lifted for.
        """
        if not piece:
            return
        if not self._counting:
            room = LINE_PREFIX - self._kept
            if room <= 0:
                return
            piece = piece[:room]
        self._prefix.append(piece)
        self._kept += len(piece)

    def _line(self) -> None:
        line = b"".join(self._prefix)
        self._prefix, self._kept = [], 0
        if self._current is None:
            match = dump_scan.COPY_HEADER.match(line.decode("utf-8", "replace"))
            if match:
                schema, table, columns = match.groups()
                self._current = f"{schema}.{table}"
                self.tables.setdefault(self._current, 0)
                self._counting = (self.census is not None
                                  and self._current == self.census.table)
                if self._counting:
                    self.census.tally.start([c.strip() for c in columns.split(",")])
            return
        if line == COPY_TERMINATOR:
            self._current, self._counting = None, False
            return
        self.tables[self._current] += 1
        if self._counting:
            self.census.tally.line(line)


class DumpedRows(NamedTuple):
    """What the dump wrote: {table: rows} per schema, and the kind census.

    Two schemas because two parts of the manifest are stamped from it:
    documents_count/pages_count at the top level, and the citation block's
    table_rows with its two headline totals. One shape for both profiles,
    so build_package.py stamps the manifest the same way whichever writer
    produced the file.

    work_by_kind rides along because it is the same kind of answer: a fact
    about the FILE, produced by the writing of it. Read from the live
    database beside the dump it was the one citation number nothing held to
    the bytes, and the crawl writes ~100k journal rows per pass against the
    same instance.
    """

    corpus: dict[str, int]
    citation: dict[str, int]
    work_by_kind: dict[str, int] = {}

    @classmethod
    def from_blocks(cls, blocks: dict[str, int],
                    work_by_kind: dict[str, int] | None = None) -> "DumpedRows":
        """The same tally out of {"<schema>.<table>": rows} -- what
        CopyBlockCounter holds when the dump has finished streaming.
        """
        per_schema: dict[str, dict[str, int]] = {CORPUS_SCHEMA: {}, CITATION_SCHEMA: {}}
        for qualified, rows in blocks.items():
            schema, _, table = qualified.partition(".")
            if schema in per_schema:
                per_schema[schema][table] = rows
        return cls(corpus=per_schema[CORPUS_SCHEMA], citation=per_schema[CITATION_SCHEMA],
                   work_by_kind=dict(work_by_kind or {}))

    @classmethod
    def from_contents(cls, contents) -> "DumpedRows":
        """The same tally read back off a FINISHED dump (dump_scan.scan).

        The independent reading: nothing on the build's path calls it any
        more, and that is the point -- it is what a test or a verifier uses
        to ask the file the question the counter answered from the stream.
        The census is not among its answers: a scan keeps no column values,
        and the artifact-side reader that does collect them is
        citation_content_checks.py, on profile_checks.py's own pass.
        """
        return cls.from_blocks({f"{scan.schema}.{scan.table}": scan.rows
                                for scan in contents.tables.values()})
