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
the whole file itself. There the equivalent answer is the dump read back
(dump_scan.scan), which is the same polarity -- the number comes from the
bytes -- and DumpedRows.from_contents() is where that reading turns into
the same shape.
"""
from __future__ import annotations

from typing import IO, NamedTuple

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
    def from_contents(cls, contents) -> "DumpedRows":
        """The same tally read back off a finished dump (dump_scan.scan).

        The full profile's answer: pg_dump owns the whole file, so the only
        place its row counts exist is in the file.
        """
        per_schema: dict[str, dict[str, int]] = {CORPUS_SCHEMA: {}, CITATION_SCHEMA: {}}
        for scan in contents.tables.values():
            if scan.schema in per_schema:
                per_schema[scan.schema][scan.table] = scan.rows
        return cls(corpus=per_schema[CORPUS_SCHEMA], citation=per_schema[CITATION_SCHEMA])
