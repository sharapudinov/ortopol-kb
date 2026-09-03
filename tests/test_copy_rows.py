"""The streaming counters themselves: what a dump wrote, counted as it was
written.

Split out of test_artifact_bundle.py (kb/CLAUDE.md FILE_SIZE, and by
responsibility): that module is about assembling the artifact, this one
about the seam both profiles' bytes pass through -- RowCounter for the
public profile's per-block writes, CopyBlockCounter for pg_dump's single
stream, and FieldTally for the one manifest number that is a census rather
than a total. Nothing here needs a database, a dump or a subprocess: the
input is bytes.
"""
from __future__ import annotations

import io
import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import copy_rows
from block_census import BlockCensus, FieldTally
from citation_columns import CENSUS_COLUMN, CENSUS_TABLE
from copy_rows import CITATION_SCHEMA, CopyBlockCounter, RowCounter

CENSUS_BLOCK = f"{CITATION_SCHEMA}.{CENSUS_TABLE}"


def _census() -> BlockCensus:
    return BlockCensus(CENSUS_BLOCK, FieldTally(CENSUS_COLUMN))


class CopyBlockCounterTests(unittest.TestCase):
    """The full profile's counter: pg_dump owns the file, so the block
    structure is read out of the bytes on their way into gzip.
    """

    def test_a_row_split_across_two_writes_is_still_one_row(self):
        """shutil.copyfileobj hands the counter fixed-size chunks, so a
        block header, a terminator and any row can arrive in halves -- and
        the full profile's documents block is one hex field per row, lines
        far longer than any chunk. Counted per chunk instead of per line,
        every one of those would be several rows or none.
        """
        counter = CopyBlockCounter(io.BytesIO())
        for piece in (b"COPY corpus.doc", b"uments (id) FROM stdin;\n1997",
                      b"_sm280\n2009_isu34\n\\", b".\n"):
            counter.write(piece)
        counter.finish()
        self.assertEqual(counter.tables, {"corpus.documents": 2})

    def test_a_line_longer_than_the_kept_prefix_is_one_row_not_a_buffer(self):
        """A source PDF as hex is one line of hundreds of megabytes. The
        counter keeps at most LINE_PREFIX of it -- enough to recognise a
        header or a terminator, and nothing like enough to rebuild the dump
        in memory.
        """
        blob = b"a" * (copy_rows.LINE_PREFIX * 3)
        counter = CopyBlockCounter(io.BytesIO())
        counter.write(b"COPY corpus.documents (id, source_blob) FROM stdin;\n")
        counter.write(b"1997_sm280\t" + blob + b"\n\\.\n")
        counter.finish()
        self.assertEqual(counter.tables, {"corpus.documents": 1})
        self.assertLessEqual(counter._kept, copy_rows.LINE_PREFIX)

    def test_the_kept_prefix_stops_growing_across_chunks_too(self):
        """The cap is on the LINE, not on one write: a hex blob arrives as
        hundreds of 64KiB chunks, and a cap applied per chunk would keep all
        of them.
        """
        counter = CopyBlockCounter(io.BytesIO())
        counter.write(b"COPY corpus.documents (id, source_blob) FROM stdin;\n")
        for _ in range(10):
            counter.write(b"f" * copy_rows.LINE_PREFIX)
            self.assertLessEqual(counter._kept, copy_rows.LINE_PREFIX)
        counter.write(b"\n\\.\n")
        counter.finish()
        self.assertEqual(counter.tables, {"corpus.documents": 1})

    def test_a_trailing_line_with_no_newline_is_still_counted(self):
        counter = CopyBlockCounter(io.BytesIO())
        counter.write(b"COPY citation.work (id) FROM stdin;\n1\n2")
        counter.finish()
        self.assertEqual(counter.tables, {"citation.work": 2})

    def test_the_bytes_reach_the_destination_unchanged(self):
        """The counter is a write-through wrapper: whatever it reads for its
        own answer, the file gets byte for byte.
        """
        payload = (b"COPY citation.work (id) FROM stdin;\n1\n2\n\\.\n"
                   b"COPY corpus.documents (id) FROM stdin;\n1997_sm280\n\\.\n")
        buffer = io.BytesIO()
        counter = CopyBlockCounter(buffer)
        for at in range(0, len(payload), 7):
            counter.write(payload[at:at + 7])
        counter.finish()
        self.assertEqual(buffer.getvalue(), payload)


class CensusOffTheStreamTests(unittest.TestCase):
    """One column of one block, counted by value off the same bytes.

    citation.work.kind is the manifest's only citation number that is not a
    total, and MANIFEST_DESCRIBES_ARTIFACT makes it the dump's own answer
    rather than a live read beside it.
    """

    def _counted(self, *chunks: bytes) -> CopyBlockCounter:
        counter = CopyBlockCounter(io.BytesIO(), _census())
        for chunk in chunks:
            counter.write(chunk)
        counter.finish()
        return counter

    def test_every_kind_is_counted_by_value(self):
        counter = self._counted(
            f"COPY {CENSUS_BLOCK} (id, key, {CENSUS_COLUMN}) FROM stdin;\n".encode(),
            b"1\tW1\tour-document\n2\tW2\texternal\n3\tW3\texternal\n\\.\n")
        self.assertEqual(counter.tables, {CENSUS_BLOCK: 3})
        self.assertEqual(counter.census.tally.counts,
                         {"our-document": 1, "external": 2})

    def test_the_column_is_found_by_position_from_the_block_header(self):
        """A column added to the table shifts every index after it, so the
        position comes from the header the block itself carries.
        """
        counter = self._counted(
            f"COPY {CENSUS_BLOCK} (id, {CENSUS_COLUMN}, key) FROM stdin;\n".encode(),
            b"1\texternal\tW1\n\\.\n")
        self.assertEqual(counter.census.tally.counts, {"external": 1})

    def test_no_other_block_is_tallied(self):
        counter = self._counted(
            f"COPY citation.cites ({CENSUS_COLUMN}) FROM stdin;\n".encode(),
            b"external\n\\.\n",
            f"COPY {CENSUS_BLOCK} (id, {CENSUS_COLUMN}) FROM stdin;\n".encode(),
            b"1\tour-document\n\\.\n")
        self.assertEqual(counter.census.tally.counts, {"our-document": 1})

    def test_a_census_row_spanning_several_writes_keeps_its_fields(self):
        """The row the census is taken from is the one whose length is
        bounded by nothing -- `evidence` grows with every re-sighting and
        the vector renders as 1024 floats of text -- so it arrives in as
        many chunks as pg_dump's pipe hands over, and the field the census
        reads sits AFTER them.
        """
        long_abstract = b"x" * (copy_rows.LINE_PREFIX * 4)
        counter = CopyBlockCounter(io.BytesIO(), _census())
        counter.write(
            f"COPY {CENSUS_BLOCK} (id, abstract, {CENSUS_COLUMN}) FROM stdin;\n".encode())
        counter.write(b"1\t")
        for at in range(0, len(long_abstract), 4096):
            counter.write(long_abstract[at:at + 4096])
        counter.write(b"\tour-document\n\\.\n")
        counter.finish()
        self.assertEqual(counter.tables, {CENSUS_BLOCK: 1})
        self.assertEqual(counter.census.tally.counts, {"our-document": 1})

    def test_a_block_without_the_census_column_tallies_nothing(self):
        """An empty census is one no manifest number can equal, which is
        the direction this has to fail in.
        """
        counter = self._counted(
            f"COPY {CENSUS_BLOCK} (id, key) FROM stdin;\n".encode(),
            b"1\tW1\n\\.\n")
        self.assertEqual(counter.tables, {CENSUS_BLOCK: 1})
        self.assertEqual(counter.census.tally.counts, {})


class RowCounterTests(unittest.TestCase):
    """The public profile's counter: one block at a time, between psql and
    gzip.
    """

    def test_rows_are_newlines_whatever_the_chunking(self):
        counter = RowCounter(io.BytesIO())
        counter.write(b"1\tW1\n2\t")
        counter.write(b"W2\n3\tW3\n")
        self.assertEqual(counter.rows, 3)

    def test_a_tallied_row_split_across_writes_is_rebuilt_once(self):
        tally = FieldTally(CENSUS_COLUMN)
        tally.start(["id", "abstract", CENSUS_COLUMN])
        counter = RowCounter(io.BytesIO(), tally)
        counter.write(b"1\t")
        for _ in range(8):
            counter.write(b"y" * 4096)
        counter.write(b"\tour-document\n2\tshort\texternal\n")
        self.assertEqual(counter.rows, 2)
        self.assertEqual(tally.counts, {"our-document": 1, "external": 1})

    def test_a_block_the_tally_cannot_answer_for_keeps_no_line(self):
        """The tally's column is not in this block's header, so every row
        would be rebuilt for an answer FieldTally.line() returns without
        looking at -- and the blocks a census does not apply to are exactly
        the ones carrying a source PDF per row.
        """
        tally = FieldTally(CENSUS_COLUMN)
        tally.start(["id", "source_blob"])
        counter = RowCounter(io.BytesIO(), tally)
        counter.write(b"1997_sm280\t" + b"ab" * 5000)
        self.assertEqual(counter._partial, [])
        counter.write(b"\n")
        self.assertEqual(counter.rows, 1)
        self.assertEqual(tally.counts, {})

    def test_the_bytes_reach_the_destination_unchanged(self):
        tally = FieldTally(CENSUS_COLUMN)
        tally.start(["id", CENSUS_COLUMN])
        buffer = io.BytesIO()
        counter = RowCounter(buffer, tally)
        payload = b"1\tour-document\n2\texternal\n"
        for at in range(0, len(payload), 3):
            counter.write(payload[at:at + 3])
        self.assertEqual(buffer.getvalue(), payload)
        self.assertEqual(tally.counts, {"our-document": 1, "external": 1})


if __name__ == "__main__":
    unittest.main()
