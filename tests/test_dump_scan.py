"""Unit tests for deploy/dump_scan.py: the reader that turns the artifact's
gzipped dump into COPY blocks, their column lists and their rows.

Split from test_profile_checks.py for module size (kb/CLAUDE.md FILE_SIZE)
along the seam the two modules already have: everything here asks what the
READER makes of a file, everything there asks what the CHECKS make of the
reading. Both build a real gzipped dump plus a real manifest.json in a temp
directory (_artifact_fixtures.ArtifactBuilder), which is the same input a
recipient has.
"""
from __future__ import annotations

import gzip
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import copy_row
import dump_scan
from _artifact_fixtures import ArtifactBuilder, DOCUMENT_COLUMNS, FULL_DOC


class DumpScanTests(unittest.TestCase):
    def test_counts_rows_and_names_the_columns_of_each_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            contents = dump_scan.scan(directory / "01_dump.sql.gz")
        scans = contents.tables
        documents = scans["corpus.documents"]
        self.assertEqual(documents.rows, 3)
        self.assertEqual(documents.columns, DOCUMENT_COLUMNS)
        self.assertEqual(scans["corpus.pages"].rows, 4)

    def test_a_visitor_is_how_a_caller_asks_what_a_column_held(self):
        """Per-column emptiness is a question the ROW VISITORS answer, and
        the only mechanism that answers it: the scan itself keeps no tally
        of its own, so a block nobody registered a visitor for costs one
        line and nothing else.
        """
        empty = {"corpus.documents": 0, "corpus.pages": 0}

        def counter(table: str, column: str):
            def visit(row) -> None:
                if row.is_blank(column):
                    empty[table] += 1
            return visit

        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            dump_scan.scan(directory / "01_dump.sql.gz", {
                "corpus.documents": counter("corpus.documents", "source_blob"),
                "corpus.pages": counter("corpus.pages", "body"),
            })
        self.assertEqual(empty["corpus.documents"], 1)  # the metadata-only one
        self.assertEqual(empty["corpus.pages"], 2)

    def test_schema_names_sees_ddl_and_copy_statements(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.extra_sql = "CREATE TABLE measurements.run (id integer);\n"
            directory = builder.write()
            self.assertEqual(
                dump_scan.schema_names(directory / "01_dump.sql.gz"), {"corpus", "measurements"},
            )

    def test_row_with_the_wrong_field_count_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents.append([FULL_DOC, "only-two-fields"])
            directory = builder.write()
            with self.assertRaises(ValueError):
                dump_scan.scan(directory / "01_dump.sql.gz")

    def test_truncated_copy_block_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "cut.sql.gz"
            with gzip.open(dump_path, "wt", encoding="utf-8") as f:
                f.write("COPY corpus.pages (document_id) FROM stdin;\n2009_isu34\n")
            with self.assertRaises(ValueError) as ctx:
                dump_scan.scan(dump_path)
        self.assertIn("truncated", str(ctx.exception))


class PeakMemoryTests(unittest.TestCase):
    """A visited block whose rows are hundreds of megabytes costs the line
    and nothing more.

    The full profile's corpus.documents block carries every source PDF as
    one hex field on one line, and corpus_content_checks registers a
    visitor on it unconditionally -- so this is the shape the artifact-side
    certification actually runs over, and the producer's counter
    (copy_rows.CopyBlockCounter) caps its retained prefix precisely so it
    never rebuilds such a line.

    Two controls, both spelled out below rather than described: the FLOOR
    (iterating the same file's lines and doing nothing, which is what the
    decompressing reader costs whatever we do) and the OLD SHAPE (rstrip +
    split + dict(zip(...)), which is what scan() used to hand a visitor).
    The claim is that the scan now costs the floor.
    """

    FIELD_BYTES = 50 << 20

    def _dump(self, directory: Path) -> Path:
        dump_path = directory / "big.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8", compresslevel=1) as f:
            f.write("COPY corpus.documents (id, source_blob) FROM stdin;\n")
            f.write("1997_sm280\t" + "ab" * (self.FIELD_BYTES // 2) + "\n")
            f.write("\\.\n")
        return dump_path

    @staticmethod
    def _floor(dump_path: Path) -> None:
        """What decompressing the file costs before anybody looks at a row."""
        with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
            for _raw in f:
                pass

    @staticmethod
    def _old_shape(dump_path: Path, visit) -> None:
        """What the reader did before: a copy of the line without its
        newline, then a copy of every field, then a dict of them."""
        with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("COPY ") or line == copy_row.COPY_TERMINATOR:
                    continue
                fields = line.split("\t")
                visit(dict(zip(["id", "source_blob"], fields)))

    @staticmethod
    def _peak(work) -> int:
        tracemalloc.start()
        try:
            work()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_a_presence_check_over_a_blob_row_copies_nothing(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = self._dump(Path(tmp))

            def visit(row):
                seen.append((row["id"], row.is_blank("source_blob")))

            floor = self._peak(lambda: self._floor(dump_path))
            now = self._peak(lambda: dump_scan.scan(dump_path,
                                                    {"corpus.documents": visit}))
            before = self._peak(
                lambda: self._old_shape(dump_path,
                                        lambda row: row["source_blob"] == ""))
        self.assertEqual(seen, [("1997_sm280", False)])
        # Measured on this machine, 50 MB in one field: floor 105.3 MB
        # (2.01x the field -- the reader's decode buffer plus the line),
        # scan 105.3 MB, old shape 157.4 MB (3.00x).
        self.assertLessEqual(now, floor * 1.05,
                             f"проход дороже самого чтения: пол {floor}, стало {now}")
        self.assertGreater(before, now * 1.3,
                           f"пик не упал: было {before}, стало {now}")


if __name__ == "__main__":
    unittest.main()
