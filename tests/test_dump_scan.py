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
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

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
        line split and nothing else.
        """
        empty = {"corpus.documents": 0, "corpus.pages": 0}

        def counter(table: str, column: str):
            def visit(row: dict) -> None:
                if row[column] in (dump_scan.NULL_FIELD, ""):
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


if __name__ == "__main__":
    unittest.main()
