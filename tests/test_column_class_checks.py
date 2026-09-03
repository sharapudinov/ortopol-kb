"""The artifact side of the classification polarity: an unnamed column in
the shipped bytes fails the certification.

Everything here runs a REAL dump_scan pass over hand-built COPY blocks, the
discipline test_profile_checks.py's DumpScanTests set: the check exists to
be right about actual bytes, and a mocked scan would test the mock. The
happy path goes through profile_checks.run_checks() over the ordinary
artifact fixture, so the check is asserted where a recipient runs it.
"""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import column_class_checks
import dump_scan
import profile_checks
from _artifact_fixtures import ArtifactBuilder, DOCUMENT_COLUMNS, PAGE_COLUMNS
from citation_columns import CITATION
from column_classes import CONTENT, TOPOLOGY, ColumnClasses, ColumnUnclassified
from corpus_columns import CORPUS

CHECK_NAME = "каждая колонка дампа классифицирована"


def _copy_block(table: str, columns: list[str], rows: list[list[str]]) -> str:
    lines = [f"COPY {table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


def _scan(dump_text: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        return dump_scan.scan(dump_path).tables


class EngineTests(unittest.TestCase):
    """The predicate both schemas inherit, asked of the engine itself."""

    CLASSES = ColumnClasses(
        "probe", {"t": {"id": TOPOLOGY, "body": CONTENT}}, {("t", "body"): "''"},
        hint="дополните карту", withheld_hint="дополните замены",
    )

    def test_a_named_column_is_known(self):
        self.assertEqual(self.CLASSES.unknown_columns("t", ["id", "body"]), ())

    def test_an_unnamed_column_is_reported(self):
        self.assertEqual(self.CLASSES.unknown_columns("t", ["id", "sneaked"]), ("sneaked",))

    def test_a_table_absent_from_the_map_is_unclassified_whole(self):
        self.assertEqual(self.CLASSES.unknown_columns("other", ["id"]), ("id",))

    def test_content_columns_refuses_an_unclassified_table(self):
        """A bare KeyError read as a crash; the one answer this map may give
        about something nobody classified is the refusal."""
        with self.assertRaises(ColumnUnclassified):
            self.CLASSES.content_columns("other")


class ShippedColumnsTests(unittest.TestCase):
    def test_the_ordinary_artifact_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            results = {name: (ok, detail)
                       for name, ok, detail in profile_checks.run_checks(directory)}
        ok, detail = results[CHECK_NAME]
        self.assertTrue(ok, detail)
        self.assertIn("нет", detail)

    def test_an_unnamed_corpus_column_fails_the_certification(self):
        """The failure class the invariant names: a column present in the
        dump and in no map at all. It reaches the recipient certified [OK]
        on every row unless somebody compares the two."""
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.page_columns = PAGE_COLUMNS + ["draft_note"]
            builder.pages = [row + ["черновая заметка"] for row in builder.pages]
            directory = builder.write()
            results = {name: (ok, detail)
                       for name, ok, detail in profile_checks.run_checks(directory)}
        ok, detail = results[CHECK_NAME]
        self.assertFalse(ok, detail)
        self.assertIn("corpus.pages.draft_note", detail)

    def test_a_generated_column_is_caught_by_the_same_map(self):
        """tsv is GENERATED, therefore in no classification: the hardcoded
        name in corpus_content_checks.py is a second answer to a question
        the map already answers."""
        scans = _scan(_copy_block("corpus.pages", PAGE_COLUMNS + ["tsv"], []))
        ok, detail = column_class_checks.check_columns_are_classified(scans)
        self.assertFalse(ok, detail)
        self.assertIn("corpus.pages.tsv", detail)

    def test_an_unnamed_citation_column_fails_too(self):
        """Both schemas inherit the predicate from one engine, so neither
        can answer this differently."""
        columns = sorted(CITATION.classes["work"]) + ["private_note"]
        scans = _scan(_copy_block("citation.work", columns, []))
        ok, detail = column_class_checks.check_columns_are_classified(scans)
        self.assertFalse(ok, detail)
        self.assertIn("citation.work.private_note", detail)

    def test_a_table_nobody_classified_fails_whole(self):
        """It gets no row visitor on the scan either, so nothing else in
        the package looks at it at all."""
        scans = _scan(_copy_block("citation.side_table", ["id", "payload"], []))
        ok, detail = column_class_checks.check_columns_are_classified(scans)
        self.assertFalse(ok, detail)
        self.assertIn("citation.side_table", detail)

    def test_an_unclassified_schema_is_not_this_check_s_business(self):
        """The full profile ships measurements; WHICH schemas may travel is
        check_schemas(), one line above."""
        scans = _scan(_copy_block("measurements.run", ["id", "anything"], []))
        ok, detail = column_class_checks.check_columns_are_classified(scans)
        self.assertTrue(ok, detail)

    def test_both_maps_are_the_ones_the_packager_cuts_by(self):
        self.assertEqual(column_class_checks.CLASSIFIED_SCHEMAS,
                         {"citation": CITATION, "corpus": CORPUS})
        self.assertEqual(CORPUS.unknown_columns("documents", DOCUMENT_COLUMNS), ())


if __name__ == "__main__":
    unittest.main()
