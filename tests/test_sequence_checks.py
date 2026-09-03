"""The statements BETWEEN a dump's COPY blocks, and the one thing they
have to say.

Two subjects, one pass: dump_scan.StatementReader reads sequence ownership
and sequence repositioning off the same lines it reads schema names off,
and sequence_checks.py holds the file to the answer. Both spellings of
setval are exercised -- schema_catalog.setval_sql()'s
(pg_get_serial_sequence, table and column) and pg_dump's own (the sequence
by name, resolved through the ALTER SEQUENCE ... OWNED BY of the DDL
section) -- because both profiles are certified by the same module.
"""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import dump_scan
import sequence_checks

# A miniature public-profile dump: pg_dump's own DDL for the sequence, one
# COPY block, and the setval schema_catalog.setval_sql() writes after it.
OWNED = "ALTER SEQUENCE corpus.pages_id_seq OWNED BY corpus.pages.id;\n"
BLOCK = "COPY corpus.pages (id, body) FROM stdin;\n1\tтекст\n2\tещё\n\\.\n"
SETVAL_BY_COLUMN = (
    "SELECT setval(pg_get_serial_sequence('corpus.pages', 'id'), "
    "coalesce(top.value, 1), top.value IS NOT NULL) "
    "FROM (SELECT max(id) AS value FROM corpus.pages) top;\n")
# ... and pg_dump's own, which is what the full profile carries.
SETVAL_BY_SEQUENCE = "SELECT pg_catalog.setval('corpus.pages_id_seq', 2, true);\n"


def scan_of(text: str) -> dump_scan.DumpContents:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dump.sql.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
        return dump_scan.scan(path)


class StatementReaderTests(unittest.TestCase):
    def test_ownership_is_read_off_the_dumps_own_ddl(self):
        contents = scan_of(OWNED + BLOCK + SETVAL_BY_COLUMN)
        self.assertEqual(contents.sequence_columns, {"corpus.pages.id"})

    def test_both_spellings_of_setval_name_the_same_column(self):
        for name, setval in (("pg_get_serial_sequence", SETVAL_BY_COLUMN),
                             ("pg_dump", SETVAL_BY_SEQUENCE)):
            with self.subTest(spelling=name):
                contents = scan_of(OWNED + BLOCK + setval)
                self.assertEqual(set(contents.sequence_resets), {"corpus.pages.id"})

    def test_a_sequence_named_by_name_needs_the_ownership_line(self):
        """pg_dump's setval says only which SEQUENCE moved. Without the
        OWNED BY the dump also carries, nothing here may guess which column
        that was -- and the check above then reports the column as
        unrepositioned rather than quietly satisfied.
        """
        contents = scan_of(BLOCK + SETVAL_BY_SEQUENCE)
        self.assertEqual(contents.sequence_resets, {})

    def test_the_setval_line_is_placed_against_the_block_it_follows(self):
        contents = scan_of(OWNED + BLOCK + SETVAL_BY_COLUMN)
        self.assertGreater(contents.sequence_resets["corpus.pages.id"],
                           contents.tables["corpus.pages"].ended_at)

    def test_the_one_pass_still_answers_the_schema_question(self):
        contents = scan_of("CREATE SCHEMA corpus;\n" + OWNED + BLOCK + SETVAL_BY_COLUMN)
        self.assertEqual(contents.schemas, {"corpus"})

    def test_the_single_question_helpers_read_the_same_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.sql.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write("CREATE SCHEMA corpus;\n" + OWNED + BLOCK + SETVAL_BY_COLUMN)
            self.assertEqual(dump_scan.sequence_resets(path), {"corpus.pages.id"})
            self.assertEqual(dump_scan.schema_names(path), {"corpus"})


class SequencesAreRepositionedTests(unittest.TestCase):
    def test_a_block_followed_by_its_setval_passes(self):
        ok, detail = sequence_checks.check_sequences_are_repositioned(
            scan_of(OWNED + BLOCK + SETVAL_BY_COLUMN))
        self.assertTrue(ok, detail)
        self.assertIn("corpus.pages.id", detail)

    def test_a_dump_that_forgot_the_setval_fails(self):
        """The whole point: this restores without a complaint, and hands
        the recipient's first insert an id the dump already used.
        """
        ok, detail = sequence_checks.check_sequences_are_repositioned(
            scan_of(OWNED + BLOCK))
        self.assertFalse(ok)
        self.assertIn("corpus.pages.id", detail)
        self.assertIn("setval", detail)

    def test_a_setval_before_its_own_rows_fails(self):
        ok, detail = sequence_checks.check_sequences_are_repositioned(
            scan_of(OWNED + SETVAL_BY_COLUMN + BLOCK))
        self.assertFalse(ok)
        self.assertIn("ДО", detail)

    def test_a_sequence_whose_table_shipped_no_block_is_not_a_breach(self):
        """No COPY block means no rows, so the sequence the recipient
        restores is where a fresh schema starts it. Skipped, and the count
        in the verdict says how many of the declared ones were skipped.
        """
        ok, detail = sequence_checks.check_sequences_are_repositioned(scan_of(OWNED))
        self.assertTrue(ok, detail)
        self.assertIn("0 of 1", detail)


if __name__ == "__main__":
    unittest.main()
