r"""deploy/copy_row.py: one COPY line, read by column.

The reader's whole contract with the checks lives here -- \N, the empty
string and a value are three different answers, a column the block does not
carry is a fourth -- and it is now answered off offsets rather than off a
dict of copied fields (dump_scan.scan's peak memory, measured in
test_dump_scan.PeakMemoryTests). The boundaries below are the ones that
distinguish those four: they used to be dict semantics for free.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

from copy_row import NULL_FIELD, Row, line_end

COLUMNS = ["id", "body", "blob"]


def _row(line: str) -> Row:
    return Row(COLUMNS, line, line_end(line))


class LineEndTests(unittest.TestCase):
    def test_a_newline_is_excluded_without_being_stripped(self):
        self.assertEqual(line_end("a\tb\n"), 3)

    def test_a_last_line_without_one_ends_where_it_ends(self):
        self.assertEqual(line_end("a\tb"), 3)


class FieldReadingTests(unittest.TestCase):
    def test_every_field_is_returned_raw(self):
        row = _row("1997_sm280\tтекст\t\\N\n")
        self.assertEqual(row["id"], "1997_sm280")
        self.assertEqual(row["body"], "текст")
        self.assertEqual(row["blob"], NULL_FIELD)

    def test_the_last_field_stops_before_the_newline(self):
        self.assertEqual(_row("a\tb\tcd\n")["blob"], "cd")
        self.assertEqual(_row("a\tb\tcd")["blob"], "cd")

    def test_a_column_the_block_does_not_carry_is_absent(self):
        row = _row("a\tb\tc\n")
        self.assertNotIn("embedding", row)
        self.assertIsNone(row.get("embedding"))
        self.assertEqual(row.get("embedding", "?"), "?")
        with self.assertRaises(KeyError):
            row["embedding"]

    def test_a_short_row_reports_its_missing_tail_as_absent(self):
        """The scan refuses such a row (the field count is checked before a
        visitor sees it), so this is the fail-closed answer underneath that
        refusal rather than a shape anyone relies on.
        """
        row = _row("a\tb\n")
        self.assertNotIn("blob", row)
        self.assertTrue(row.is_blank("blob"))


class IsBlankTests(unittest.TestCase):
    r"""\N, empty and absent are all "carries nothing"; anything else is
    content, and the answer must not depend on how big that content is."""

    def test_null_and_empty_are_blank(self):
        self.assertTrue(_row("a\t\t\\N\n").is_blank("body"))
        self.assertTrue(_row("a\t\t\\N\n").is_blank("blob"))

    def test_a_value_is_not_blank(self):
        self.assertFalse(_row("a\tx\t0102\n").is_blank("body"))
        self.assertFalse(_row("a\tx\t0102\n").is_blank("blob"))

    def test_a_value_the_length_of_the_null_marker_is_not_blank(self):
        """The comparison is the two characters, not their number: a
        two-character field is exactly the near miss a length test passes.
        """
        self.assertFalse(_row("a\tab\tcd\n").is_blank("body"))

    def test_an_absent_column_carries_nothing(self):
        self.assertTrue(_row("a\tb\tc\n").is_blank("embedding"))


if __name__ == "__main__":
    unittest.main()
