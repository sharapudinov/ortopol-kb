"""What deploy/citation_catalog.py reads out of the catalog, and the order
it derives from it.

The queries are stubbed here; the same functions are asked of the live
schema in test_citation_dump_live.py, which is where "the order really is
what the foreign keys require" is checked against real constraints.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_catalog


class TableColumnsTests(unittest.TestCase):
    def test_reads_the_catalog_and_keeps_order(self):
        with mock.patch.object(citation_catalog, "run_sql",
                                return_value=mock.Mock(stdout="id\nkey\ntitle\n")):
            self.assertEqual(citation_catalog.table_columns({}, "work"), ["id", "key", "title"])

    def test_empty_result_is_an_error(self):
        with mock.patch.object(citation_catalog, "run_sql", return_value=mock.Mock(stdout="")):
            with self.assertRaises(RuntimeError):
                citation_catalog.table_columns({}, "work")

    def test_query_excludes_generated_columns(self):
        self.assertIn("a.attgenerated = ''", citation_catalog._COLUMNS_SQL)


class RestoreOrderTests(unittest.TestCase):
    """The order tables are written in is the foreign keys' answer, not a
    tuple's: a COPY block replays against live constraints, so a table
    follows every table it references. Tables no key relates keep the order
    the catalog gave them.
    """

    def test_a_child_follows_its_parent_however_they_arrive(self):
        self.assertEqual(
            citation_catalog.restore_order(["cites", "work", "crawl_step"],
                                        [("cites", "work")]),
            ["work", "cites", "crawl_step"])

    def test_unrelated_tables_keep_the_catalogs_order(self):
        self.assertEqual(
            citation_catalog.restore_order(["b", "a", "c"], []), ["b", "a", "c"])

    def test_a_chain_is_resolved_to_the_end(self):
        self.assertEqual(
            citation_catalog.restore_order(["c", "b", "a"],
                                        [("c", "b"), ("b", "a")]),
            ["a", "b", "c"])

    def test_a_cycle_is_a_refusal_not_a_guess(self):
        with self.assertRaises(RuntimeError):
            citation_catalog.restore_order(["a", "b"], [("a", "b"), ("b", "a")])


if __name__ == "__main__":
    unittest.main()
