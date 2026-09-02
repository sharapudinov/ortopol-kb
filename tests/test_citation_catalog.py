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
from pg_common import FIELD_SEP


def _catalog(rows) -> mock.Mock:
    """A CompletedProcess-alike carrying (relname, attname) pairs."""
    return mock.Mock(stdout="".join(f"{t}{FIELD_SEP}{c}\n" for t, c in rows))


WORK_AND_CITES = _catalog([("cites", "citing"), ("cites", "cited"),
                           ("work", "id"), ("work", "key"), ("work", "title")])


class TableColumnsTests(unittest.TestCase):
    def test_reads_the_catalog_and_keeps_order(self):
        with mock.patch.object(citation_catalog, "run_sql", return_value=WORK_AND_CITES):
            self.assertEqual(citation_catalog.table_columns({}, "work"), ["id", "key", "title"])

    def test_empty_result_is_an_error(self):
        with mock.patch.object(citation_catalog, "run_sql", return_value=mock.Mock(stdout="")):
            with self.assertRaises(RuntimeError):
                citation_catalog.table_columns({}, "work")

    def test_a_name_the_catalog_does_not_know_is_still_an_error(self):
        """The guard moved to the lookup, so a whole-schema read answers a
        misspelt table the same way a per-table one did.
        """
        with self.assertRaises(RuntimeError) as caught:
            citation_catalog.columns_of({"work": ["id"]}, "wrok")
        self.assertIn("citation.wrok", str(caught.exception))

    def test_query_excludes_generated_columns(self):
        self.assertIn("a.attgenerated = ''", citation_catalog._COLUMNS_SQL)


class WholeSchemaInOneReadTests(unittest.TestCase):
    """pg_attribute holds every table's columns, so both column questions
    are one read each. Asked per table they were one psql process, one temp
    script and one connection per table per question, on a loop that grows
    with the schema.
    """

    def test_the_columns_of_every_table_come_back_grouped(self):
        with mock.patch.object(citation_catalog, "run_sql",
                                return_value=WORK_AND_CITES) as run_mock:
            grouped = citation_catalog.schema_columns({})
        self.assertEqual(grouped, {"cites": ["citing", "cited"],
                                   "work": ["id", "key", "title"]})
        run_mock.assert_called_once()

    def test_the_serial_columns_of_every_table_come_back_grouped(self):
        with mock.patch.object(citation_catalog, "run_sql",
                                return_value=_catalog([("crawl_step", "id"),
                                                       ("work", "id")])) as run_mock:
            self.assertEqual(citation_catalog.schema_serial_columns({}),
                             {"crawl_step": ["id"], "work": ["id"]})
        run_mock.assert_called_once()

    def test_neither_query_names_one_table(self):
        for sql in (citation_catalog._COLUMNS_SQL, citation_catalog._SERIAL_COLUMNS_SQL):
            self.assertNotIn("{table}", sql)
            self.assertIn("c.relname", sql)


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
