"""What deploy/schema_catalog.py reads out of the catalog, and the order
it derives from it.

The queries are stubbed here; the same functions are asked of the live
schemas in test_citation_dump_live.py, which is where "the order really is
what the foreign keys require" is checked against real constraints.

The schema is an argument in every one of them. It was a literal in four
statements, in a module public_dump.py had been duplicating for `corpus`
since long before -- so the safeguards this module exists for reached only
the newer schema, and every change to the engine had to be made twice.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import schema_catalog
from pg_common import FIELD_SEP


def _catalog(rows) -> mock.Mock:
    """A CompletedProcess-alike carrying (relname, attname) pairs."""
    return mock.Mock(stdout="".join(f"{t}{FIELD_SEP}{c}\n" for t, c in rows))


WORK_AND_CITES = _catalog([("cites", "citing"), ("cites", "cited"),
                           ("work", "id"), ("work", "key"), ("work", "title")])


class ColumnLookupTests(unittest.TestCase):
    def test_a_name_the_catalog_does_not_know_is_an_error_naming_its_schema(self):
        """A COPY block with no columns is not something to write and find
        out about at the recipient's end.
        """
        with self.assertRaises(RuntimeError) as caught:
            schema_catalog.columns_of({"work": ["id"]}, "wrok", "citation")
        self.assertIn("citation.wrok", str(caught.exception))

    def test_the_same_lookup_answers_for_any_schema(self):
        with self.assertRaises(RuntimeError) as caught:
            schema_catalog.columns_of({"documents": ["id"]}, "docments", "corpus")
        self.assertIn("corpus.docments", str(caught.exception))

    def test_excluded_columns_are_dropped_after_the_guard(self):
        self.assertEqual(
            schema_catalog.columns_of({"pages": ["id", "body"]}, "pages", "corpus",
                                      exclude=("id",)),
            ["body"])

    def test_query_excludes_generated_columns(self):
        self.assertIn("a.attgenerated = ''", schema_catalog._COLUMNS_SQL)


class TheSchemaIsAnArgumentTests(unittest.TestCase):
    """Neither dump's schema may be spelled inside the statements: one of
    them was, and the module that read `corpus` the same way was a second
    implementation nobody updated alongside.
    """

    STATEMENTS = ("_TABLES_SQL", "_FOREIGN_KEYS_SQL", "_SERIAL_COLUMNS_SQL", "_COLUMNS_SQL")

    def test_no_statement_names_a_schema(self):
        for name in self.STATEMENTS:
            with self.subTest(statement=name):
                sql = getattr(schema_catalog, name)
                self.assertNotIn("'citation'", sql)
                self.assertNotIn("'corpus'", sql)
                self.assertIn(":'schema'", sql)

    def test_the_schema_travels_as_a_psql_variable(self):
        with mock.patch.object(schema_catalog, "run_sql",
                                return_value=WORK_AND_CITES) as run_mock:
            schema_catalog.schema_columns({}, "corpus")
        self.assertEqual(run_mock.call_args.kwargs["variables"], {"schema": "corpus"})


class WholeSchemaInOneReadTests(unittest.TestCase):
    """pg_attribute holds every table's columns, so both column questions
    are one read each. Asked per table they were one psql process, one temp
    script and one connection per table per question, on a loop that grows
    with the schema.
    """

    def test_the_columns_of_every_table_come_back_grouped(self):
        with mock.patch.object(schema_catalog, "run_sql",
                                return_value=WORK_AND_CITES) as run_mock:
            grouped = schema_catalog.schema_columns({}, "citation")
        self.assertEqual(grouped, {"cites": ["citing", "cited"],
                                   "work": ["id", "key", "title"]})
        run_mock.assert_called_once()

    def test_the_serial_columns_of_every_table_come_back_grouped(self):
        with mock.patch.object(schema_catalog, "run_sql",
                                return_value=_catalog([("crawl_step", "id"),
                                                       ("work", "id")])) as run_mock:
            self.assertEqual(schema_catalog.schema_serial_columns({}, "citation"),
                             {"crawl_step": ["id"], "work": ["id"]})
        run_mock.assert_called_once()

    def test_neither_query_names_one_table(self):
        for sql in (schema_catalog._COLUMNS_SQL, schema_catalog._SERIAL_COLUMNS_SQL):
            self.assertNotIn("{table}", sql)
            self.assertIn("c.relname", sql)

    def test_an_empty_schema_is_a_refusal(self):
        with mock.patch.object(schema_catalog, "run_sql",
                                return_value=mock.Mock(stdout="")):
            with self.assertRaises(RuntimeError) as caught:
                schema_catalog.present_tables({}, "corpus")
        self.assertIn("corpus", str(caught.exception))


class ClassifiedTablesTests(unittest.TestCase):
    """The refusal is generic; the classification is the caller's. Both
    dumps hold their table list to a map they maintain by hand, and the
    catalog is what says whether that map still covers the schema.
    """

    def test_a_table_nobody_classified_stops_the_build(self):
        with self.assertRaises(schema_catalog.TableUnclassified) as caught:
            schema_catalog.classified_tables(["documents", "annotations"],
                                             {"documents": "d"}, "corpus", "дополните X")
        self.assertIn("corpus.annotations", str(caught.exception))
        self.assertIn("дополните X", str(caught.exception))

    def test_a_covered_schema_passes_through_in_catalog_order(self):
        self.assertEqual(
            schema_catalog.classified_tables(["pages", "documents"],
                                             {"documents", "pages"}, "corpus", "hint"),
            ["pages", "documents"])


class RestoreOrderTests(unittest.TestCase):
    """The order tables are written in is the foreign keys' answer, not a
    tuple's: a COPY block replays against live constraints, so a table
    follows every table it references. Tables no key relates keep the order
    the catalog gave them.
    """

    def test_a_child_follows_its_parent_however_they_arrive(self):
        self.assertEqual(
            schema_catalog.restore_order(["cites", "work", "crawl_step"],
                                         [("cites", "work")], "citation"),
            ["work", "cites", "crawl_step"])

    def test_unrelated_tables_keep_the_catalogs_order(self):
        self.assertEqual(
            schema_catalog.restore_order(["b", "a", "c"], [], "citation"), ["b", "a", "c"])

    def test_a_chain_is_resolved_to_the_end(self):
        self.assertEqual(
            schema_catalog.restore_order(["c", "b", "a"],
                                         [("c", "b"), ("b", "a")], "citation"),
            ["a", "b", "c"])

    def test_a_cycle_is_a_refusal_naming_its_schema(self):
        with self.assertRaises(RuntimeError) as caught:
            schema_catalog.restore_order(["a", "b"], [("a", "b"), ("b", "a")], "corpus")
        self.assertIn("schema corpus", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
