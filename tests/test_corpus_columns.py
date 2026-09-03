"""Schema corpus's column classification, and the two sides that read it.

deploy/corpus_columns.py is to schema corpus what deploy/citation_columns.py
is to schema citation: every column named topology or content, an
unclassified one a refusal, and ONE map for the packager that cuts and the
checker that travels inside the artifact. The corpus half was the one left
on the old shape -- two hardcoded names in public_dump.py's projection
ending in a ship-by-default fall-through, and the same two names restated as
constants on the artifact side.

Three questions, and each needs a different kind of test:

- is the map complete against the DATABASE (live, below): the catalog reads
  the column list, so the artifact's shape grows with the schema and the
  classification has to grow with it in the same commit;
- do both sides read THAT map (offline): a second copy could only agree with
  the producer by accident, and catching the disagreement is what the
  checker is for;
- does an unclassified column stop the build (offline, in
  test_public_dump.py beside the projection it stops).
"""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import corpus_columns
import corpus_content_checks
import corpus_cut
import schema_catalog
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class ClassificationCoversTheCatalogTests(unittest.TestCase):
    """The map is complete against the DATABASE, not against itself.

    Asserted as set equality both ways, tables and columns alike: something
    in the schema and not in the map is the leak the whole mechanism exists
    to prevent, and something in the map the schema no longer has is a
    classification of nothing, quietly rotting.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_every_table_in_the_schema_is_classified_and_nothing_else_is(self):
        self.assertEqual(
            set(corpus_cut.corpus_tables(self.env)),
            set(corpus_columns.CORPUS_COLUMN_CLASS),
            "каталог схемы corpus и классификация таблиц разошлись",
        )

    def test_every_dumped_column_is_classified_and_nothing_else_is(self):
        """The catalog's own list -- generated columns excluded, since a
        COPY may not carry them (schema_catalog.schema_columns reads
        attgenerated = '').
        """
        columns = schema_catalog.schema_columns(self.env, corpus_cut.SCHEMA)
        for table in corpus_cut.corpus_tables(self.env):
            self.assertEqual(
                set(schema_catalog.columns_of(columns, table, corpus_cut.SCHEMA)),
                set(corpus_columns.CORPUS_COLUMN_CLASS[table]),
                f"corpus.{table}: каталог и классификация разошлись",
            )

    def test_a_column_the_catalog_grew_and_nobody_classified_stops_the_build(self):
        """Asked of the real catalog: the column is added inside a
        transaction that is rolled back, and the refusal names it.
        """
        added = run_sql(
            self.env,
            "BEGIN;\n"
            "ALTER TABLE corpus.documents ADD COLUMN test_unclassified_probe text;\n"
            + schema_catalog._COLUMNS_SQL
            + "ROLLBACK;",
            variables={"schema": corpus_cut.SCHEMA},
            extra_args=["-t", "-A"],
        ).stdout
        self.assertIn("test_unclassified_probe", added)
        with self.assertRaises(corpus_columns.ColumnUnclassified) as caught:
            corpus_cut.copy_select("documents", ["id", "test_unclassified_probe"])
        self.assertIn("corpus.documents.test_unclassified_probe", str(caught.exception))

    def test_the_probe_column_did_not_survive_the_rollback(self):
        columns = schema_catalog.schema_columns(self.env, corpus_cut.SCHEMA)
        self.assertNotIn("test_unclassified_probe",
                         schema_catalog.columns_of(columns, "documents", corpus_cut.SCHEMA))


class BothSidesReadOneMapTests(unittest.TestCase):
    """The producer's cut and the recipient's check, off the same
    declaration -- which is the whole reason the map is a module of its own
    and travels inside the package.
    """

    def test_the_checker_takes_its_content_columns_from_the_map(self):
        self.assertEqual(corpus_content_checks.DOCUMENT_CONTENT,
                         corpus_columns.content_columns("documents"))
        self.assertEqual(corpus_content_checks.PAGE_CONTENT,
                         corpus_columns.content_columns("pages"))

    def test_the_checker_spells_no_content_column_of_its_own(self):
        """The recurrence this catches: a name typed here again is a second
        classification, agreeing with the packager only by attention.
        """
        source = pathlib.Path(corpus_content_checks.__file__).read_text(encoding="utf-8")
        for table, columns in corpus_columns.CORPUS_COLUMN_CLASS.items():
            for column in corpus_columns.content_columns(table):
                with self.subTest(column=column):
                    self.assertNotIn(f'"{column}"', source)
                    self.assertNotIn(f"'{column}'", source)

    def test_a_column_promoted_to_content_is_withheld_by_both_sides_at_once(self):
        """Neither side is edited for it: the packager stops projecting the
        column and the checker starts watching it, from the one change.
        """
        promoted = dict(corpus_columns.CORPUS_COLUMN_CLASS["documents"])
        promoted["note"] = corpus_columns.CONTENT
        withheld = dict(corpus_columns.CONTENT_WITHHELD)
        withheld[("documents", "note")] = "NULL::text"
        with mock.patch.dict(corpus_columns.CORPUS_COLUMN_CLASS,
                             {"documents": promoted}), \
             mock.patch.dict(corpus_columns.CONTENT_WITHHELD, withheld, clear=True):
            self.assertIn("END AS note", corpus_cut.copy_select("documents", ["note"]))
            self.assertIn("note", corpus_columns.content_columns("documents"))

    def test_a_content_column_with_nothing_to_stand_in_for_it_is_a_refusal(self):
        """Half a declaration is a refusal, exactly as no declaration is:
        the build must not guess what a withheld blob looks like.
        """
        promoted = dict(corpus_columns.CORPUS_COLUMN_CLASS["documents"])
        promoted["note"] = corpus_columns.CONTENT
        with mock.patch.dict(corpus_columns.CORPUS_COLUMN_CLASS,
                             {"documents": promoted}):
            with self.assertRaises(corpus_columns.ColumnUnclassified) as caught:
                corpus_cut.copy_select("documents", ["note"])
        self.assertIn("CONTENT_WITHHELD", str(caught.exception))

    def test_the_two_schemas_share_one_engine_rather_than_two_copies(self):
        from column_classes import ColumnClasses

        import citation_columns
        self.assertIsInstance(corpus_columns.CORPUS, ColumnClasses)
        self.assertIsInstance(citation_columns.CITATION, ColumnClasses)
        self.assertEqual(corpus_columns.CORPUS.schema, "corpus")
        self.assertEqual(citation_columns.CITATION.schema, "citation")


if __name__ == "__main__":
    unittest.main()
