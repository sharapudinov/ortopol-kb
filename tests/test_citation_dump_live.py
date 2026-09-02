"""The citation dump against the LIVE catalog and the live database.

Split from test_citation_dump.py, whose statement-shaping tests stub
stream_stdout/run_sql and need no server: everything here asks the database
what schema citation actually holds -- which tables, which columns, which
of them carry a sequence, which foreign keys order the restore -- and runs
the real COPY selects inside a transaction that is rolled back. It skips
(not fails) when Postgres is unreachable.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_catalog
import citation_columns
import citation_dump
import citation_profile
from legal_profile import SHIPPED_SQL
from manifest_contract import CitationMode
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

# What the schema holds today, in the order its foreign keys require.
DUMPED_TABLES = ("work", "cites", "crawl_step", "public_policy", "schema_backfill")


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

    table_columns() reads pg_attribute and citation_tables() reads pg_class,
    so the artifact's shape grows with the schema; the classification has to
    grow with it in the same commit or the build stops. Asserted as set
    equality both ways, tables and columns alike: something in the schema
    and not in the map is the leak this whole mechanism exists to prevent,
    and something in the map the schema no longer has is a classification of
    nothing, quietly rotting.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_every_table_in_the_schema_is_classified_and_nothing_else_is(self):
        self.assertEqual(
            set(citation_dump.citation_tables(self.env)),
            set(citation_columns.CITATION_COLUMN_CLASS),
            "каталог схемы citation и классификация таблиц разошлись",
        )

    def test_every_dumped_column_is_classified_and_nothing_else_is(self):
        for table in citation_dump.citation_tables(self.env):
            self.assertEqual(
                set(citation_catalog.table_columns(self.env, table)),
                set(citation_columns.CITATION_COLUMN_CLASS[table]),
                f"citation.{table}: каталог и классификация разошлись",
            )

    def test_a_table_nobody_classified_stops_the_build(self):
        """Asked of the real catalog: the extra table is created inside a
        transaction that is rolled back, and the refusal names it.

        Without this the failure mode is silent -- pg_dump --schema-only
        emits the new table's DDL, no COPY block follows it, and the
        recipient restores a correctly-created empty table.
        """
        seen = run_sql(
            self.env,
            "BEGIN;\n"
            "CREATE TABLE citation.test_unclassified_probe (id BIGINT);\n"
            + citation_catalog._TABLES_SQL
            + "ROLLBACK;",
            extra_args=["-t", "-A"],
        ).stdout
        present = [line.strip() for line in seen.splitlines() if line.strip()]
        self.assertIn("test_unclassified_probe", present)
        with self.assertRaises(citation_dump.TableUnclassified) as caught:
            citation_dump.classified_tables(present)
        self.assertIn("citation.test_unclassified_probe", str(caught.exception))

    def test_the_probe_table_did_not_survive_the_rollback(self):
        self.assertNotIn("test_unclassified_probe",
                         citation_dump.citation_tables(self.env))


class CatalogDerivedShapeTests(unittest.TestCase):
    """The two per-table facts a restore depends on, read from the catalog
    rather than declared beside the classification.

    A hand-kept list of "which tables carry a serial" is a list that can be
    forgotten: the table ships, the sequence does not, and a crawl continued
    on the restored artifact tries to reuse an id already taken. Same for
    the order: it is what the foreign keys say, and pg_constraint says it.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_the_serial_columns_are_the_ones_with_a_sequence(self):
        serials = {table: citation_catalog.serial_columns(self.env, table)
                   for table in citation_dump.citation_tables(self.env)}
        self.assertEqual({t: c for t, c in serials.items() if c},
                         {"work": ["id"], "crawl_step": ["id"]})

    def test_the_dump_order_is_what_the_foreign_keys_require(self):
        order = citation_dump.citation_tables(self.env)
        self.assertEqual(order, list(DUMPED_TABLES))
        for child, parent in citation_catalog.foreign_key_edges(self.env):
            self.assertLess(order.index(parent), order.index(child),
                            f"{child} восстанавливается раньше {parent}")

    def test_the_foreign_keys_read_are_the_ones_inside_this_schema(self):
        """citation.work references corpus.documents, and that key says
        nothing about the order of THIS dump -- the corpus slice is written
        by public_dump.py before it.
        """
        self.assertEqual(citation_catalog.foreign_key_edges(self.env),
                         [("cites", "work")])


class LiveLegalCutTests(unittest.TestCase):
    """Runs the real COPY selects against the live database inside a
    transaction that is ROLLED BACK, so a fixture carrying an `excluded`
    document exists only for the duration of the statement block -- no test
    row can survive a crash, and corpus.documents is never actually written.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _run(self, select: str, fixture: str) -> list[str]:
        out = run_sql(
            self.env,
            "BEGIN;\n" + fixture + "\n" + select + ";\nROLLBACK;\n",
            extra_args=["-t", "-A"],
        ).stdout
        return [line for line in out.splitlines() if line.strip()]

    def _rows(self, table: str, columns: list[str], fixture: str) -> list[str]:
        return self._run(citation_dump.copy_select(table, columns, CitationMode.FULL_SKELETON),
                         fixture)

    FIXTURE = """
    INSERT INTO corpus.documents (id, filename, extraction_state, legal_class,
                                  public_distribution, legal_note)
    VALUES ('test:cut:excluded', 'x.pdf', 'clean', 'unknown', 'excluded', 'test fixture'),
           ('test:cut:shipped', 'y.pdf', 'clean', 'cc-by', 'full-text', 'test fixture');
    INSERT INTO citation.work (key, title, source, kind, document_id) VALUES
      ('test:cut:a', 'A', 'manual', 'our-document', 'test:cut:excluded'),
      ('test:cut:c', 'C', 'manual', 'our-document', 'test:cut:shipped');
    INSERT INTO citation.work (key, title, source, kind) VALUES
      ('test:cut:b', 'B', 'manual', 'external-skeleton');
    INSERT INTO citation.cites (citing, cited, source)
    SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
    WHERE x.key = 'test:cut:b' AND y.key = 'test:cut:a';
    INSERT INTO citation.cites (citing, cited, source)
    SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
    WHERE x.key = 'test:cut:b' AND y.key = 'test:cut:c';
    INSERT INTO citation.crawl_step (crawl_id, depth, frontier_key, candidate_key,
                                     node_key, action, reason)
    VALUES ('test:cut', 0, 'test:cut:excluded', 'test:cut:a', NULL, 'seed', NULL),
           ('test:cut', 1, 'test:cut:b', 'test:cut:a', 'test:cut:a', 'keep',
            'kept; node=test:cut:a'),
           ('test:cut', 0, 'test:cut:b', 'test:cut:c', NULL, 'keep',
            'twin-of=test:cut:shipped'),
           ('test:cut', 2, 'test:cut:b', NULL, NULL, 'fetch', NULL);
    """

    # A row written AFTER the columns existed: the only place the cut name
    # appears is node_key, and reason is prose that names nothing.
    NODE_KEY_ONLY_FIXTURE = """
    INSERT INTO corpus.documents (id, filename, extraction_state, legal_class,
                                  public_distribution, legal_note)
    VALUES ('test:cut:excluded', 'x.pdf', 'clean', 'unknown', 'excluded', 'test fixture');
    INSERT INTO citation.work (key, title, source, kind, document_id) VALUES
      ('test:cut:a', 'A', 'manual', 'our-document', 'test:cut:excluded');
    INSERT INTO citation.crawl_step (crawl_id, depth, frontier_key, candidate_key,
                                     node_key, action, reason)
    VALUES ('test:cut', 1, 'test:cut:b', 'test:cut:d', 'test:cut:a', 'keep',
            'kept, relation=cites'),
           ('test:cut', 1, 'test:cut:b', 'test:cut:e', NULL, 'keep',
            'kept, relation=cites');
    """

    def test_work_row_of_an_excluded_document_does_not_ship(self):
        keys = self._rows("work", ["key"], self.FIXTURE)
        test_keys = sorted(k for k in keys if k.startswith("test:cut:"))
        self.assertEqual(test_keys, ["test:cut:b", "test:cut:c"])

    def test_edges_touching_that_work_do_not_ship(self):
        # Endpoint ids are surrogates, so the fixture's own two edges are
        # counted against the same select run without the fixture: of b->a
        # and b->c only the second may survive.
        baseline = len(self._rows("cites", ["citing", "cited"], ""))
        rows = self._rows("cites", ["citing", "cited"], self.FIXTURE)
        self.assertEqual(len(rows), baseline + 1, "ребро на вырезанный узел вышло в дамп")

    def test_the_cut_set_form_selects_exactly_what_the_per_row_form_did(self):
        """The predicate moved from "re-derive both cut sets for every
        crawl_step row, matching the name as a substring of reason" to "is
        this row's name in either set, by equality on three columns", and
        the sets are now materialised once per statement. Same rows on the
        backfilled journal, or it is not a performance change but a policy
        change -- the reference below is the ORIGINAL substring form.
        """
        # No id: a BIGSERIAL advances even in a rolled-back transaction, so
        # the two runs' fixture rows carry different ones. The comparison is
        # about WHICH rows the predicate lets through.
        columns = ["crawl_id", "depth", "frontier_key", "candidate_key", "reason"]
        projection = ",\n       ".join(f"s.{c}" for c in columns)
        # The naive predicate spelled out here rather than imported: it is
        # the independent reference the fast form is compared against, and
        # an imported one would drift with the thing under test.
        mentions = ("s.frontier_key = {ref} OR s.candidate_key = {ref} "
                    "OR strpos(coalesce(s.reason, ''), {ref}) > 0")
        per_row = (
            "COPY (SELECT " + projection + "\nFROM citation.crawl_step s WHERE "
            "(NOT EXISTS (SELECT 1 FROM corpus.documents d "
            f"WHERE NOT ({SHIPPED_SQL}) AND ("
            + mentions.format(ref="d.id") + ")) "
            "AND NOT EXISTS (SELECT 1 FROM citation.work w "
            f"WHERE NOT {citation_profile.shipped_work_sql('w')} AND ("
            + mentions.format(ref="w.key") + ")))"
            " ORDER BY s.id) TO STDOUT"
        )
        self.assertEqual(self._rows("crawl_step", columns, self.FIXTURE),
                         self._run(per_row, self.FIXTURE))

    def test_a_row_naming_the_cut_work_only_in_node_key_does_not_ship(self):
        """What the substring form could not see: the name is in no prose."""
        rows = self._rows("crawl_step", ["crawl_id", "candidate_key", "node_key"],
                          self.NODE_KEY_ONLY_FIXTURE)
        ours = [r for r in rows if r.split("\t")[0] == "test:cut"]
        self.assertEqual([r.split("\t")[1] for r in ours], ["test:cut:e"], ours)

    def test_journal_rows_naming_the_cut_document_or_work_do_not_ship(self):
        rows = self._rows("crawl_step", ["crawl_id", "depth", "frontier_key",
                                          "candidate_key", "reason"], self.FIXTURE)
        ours = [r for r in rows if r.split("\t")[0] == "test:cut"]
        self.assertEqual(len(ours), 2, f"лишние журнальные строки: {ours}")
        self.assertTrue(all("test:cut:a" not in r for r in ours), ours)
        self.assertTrue(all("test:cut:excluded" not in r for r in ours), ours)


if __name__ == "__main__":
    unittest.main()
