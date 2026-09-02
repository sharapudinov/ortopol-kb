"""Unit tests for deploy/citation_dump.py: no live Postgres, no real
pg_dump/psql (stream_stdout/run_sql are stubbed). Mirrors test_public_dump.py's
style for the corpus-schema equivalent.
"""
from __future__ import annotations

import io
import re
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_columns
import citation_dump
import citation_profile
from legal_profile import SHIPPED_SQL
from manifest_contract import CitationMode
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class TableColumnsTests(unittest.TestCase):
    def test_reads_the_catalog_and_keeps_order(self):
        with mock.patch.object(citation_dump, "run_sql",
                                return_value=mock.Mock(stdout="id\nkey\ntitle\n")):
            self.assertEqual(citation_dump.table_columns({}, "work"), ["id", "key", "title"])

    def test_empty_result_is_an_error(self):
        with mock.patch.object(citation_dump, "run_sql", return_value=mock.Mock(stdout="")):
            with self.assertRaises(RuntimeError):
                citation_dump.table_columns({}, "work")

    def test_query_excludes_generated_columns(self):
        self.assertIn("a.attgenerated = ''", citation_dump._COLUMNS_SQL)


class CopySelectTests(unittest.TestCase):
    def test_full_skeleton_ships_abstract_and_evidence_unmodified(self):
        sql = citation_dump.copy_select("work", ["id", "abstract", "evidence"], CitationMode.FULL_SKELETON)
        self.assertIn("abstract", sql)
        self.assertNotIn("NULL::", sql)

    def test_topology_only_blanks_work_abstract_and_evidence(self):
        sql = citation_dump.copy_select("work", ["id", "abstract", "evidence"], CitationMode.TOPOLOGY_ONLY)
        self.assertIn("NULL::text AS abstract", sql)
        self.assertIn("NULL::jsonb AS evidence", sql)
        # id is never blanked -- cites.citing/cited references it by value.
        self.assertIn("SELECT w.id,", sql)

    def test_topology_only_blanks_cites_evidence(self):
        sql = citation_dump.copy_select("cites", ["citing", "cited", "evidence"], CitationMode.TOPOLOGY_ONLY)
        self.assertIn("NULL::jsonb AS evidence", sql)
        self.assertIn("citing", sql)
        self.assertIn("cited", sql)

    def test_topology_only_blanks_every_column_classified_content(self):
        """Not the two the mode was first written around: every column the
        classification calls content, in every dumped table. The embedding
        was the column this test would have caught -- it is classified
        topology (a vector ranks, it does not reproduce an abstract) and
        must therefore still be in the projection.
        """
        for table, columns in citation_columns.CITATION_COLUMN_CLASS.items():
            sql = citation_dump.copy_select(table, list(columns), CitationMode.TOPOLOGY_ONLY)
            alias = citation_dump.TABLE_ALIASES[table]
            for column, kind in columns.items():
                if kind == citation_columns.CONTENT:
                    self.assertIn(f"AS {column}", sql, f"{table}.{column}")
                    self.assertNotIn(f"{alias}.{column},", sql, f"{table}.{column}")
                else:
                    self.assertIn(f"{alias}.{column}", sql, f"{table}.{column}")

    def test_an_unclassified_column_stops_the_build(self):
        with self.assertRaises(citation_columns.ColumnUnclassified) as raised:
            citation_dump.copy_select("work", ["id", "brand_new"], CitationMode.FULL_SKELETON)
        self.assertIn("citation.work.brand_new", str(raised.exception))

    def test_full_skeleton_blanks_nothing_but_still_classifies_everything(self):
        for table, columns in citation_columns.CITATION_COLUMN_CLASS.items():
            sql = citation_dump.copy_select(table, list(columns), CitationMode.FULL_SKELETON)
            self.assertNotIn("NULL::", sql, table)

    def test_public_policy_is_never_blanked_under_any_mode(self):
        for mode in CitationMode.ALL:
            sql = citation_dump.copy_select("public_policy", ["id", "mode", "note"], mode)
            self.assertNotIn("NULL::", sql)

    def test_order_by_matches_each_table_shape(self):
        self.assertIn("ORDER BY w.id",
                      citation_dump.copy_select("work", ["id"], CitationMode.FULL_SKELETON))
        self.assertIn(
            "ORDER BY c.citing, c.cited, c.source",
            citation_dump.copy_select("cites", ["citing", "cited", "source"], CitationMode.FULL_SKELETON),
        )


class SetvalTests(unittest.TestCase):
    def test_serial_tables_get_a_setval_statement(self):
        sql = citation_dump._setval_sql("work").decode()
        self.assertIn("pg_get_serial_sequence('citation.work', 'id')", sql)
        self.assertIn("coalesce((SELECT max(id) FROM citation.work), 1)", sql)


class WriteCopyBlockTests(unittest.TestCase):
    def test_serial_table_writes_setval_after_the_terminator(self):
        buffer = io.BytesIO()

        def fake_stream(argv, env, dst):
            dst.write(b"1\tk1\n")

        with mock.patch.object(citation_dump, "stream_stdout", side_effect=fake_stream):
            citation_dump.write_copy_block({}, buffer, "work", ["id", "key"], CitationMode.FULL_SKELETON)
        text = buffer.getvalue().decode()
        self.assertIn("COPY citation.work (id, key) FROM stdin;\n1\tk1\n\\.\n", text)
        self.assertIn("setval(pg_get_serial_sequence", text)

    def test_non_serial_table_writes_no_setval(self):
        buffer = io.BytesIO()

        def fake_stream(argv, env, dst):
            dst.write(b"1\t2\tmanual\n")

        with mock.patch.object(citation_dump, "stream_stdout", side_effect=fake_stream):
            citation_dump.write_copy_block({}, buffer, "cites", ["citing", "cited", "source"],
                                            CitationMode.FULL_SKELETON)
        self.assertNotIn("setval", buffer.getvalue().decode())


class DumpCitationTests(unittest.TestCase):
    COLUMNS = {
        "work": ["id", "key", "title", "abstract", "evidence"],
        "cites": ["citing", "cited", "source", "evidence"],
        "crawl_step": ["id", "crawl_id"],
        "public_policy": ["id", "mode", "note"],
        "schema_backfill": ["name", "applied_at"],
    }

    def _fake_stream(self, argv, env, dst):
        dst.write(b"-- DDL\n" if "pg_dump" in argv[0] else b"row\n")

    def test_none_mode_writes_nothing(self):
        buffer = io.BytesIO()
        with mock.patch.object(citation_dump, "stream_stdout") as stream_mock:
            citation_dump.dump_citation({}, buffer, CitationMode.NONE)
        stream_mock.assert_not_called()
        self.assertEqual(buffer.getvalue(), b"")

    def test_shipping_mode_writes_ddl_then_every_table(self):
        buffer = io.BytesIO()
        with mock.patch.object(citation_dump, "table_columns",
                                side_effect=lambda env, table: self.COLUMNS[table]), \
             mock.patch.object(citation_dump, "stream_stdout", side_effect=self._fake_stream):
            citation_dump.dump_citation({}, buffer, CitationMode.FULL_SKELETON)
        text = buffer.getvalue().decode()
        self.assertIn("-- DDL", text)
        self.assertLess(text.index("-- DDL"), text.index("COPY citation.work"))
        for table in citation_dump.CITATION_TABLES:
            self.assertIn(f"COPY citation.{table}", text)

    def test_ddl_excludes_age_owned_schemas(self):
        with mock.patch.object(citation_dump, "stream_stdout") as stream_mock:
            citation_dump.dump_ddl({}, io.BytesIO())
        (argv, _env, _dst), _kwargs = stream_mock.call_args
        self.assertIn("--schema=citation", argv)
        self.assertIn("--exclude-schema=citation_graph", argv)
        self.assertIn("--exclude-schema=ag_catalog", argv)


class ClassificationCoversTheCatalogTests(unittest.TestCase):
    """The map is complete against the DATABASE, not against itself.

    table_columns() reads pg_attribute, so the artifact's column list grows
    with the schema; the classification has to grow with it in the same
    commit or the build stops. Asserted as set equality both ways: a column
    added to the schema and not to the map is the leak this whole mechanism
    exists to prevent, and a column in the map the table no longer has is a
    classification of nothing, quietly rotting.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_every_dumped_column_is_classified_and_nothing_else_is(self):
        for table in citation_dump.CITATION_TABLES:
            self.assertEqual(
                set(citation_dump.table_columns(self.env, table)),
                set(citation_columns.CITATION_COLUMN_CLASS[table]),
                f"citation.{table}: каталог и классификация разошлись",
            )


class LegalCutSqlTests(unittest.TestCase):
    """The citation slice honours corpus.documents.public_distribution, and
    it does so the LEGAL_IS_DATA way: through the predicate legal_profile.py
    derives from the column, never through a list of ids or filenames here.
    """

    def test_work_select_drops_rows_naming_an_unshipped_document(self):
        sql = citation_dump.copy_select("work", ["id", "key", "document_id"],
                                        CitationMode.FULL_SKELETON)
        self.assertIn("public_distribution IN (", sql)
        self.assertIn("w.document_id IS NULL", sql)
        self.assertIn("corpus.documents d", sql)

    def test_metadata_only_documents_keep_their_work_row(self):
        # The predicate is SHIPPED (documents that have a row in the public
        # artifact at all), not FULL_CONTENT: a metadata-only document ships
        # its bibliography, so the work row naming it ships too.
        sql = citation_dump.copy_select("work", ["id"], CitationMode.FULL_SKELETON)
        self.assertIn("'metadata-only'", sql)

    def test_cites_select_requires_both_endpoints_to_ship(self):
        sql = citation_dump.copy_select("cites", ["citing", "cited", "source"],
                                        CitationMode.FULL_SKELETON)
        self.assertIn("JOIN citation.work wa ON wa.id = c.citing", sql)
        self.assertIn("JOIN citation.work wb ON wb.id = c.cited", sql)
        self.assertIn("wa.document_id IS NULL", sql)
        self.assertIn("wb.document_id IS NULL", sql)

    def test_crawl_step_select_carries_the_cut_sets_as_ctes(self):
        sql = citation_dump.copy_select("crawl_step", ["id"], CitationMode.FULL_SKELETON)
        self.assertTrue(sql.startswith("COPY (WITH cut_documents AS MATERIALIZED ("), sql[:70])
        self.assertIn("cut_keys AS MATERIALIZED (", sql)
        # The predicate is membership in them, once per statement.
        self.assertEqual(sql.count("FROM corpus.documents d"), 1)

    def test_only_crawl_step_gets_the_cut_set_prefix(self):
        # A real column of each table: an invented one is now a build error
        # (the classification covers exactly the catalog), which is a
        # different question from the one this test asks.
        for table, column in (("work", "id"), ("cites", "citing"),
                              ("public_policy", "id")):
            sql = citation_dump.copy_select(table, [column], CitationMode.FULL_SKELETON)
            self.assertNotIn("cut_documents", sql, table)

    def test_crawl_step_select_drops_rows_naming_a_cut_document_or_work(self):
        sql = citation_dump.copy_select("crawl_step", ["id", "frontier_key", "reason"],
                                        CitationMode.FULL_SKELETON)
        # frontier_key carries a document_id for seed/twin rows and a work
        # key for the rest, candidate_key the record decided about, node_key
        # the node it resolved to -- all three name-bearing columns are
        # checked against both vocabularies, in the cut CTEs' own alias.
        for column in ("frontier_key", "candidate_key", "node_key"):
            self.assertIn(f"j.{column}", sql)
        self.assertNotIn("strpos(", sql, "a name is matched as a name, not as a substring")
        self.assertNotIn("LIKE", sql, "'_' in a document id is a LIKE wildcard")

    def test_public_policy_is_not_cut(self):
        sql = citation_dump.copy_select("public_policy", ["id", "mode", "note"],
                                        CitationMode.FULL_SKELETON)
        self.assertNotIn("WHERE", sql)

    def test_no_document_id_or_filename_pattern_anywhere_in_the_module(self):
        """Mirror of test_legal_profile's own guard: the moment a document id
        or a filename shape appears here, the packager -- not the owner --
        has become the authority on a legal question.
        """
        source = citation_dump.__file__.replace(".pyc", ".py")
        text = open(source, encoding="utf-8").read()
        self.assertIsNone(re.search(r"\b(19|20)\d{2}_[a-z]", text),
                          "document id shaped literal in citation_dump.py")
        self.assertNotIn(".pdf", text)
        self.assertNotIn(".djvu", text)


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
