"""Unit tests for deploy/citation_dump.py: no live Postgres, no real
pg_dump/psql (stream_stdout/run_sql are stubbed). Mirrors test_public_dump.py's
style for the corpus-schema equivalent.
"""
from __future__ import annotations

import io
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_dump
from manifest_contract import CitationMode


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
        self.assertIn("SELECT id,", sql)

    def test_topology_only_blanks_cites_evidence(self):
        sql = citation_dump.copy_select("cites", ["citing", "cited", "evidence"], CitationMode.TOPOLOGY_ONLY)
        self.assertIn("NULL::jsonb AS evidence", sql)
        self.assertIn("citing", sql)
        self.assertIn("cited", sql)

    def test_crawl_step_and_public_policy_are_never_blanked_under_any_mode(self):
        for mode in CitationMode.ALL:
            sql = citation_dump.copy_select("crawl_step", ["id", "reason"], mode)
            self.assertNotIn("NULL::", sql)
            sql = citation_dump.copy_select("public_policy", ["id", "mode", "note"], mode)
            self.assertNotIn("NULL::", sql)

    def test_order_by_matches_each_table_shape(self):
        self.assertIn("ORDER BY id", citation_dump.copy_select("work", ["id"], CitationMode.FULL_SKELETON))
        self.assertIn(
            "ORDER BY citing, cited, source",
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


if __name__ == "__main__":
    unittest.main()
