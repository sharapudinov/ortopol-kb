"""Unit tests for deploy/citation_dump.py: no live Postgres, no real
pg_dump/psql (stream_stdout/run_sql are stubbed). Mirrors test_public_dump.py's
style for the corpus-schema equivalent.

The half that asks the live catalog what the schema holds -- and therefore
which tables ship, which columns carry a sequence and what order the
foreign keys require -- is test_citation_dump_live.py.
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
import schema_catalog
import citation_profile
from legal_profile import SHIPPED_SQL
from manifest_contract import CitationMode, Profile, schemas_for
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"

# What the schema holds today, in the order its foreign keys require --
# asserted against the live catalog in CatalogDerivedShapeTests, and used
# here as the stub for citation_tables().
DUMPED_TABLES = ("work", "cites", "crawl_step", "public_policy", "schema_backfill")


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


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

    def test_a_mode_nobody_declared_full_content_still_blanks_content(self):
        """The polarity is the classification's, not topology-only's.

        A mode added to the vocabulary and forgotten here must strip rather
        than ship: CitationMode.FULL_CONTENT names the modes whose rows
        carry abstract/evidence, and everything outside it -- including a
        value this build has never heard of -- is stripped, the same
        refusal-by-default citation_columns.blanked_cast() applies per
        column.
        """
        modes = [m for m in CitationMode.ALL if m not in CitationMode.FULL_CONTENT]
        modes.append("skeleton-with-notes")
        for mode in modes:
            for table, columns in citation_columns.CITATION_COLUMN_CLASS.items():
                sql = citation_dump.copy_select(table, list(columns), mode)
                alias = citation_dump.TABLE_ALIASES[table]
                for column, kind in columns.items():
                    if kind == citation_columns.CONTENT:
                        self.assertIn(f"AS {column}", sql, f"{mode}: {table}.{column}")
                        self.assertNotIn(f"{alias}.{column},", sql, f"{mode}: {table}.{column}")

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
    """The statement itself is schema_catalog's, written for both dumps
    (test_public_dump.SerialColumnsAreRepositionedTests holds the two to
    one emission); this asserts the citation half's own arguments.
    """

    def test_a_serial_column_gets_a_setval_statement(self):
        sql = schema_catalog.setval_sql(citation_dump.SCHEMA, "work", "id").decode()
        self.assertIn("pg_get_serial_sequence('citation.work', 'id')", sql)
        self.assertIn("coalesce((SELECT max(id) FROM citation.work), 1)", sql)


class WriteCopyBlockTests(unittest.TestCase):
    def test_serial_table_writes_setval_after_the_terminator(self):
        buffer = io.BytesIO()

        def fake_stream(argv, env, dst):
            dst.write(b"1\tk1\n")

        with mock.patch.object(citation_dump, "stream_stdout", side_effect=fake_stream):
            citation_dump.write_copy_block({}, buffer, citation_dump.CopyBlock(
                "work", ["id", "key"], ("id",),
                citation_dump.copy_select("work", ["id", "key"],
                                          CitationMode.FULL_SKELETON)))
        text = buffer.getvalue().decode()
        self.assertIn("COPY citation.work (id, key) FROM stdin;\n1\tk1\n\\.\n", text)
        self.assertIn("setval(pg_get_serial_sequence", text)

    def test_non_serial_table_writes_no_setval(self):
        buffer = io.BytesIO()

        def fake_stream(argv, env, dst):
            dst.write(b"1\t2\tmanual\n")

        with mock.patch.object(citation_dump, "stream_stdout", side_effect=fake_stream):
            columns = ["citing", "cited", "source"]
            citation_dump.write_copy_block({}, buffer, citation_dump.CopyBlock(
                "cites", columns, (),
                citation_dump.copy_select("cites", columns, CitationMode.FULL_SKELETON)))
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

    def _plan(self, mode):
        """plan_citation() over DUMPED_TABLES, with the catalog mocked."""
        with mock.patch.object(citation_dump, "citation_tables",
                               return_value=list(DUMPED_TABLES)), \
             mock.patch.object(citation_dump, "schema_columns",
                               return_value=dict(self.COLUMNS)), \
             mock.patch.object(citation_dump, "schema_serial_columns",
                               return_value={"work": ["id"]}):
            return citation_dump.plan_citation({}, mode)

    def test_none_mode_writes_nothing(self):
        buffer = io.BytesIO()
        with mock.patch.object(citation_dump, "stream_stdout") as stream_mock:
            citation_dump.dump_citation({}, buffer,
                                        citation_dump.plan_citation({}, CitationMode.NONE))
        stream_mock.assert_not_called()
        self.assertEqual(buffer.getvalue(), b"")

    def test_a_non_shipping_mode_asks_the_catalog_nothing_at_all(self):
        """The plan for a schema that does not travel costs no round trip:
        the refusal is the whole answer, and a catalog read would be a
        question about a schema this build has no business describing.
        """
        with mock.patch.object(citation_dump, "schema_columns") as columns_mock, \
             mock.patch.object(citation_dump, "citation_tables") as tables_mock:
            plan = citation_dump.plan_citation({}, CitationMode.NONE)
        self.assertFalse(plan.ships)
        self.assertEqual(plan.blocks, ())
        columns_mock.assert_not_called()
        tables_mock.assert_not_called()

    def test_the_plan_resolves_every_statement_before_a_byte_is_written(self):
        """What public_dump.py relies on: an unclassified table or column
        raises from plan_citation(), i.e. before the gzip file exists, not
        from inside the loop that writes into it.
        """
        plan = self._plan(CitationMode.FULL_SKELETON)
        self.assertTrue(plan.ships)
        self.assertEqual([block.table for block in plan.blocks], list(DUMPED_TABLES))
        for block in plan.blocks:
            with self.subTest(table=block.table):
                self.assertIn(f"FROM citation.{block.table}", block.statement)
        self.assertEqual(plan.blocks[0].serials, ("id",))

    def test_the_dump_and_the_manifest_agree_on_every_mode(self):
        """One predicate on both sides of "does this schema travel".

        The dump used to refuse by denylist (`== NONE`) while the manifest
        declared by allowlist (`in SHIPPED`): a mode added to the inherited
        vocabulary and not to SHIPPED would have been written into the
        artifact and left out of manifest.json. Every declared mode is
        walked here, so the two spellings cannot drift apart silently.
        """
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode):
                declared = "citation" in schemas_for(Profile.PUBLIC, mode)
                buffer = io.BytesIO()
                with mock.patch.object(citation_dump, "citation_tables",
                                        return_value=list(DUMPED_TABLES)), \
                     mock.patch.object(citation_dump, "schema_columns",
                                        return_value=dict(self.COLUMNS)), \
                     mock.patch.object(citation_dump, "schema_serial_columns",
                                        return_value={}), \
                     mock.patch.object(citation_dump, "stream_stdout",
                                        side_effect=self._fake_stream):
                    citation_dump.dump_citation({}, buffer,
                                                citation_dump.plan_citation({}, mode))
                self.assertEqual(bool(buffer.getvalue()), declared,
                                 f"режим {mode!r}: дамп и манифест разошлись")

    def test_a_mode_outside_the_shipping_list_writes_nothing(self):
        """The control for the walk above: an unheard-of mode is not NONE,
        and must still ship no byte.
        """
        buffer = io.BytesIO()
        with mock.patch.object(citation_dump, "stream_stdout") as stream_mock:
            citation_dump.dump_citation({}, buffer,
                                        citation_dump.plan_citation({}, "graph-only"))
        stream_mock.assert_not_called()
        self.assertEqual(buffer.getvalue(), b"")

    def _dump(self, buffer):
        """dump_citation() over DUMPED_TABLES; returns the two catalog mocks."""
        with mock.patch.object(citation_dump, "citation_tables",
                                return_value=list(DUMPED_TABLES)), \
             mock.patch.object(citation_dump, "schema_columns",
                                return_value=dict(self.COLUMNS)) as columns_mock, \
             mock.patch.object(citation_dump, "schema_serial_columns",
                                return_value={"work": ["id"]}) as serials_mock, \
             mock.patch.object(citation_dump, "stream_stdout", side_effect=self._fake_stream):
            citation_dump.dump_citation(
                {}, buffer, citation_dump.plan_citation({}, CitationMode.FULL_SKELETON))
        return columns_mock, serials_mock

    def test_shipping_mode_writes_ddl_then_every_table(self):
        buffer = io.BytesIO()
        self._dump(buffer)
        text = buffer.getvalue().decode()
        self.assertIn("-- DDL", text)
        self.assertLess(text.index("-- DDL"), text.index("COPY citation.work"))
        for table in DUMPED_TABLES:
            self.assertIn(f"COPY citation.{table}", text)

    def test_the_catalog_is_asked_twice_for_the_schema_not_twice_per_table(self):
        """One psql process per COPY block is the work; one more per table
        per catalog question was the loop pg_attribute answers in one read.
        """
        columns_mock, serials_mock = self._dump(io.BytesIO())
        columns_mock.assert_called_once()
        serials_mock.assert_called_once()

    def test_only_the_tables_with_a_sequence_get_a_setval(self):
        buffer = io.BytesIO()
        self._dump(buffer)
        text = buffer.getvalue().decode()
        self.assertIn("pg_get_serial_sequence('citation.work', 'id')", text)
        self.assertNotIn("pg_get_serial_sequence('citation.cites'", text)

    def test_ddl_excludes_age_owned_schemas(self):
        with mock.patch.object(citation_dump, "stream_stdout") as stream_mock:
            citation_dump.dump_ddl({}, io.BytesIO())
        (argv, _env, _dst), _kwargs = stream_mock.call_args
        self.assertIn("--schema=citation", argv)
        self.assertIn("--exclude-schema=citation_graph", argv)
        self.assertIn("--exclude-schema=ag_catalog", argv)


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


if __name__ == "__main__":
    unittest.main()
