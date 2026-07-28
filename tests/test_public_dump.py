"""Unit tests for deploy/public_dump.py: no live Postgres, no pg_dump.

The dump is assembled from real subprocess output, so the pieces under test
here are the ones that decide WHAT gets asked of the server -- the column
lists (read from the catalog, never typed out) and the COPY selects that
apply the legal cut -- plus the assembled file's shape, with pg_dump/psql
replaced by stubs that write known text.
"""
from __future__ import annotations

import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import dump_scan
import public_dump
from legal_profile import Unclassified


class TableColumnsTests(unittest.TestCase):
    def _columns(self, stdout, **kwargs):
        with mock.patch.object(public_dump, "run_sql", return_value=mock.Mock(stdout=stdout)):
            return public_dump.table_columns({}, "documents", **kwargs)

    def test_reads_the_catalog_and_keeps_order(self):
        columns = self._columns("id\nfilename\nlegal_class\npublic_distribution\n")
        self.assertEqual(columns, ["id", "filename", "legal_class", "public_distribution"])

    def test_excludes_what_the_caller_names(self):
        columns = self._columns("id\ndocument_id\nbody\n", exclude=("id",))
        self.assertEqual(columns, ["document_id", "body"])

    def test_empty_result_is_an_error_not_an_empty_copy_block(self):
        with self.assertRaises(RuntimeError):
            self._columns("")

    def test_query_excludes_generated_columns(self):
        # tsv and source_path are GENERATED: including them would make the
        # dump either unrestorable or (worse) carry text that no longer
        # matches body.
        self.assertIn("a.attgenerated = ''", public_dump._COLUMNS_SQL)


class CopySelectTests(unittest.TestCase):
    def test_documents_blob_is_cut_by_the_legal_predicate(self):
        sql = public_dump._copy_select(
            "documents", ["id", "legal_class", "public_distribution", "source_blob"],
        )
        self.assertIn("CASE WHEN public_distribution IN ('full-text', 'internal') "
                      "THEN d.source_blob END AS source_blob", sql)
        # The classification columns themselves always ship: the public
        # artifact is the one package whose whole point is carrying them.
        self.assertIn("d.legal_class", sql)
        self.assertIn("d.public_distribution", sql)

    def test_pages_body_becomes_empty_for_metadata_only_documents(self):
        sql = public_dump._copy_select("pages", ["document_id", "page_number", "body", "embedding"])
        self.assertIn("CASE WHEN public_distribution IN ('full-text', 'internal') "
                      "THEN p.body ELSE '' END AS body", sql)
        # The embedding is never cut -- semantic search must still find the
        # document; only its text is withheld.
        self.assertIn("p.embedding", sql)
        self.assertIn("JOIN corpus.documents d ON d.id = p.document_id", sql)

    def test_an_excluded_document_is_filtered_out_of_both_tables(self):
        # Not blanked -- filtered: no documents row and no page row is
        # written for a document whose regime the owner has not established.
        shipped = "WHERE public_distribution IN ('full-text', 'metadata-only', 'internal')"
        documents = public_dump._copy_select("documents", ["id", "source_blob"])
        pages = public_dump._copy_select("pages", ["document_id", "page_number", "embedding"])
        self.assertIn(shipped, documents)
        self.assertIn(shipped, pages)

    def test_the_filter_is_the_shared_predicate_not_a_second_list(self):
        from legal_profile import SHIPPED_SQL
        self.assertIn(SHIPPED_SQL, public_dump._copy_select("documents", ["id"]))

    def test_pages_are_ordered_so_the_reassigned_sequence_is_deterministic(self):
        sql = public_dump._copy_select("pages", ["document_id", "page_number"])
        self.assertIn("ORDER BY p.document_id, p.page_number", sql)

    def test_embedding_model_is_copied_verbatim(self):
        sql = public_dump._copy_select("embedding_model", ["id", "model", "dims"])
        self.assertIn("m.model", sql)
        self.assertNotIn("CASE WHEN", sql)

    def test_unknown_table_raises_instead_of_guessing_an_alias(self):
        with self.assertRaises(KeyError):
            public_dump._copy_select("findings", ["id"])


class WriteCopyBlockTests(unittest.TestCase):
    def test_block_is_shaped_like_pg_dump(self):
        def fake_stream(argv, env, dst):
            dst.write(b"2009_isu34\t\\N\n")

        buffer = io.BytesIO()
        with mock.patch.object(public_dump, "stream_stdout", side_effect=fake_stream):
            public_dump.write_copy_block({}, buffer, "documents", ["id", "source_blob"])
        self.assertEqual(
            buffer.getvalue().decode(),
            "COPY corpus.documents (id, source_blob) FROM stdin;\n2009_isu34\t\\N\n\\.\n\n",
        )

    def test_the_server_side_copy_is_what_streams(self):
        with mock.patch.object(public_dump, "stream_stdout") as stream_mock:
            public_dump.write_copy_block({}, io.BytesIO(), "pages", ["document_id"])
        (argv, _env, _dst), _kwargs = stream_mock.call_args
        self.assertEqual(argv[0], "psql")
        self.assertIn("COPY (SELECT", argv[-1])
        self.assertIn("TO STDOUT", argv[-1])


class DumpPublicTests(unittest.TestCase):
    COLUMNS = {
        "documents": ["id", "source_blob"],
        "pages": ["document_id", "page_number", "body", "embedding"],
        "embedding_model": ["id", "model"],
    }

    def _run(self, tmp, stream_side_effect=None):
        gz_path = Path(tmp) / "01_dump.sql.gz"

        def fake_stream(argv, env, dst):
            if stream_side_effect is not None:
                stream_side_effect(argv, env, dst)
                return
            dst.write(b"-- DDL\n" if argv[0] == "pg_dump" else b"row\n")

        with mock.patch.object(public_dump, "require_classified") as classified_mock, \
             mock.patch.object(public_dump, "table_columns",
                                side_effect=lambda env, table, exclude=(): self.COLUMNS[table]), \
             mock.patch.object(public_dump, "stream_stdout", side_effect=fake_stream):
            public_dump.dump_public({}, gz_path)
        return gz_path, classified_mock

    def test_refuses_before_writing_when_a_document_is_unclassified(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified",
                                    side_effect=Unclassified("2026_x")), \
                 mock.patch.object(public_dump, "stream_stdout") as stream_mock:
                with self.assertRaises(Unclassified):
                    public_dump.dump_public({}, gz_path)
            self.assertFalse(gz_path.exists())
            stream_mock.assert_not_called()

    def test_writes_ddl_then_documents_then_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path, classified_mock = self._run(tmp)
            with gzip.open(gz_path, "rt") as f:
                text = f.read()
        classified_mock.assert_called_once()
        self.assertIn("PUBLIC profile", text)
        self.assertLess(text.index("-- DDL"), text.index("COPY corpus.documents"))
        # documents before pages: corpus.pages.document_id is a FK to it.
        self.assertLess(text.index("COPY corpus.documents"), text.index("COPY corpus.pages"))
        self.assertIn("COPY corpus.embedding_model", text)

    def test_measurements_is_never_dumped(self):
        # Asserted the way profile_checks.check_schemas asserts it against a
        # real artifact -- by the schemas the dump's own statements touch, not
        # by grepping for the word (the preamble mentions it on purpose).
        with tempfile.TemporaryDirectory() as tmp:
            gz_path, _ = self._run(tmp)
            self.assertEqual(dump_scan.schema_names(gz_path), {"corpus"})
        self.assertEqual(public_dump.PUBLIC_SCHEMAS, ("corpus",))

    def test_failed_child_removes_the_partial_file(self):
        def boom(argv, env, dst):
            raise public_dump.CommandFailed("pg_dump failed (exit 1): nope")

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(public_dump, "table_columns",
                                    side_effect=lambda env, table, exclude=(): self.COLUMNS[table]), \
                 mock.patch.object(public_dump, "stream_stdout", side_effect=boom):
                with self.assertRaises(RuntimeError) as ctx:
                    public_dump.dump_public({}, gz_path)
            self.assertIn("nope", str(ctx.exception))
            self.assertFalse(gz_path.exists())


if __name__ == "__main__":
    unittest.main()
