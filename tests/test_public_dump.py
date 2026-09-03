"""Unit tests for deploy/public_dump.py's corpus half: no live Postgres,
no pg_dump.

The dump is assembled from real subprocess output, so the pieces under test
here are the ones that decide WHAT gets asked of the server -- the column
lists (read from the catalog, never typed out) and the COPY selects that
apply the legal cut -- plus the assembled file's shape, with pg_dump/psql
replaced by stubs that write known text.

The handshake with the citation half (whether dump_citation() is called at
all, and what its slice of the file then looks like) is
test_public_dump_citation.py's -- the split is kb/CLAUDE.md FILE_SIZE, along
the seam the module itself keeps between the schema it cuts per document
and the schema it cuts per policy.
"""
from __future__ import annotations

import ast
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401
from _dump_fixtures import CORPUS_COLUMNS, CORPUS_SERIALS

import dump_scan
import corpus_columns
import corpus_cut
import public_dump
import schema_catalog
from legal_profile import Unclassified
from manifest_contract import CitationMode, Profile, base_schemas_for, schemas_for



class CorpusTablesComeFromTheCatalogTests(unittest.TestCase):
    """The corpus half of the dump asks the same engine the citation half
    does (schema_catalog.py). Its table list used to be a hand-typed map
    and a hand-ordered sequence of calls: pg_schema.sql declares exactly
    the three below, so they agreed by attention alone, and a fourth table
    would have shipped its DDL with no COPY block -- profile_checks
    compares schemas, not tables, so nothing downstream contradicts one.
    """

    def test_the_list_and_the_order_are_the_catalogs(self):
        with mock.patch.object(corpus_cut.schema_catalog, "present_tables",
                                return_value=["documents", "pages", "embedding_model"]), \
             mock.patch.object(corpus_cut.schema_catalog, "foreign_key_edges",
                                return_value=[("pages", "documents")]):
            self.assertEqual(corpus_cut.corpus_tables({}),
                             ["documents", "pages", "embedding_model"])

    def test_a_child_written_before_its_parent_is_reordered(self):
        with mock.patch.object(corpus_cut.schema_catalog, "present_tables",
                                return_value=["pages", "documents", "embedding_model"]), \
             mock.patch.object(corpus_cut.schema_catalog, "foreign_key_edges",
                                return_value=[("pages", "documents")]):
            self.assertEqual(corpus_cut.corpus_tables({}),
                             ["documents", "pages", "embedding_model"])

    def test_a_table_nobody_classified_stops_the_build(self):
        with mock.patch.object(corpus_cut.schema_catalog, "present_tables",
                                return_value=["documents", "pages", "embedding_model",
                                              "annotations"]), \
             mock.patch.object(corpus_cut.schema_catalog, "foreign_key_edges",
                                return_value=[]):
            with self.assertRaises(corpus_cut.schema_catalog.TableUnclassified) as caught:
                corpus_cut.corpus_tables({})
        self.assertIn("corpus.annotations", str(caught.exception))

    def test_a_table_with_an_alias_but_no_row_source_is_unclassified(self):
        """Half a classification is none: an alias says how to spell the
        table's columns, not which of its rows may leave. The build used to
        hold the list to TABLE_ALIASES alone and hand anything else an
        unfiltered `FROM corpus.<table> ORDER BY id`.
        """
        aliased = dict(corpus_cut.TABLE_ALIASES, annotations="a")
        classified = set(aliased) & set(corpus_cut._SOURCE)
        with mock.patch.object(corpus_cut, "CLASSIFIED", classified), \
             mock.patch.object(corpus_cut.schema_catalog, "present_tables",
                                return_value=["documents", "pages", "embedding_model",
                                              "annotations"]), \
             mock.patch.object(corpus_cut.schema_catalog, "foreign_key_edges",
                                return_value=[]):
            with self.assertRaises(corpus_cut.schema_catalog.TableUnclassified) as caught:
                corpus_cut.corpus_tables({})
        self.assertIn("corpus.annotations", str(caught.exception))
        self.assertIn("_SOURCE", str(caught.exception))

    def test_both_maps_answer_for_the_same_tables(self):
        self.assertEqual(set(corpus_cut.TABLE_ALIASES), set(corpus_cut._SOURCE))
        self.assertEqual(corpus_cut.CLASSIFIED, set(corpus_cut._SOURCE))

    def test_the_dump_writes_no_byte_when_a_table_is_unclassified(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(corpus_cut.schema_catalog, "present_tables",
                                    return_value=["documents", "pages",
                                                  "embedding_model", "annotations"]), \
                 mock.patch.object(corpus_cut.schema_catalog, "foreign_key_edges",
                                    return_value=[]), \
                 mock.patch.object(public_dump, "stream_stdout") as stream_mock:
                with self.assertRaises(corpus_cut.schema_catalog.TableUnclassified):
                    public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
            self.assertFalse(gz_path.exists())
            stream_mock.assert_not_called()

    def test_the_column_read_excludes_generated_columns(self):
        # tsv and source_path are GENERATED: including them would make the
        # dump either unrestorable or (worse) carry text that no longer
        # matches body.
        self.assertIn("a.attgenerated = ''", public_dump.schema_catalog._COLUMNS_SQL)

    def test_the_module_no_longer_reads_the_catalog_itself(self):
        """One implementation of "what does this schema hold", asked with a
        schema argument -- not one per schema.
        """
        source = Path(public_dump.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pg_attribute", source)
        self.assertNotIn("pg_namespace", source)


class BaseSchemasAreAskedForByNameTests(unittest.TestCase):
    """_dump_ddl() must ask pg_dump for everything EXCEPT citation, whose
    DDL citation_dump.dump_ddl() writes into the same file afterwards.

    That is a mechanical fact about who writes what, and it used to be
    spelled as a policy value -- schemas_for(PUBLIC, <a mode that ships
    nothing>) -- so nothing on the contract's side recorded the dependency.
    Both statements below would have been emitted for schema citation had
    the mode ever started including it, and a dump with two CREATE SCHEMA
    citation statements aborts the recipient's restore.
    """

    def test_the_public_dump_asks_pg_dump_for_the_base_schemas(self):
        self.assertEqual(public_dump.PUBLIC_SCHEMAS, base_schemas_for(Profile.PUBLIC))
        self.assertEqual(public_dump.PUBLIC_SCHEMAS, ("corpus",))

    def test_the_whole_list_is_the_base_list_plus_citation_when_shipped(self):
        """The one relationship between the two accessors, pinned where it
        can be read: everything else in either list is the same fact.
        """
        for profile in Profile.ALL:
            for mode in CitationMode.ALL:
                with self.subTest(profile=profile, mode=mode):
                    extra = ["citation"] if mode in CitationMode.SHIPPED else []
                    self.assertEqual(schemas_for(profile, mode),
                                     list(base_schemas_for(profile)) + extra)

    def test_an_unknown_profile_is_refused_rather_than_defaulted(self):
        with self.assertRaises(ValueError) as caught:
            base_schemas_for("draft")
        self.assertIn("draft", str(caught.exception))

    def test_no_mode_reaches_the_ddl_schema_list_at_all(self):
        """A mode named here would be a second place deciding what the
        citation half of the file contains: this module decides WHETHER to
        call citation_dump, and the mode it passes on is the one
        build_package resolved.
        """
        source = Path(public_dump.__file__).read_text(encoding="utf-8")
        called = {node.func.id for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("base_schemas_for", called)
        self.assertNotIn("schemas_for", called)


class CopySelectTests(unittest.TestCase):
    def test_documents_blob_is_cut_by_the_legal_predicate(self):
        sql = corpus_cut.copy_select(
            "documents", ["id", "legal_class", "public_distribution", "source_blob"],
        )
        # ELSE NULL::bytea, not a bare END: what stands in for a withheld
        # value is declared per column (corpus_columns.CONTENT_WITHHELD) and
        # typed, so the COPY column list never leaves the type to be guessed.
        self.assertIn("CASE WHEN public_distribution IN ('full-text', 'internal') "
                      "THEN d.source_blob ELSE NULL::bytea END AS source_blob", sql)
        # The classification columns themselves always ship: the public
        # artifact is the one package whose whole point is carrying them.
        self.assertIn("d.legal_class", sql)
        self.assertIn("d.public_distribution", sql)

    def test_pages_body_becomes_empty_for_metadata_only_documents(self):
        sql = corpus_cut.copy_select("pages", ["document_id", "page_number", "body", "embedding"])
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
        documents = corpus_cut.copy_select("documents", ["id", "source_blob"])
        pages = corpus_cut.copy_select("pages", ["document_id", "page_number", "embedding"])
        self.assertIn(shipped, documents)
        self.assertIn(shipped, pages)

    def test_the_filter_is_the_shared_predicate_not_a_second_list(self):
        from legal_profile import SHIPPED_SQL
        self.assertIn(SHIPPED_SQL, corpus_cut.copy_select("documents", ["id"]))

    def test_pages_are_ordered_so_the_reassigned_sequence_is_deterministic(self):
        sql = corpus_cut.copy_select("pages", ["document_id", "page_number"])
        self.assertIn("ORDER BY p.document_id, p.page_number", sql)

    def test_embedding_model_is_copied_verbatim(self):
        sql = corpus_cut.copy_select("embedding_model", ["id", "model", "dims"])
        self.assertIn("m.model", sql)
        self.assertNotIn("CASE WHEN", sql)

    def test_unknown_table_raises_instead_of_guessing_an_alias(self):
        with self.assertRaises(KeyError):
            corpus_cut.copy_select("findings", ["id"])

    def test_a_table_with_an_alias_and_no_source_still_has_no_select(self):
        """The projection can be spelled for it, the rows cannot: an alias
        alone must not produce a statement.

        Which of the three maps refuses first is not the point -- with an
        alias in place it is the column classification (corpus_columns.py),
        which no more knows this table than _SOURCE does. The point is that
        no statement comes out.
        """
        with mock.patch.dict(corpus_cut.TABLE_ALIASES, {"findings": "f"}):
            with self.assertRaises((KeyError, corpus_columns.ColumnUnclassified)):
                corpus_cut.copy_select("findings", ["id"])

    def test_a_column_nobody_classified_stops_the_build(self):
        """The fall-through this map replaced: a column added to
        corpus.documents and named in no map used to SHIP, whatever it
        carried -- the catalog reads the column list, so it reached the
        projection on its own.
        """
        with self.assertRaises(corpus_columns.ColumnUnclassified) as raised:
            corpus_cut.copy_select("documents", ["id", "brand_new"])
        self.assertIn("corpus.documents.brand_new", str(raised.exception))

    def test_every_classified_column_of_every_table_can_be_projected(self):
        """The complement: nothing in the map is unprojectable, and every
        content column has its replacement declared (a content column
        without one raises just as loudly as an unclassified one).
        """
        for table, columns in corpus_columns.CORPUS_COLUMN_CLASS.items():
            sql = corpus_cut.copy_select(table, list(columns))
            for column, kind in columns.items():
                with self.subTest(table=table, column=column):
                    if kind == corpus_columns.CONTENT:
                        self.assertIn(f"END AS {column}", sql)
                    else:
                        self.assertIn(f"{corpus_cut.TABLE_ALIASES[table]}.{column}", sql)


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


class SerialColumnsAreRepositionedTests(unittest.TestCase):
    """The safeguard schema_catalog was introduced for, on the schema its
    own docstring calls the more sensitive one.

    corpus.pages.id is the only BIGSERIAL the corpus has, and the dump was
    correct only because PAGES_EXCLUDED happens to drop it from the COPY --
    hand-kept knowledge beside a classification guard that never looks at
    it. A corpus table added with a serial id has to appear in TABLE_ALIASES
    and _SOURCE to build, and neither of them says anything about sequences.
    """

    def _block(self, table, columns, serials):
        buffer = io.BytesIO()
        with mock.patch.object(public_dump, "stream_stdout",
                                side_effect=lambda argv, env, dst: dst.write(b"row\n")):
            public_dump.write_copy_block({}, buffer, table, columns, serials=serials)
        return buffer.getvalue().decode()

    def test_a_serial_column_is_reset_after_the_copy_terminator(self):
        text = self._block("pages", ["document_id", "page_number"], ["id"])
        self.assertLess(text.index("\\.\n"), text.index("setval"))
        self.assertIn("pg_get_serial_sequence('corpus.pages', 'id')", text)

    def test_a_table_with_no_sequence_gets_no_statement(self):
        self.assertNotIn("setval", self._block("documents", ["id"], ()))

    def test_both_dumps_write_the_same_statement(self):
        """One emission, parameterised by schema: the corpus half used to
        have none at all while the citation half had its own copy.
        """
        self.assertEqual(
            schema_catalog.setval_sql("corpus", "pages", "id"),
            schema_catalog.setval_sql("citation", "work", "id")
            .replace(b"citation.work", b"corpus.pages"))

    def test_an_empty_table_does_not_burn_its_first_id(self):
        sql = schema_catalog.setval_sql("corpus", "pages", "id").decode()
        self.assertIn("IS NOT NULL", sql)
        self.assertIn("coalesce", sql)

    def test_the_table_is_scanned_once_per_sequence_not_twice(self):
        """The value and the is_called flag are two answers to ONE
        max(): spelled as two scalar subqueries, Postgres evaluates both,
        so every sequence-owning column cost the recipient's restore a
        second aggregate over the table that had just been copied in.
        """
        sql = schema_catalog.setval_sql("corpus", "pages", "id").decode()
        self.assertEqual(sql.count("max(id)"), 1, sql)
        self.assertEqual(sql.count("FROM corpus.pages"), 1, sql)

    def test_the_statement_stays_on_one_line(self):
        """dump_scan.sequence_resets() reads the shipped bytes line by
        line, so a statement broken across lines is a setval nothing can
        see -- and its absence is a restore that succeeds and collides on
        the recipient's first INSERT.
        """
        sql = schema_catalog.setval_sql("corpus", "pages", "id").decode()
        self.assertEqual(sql.count("\n"), 1)
        self.assertTrue(sql.endswith(";\n"), sql)

    def test_the_columns_to_reset_are_the_catalogs_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(public_dump, "corpus_tables",
                                    return_value=list(DumpPublicTests.COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                    return_value=dict(DumpPublicTests.COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS) as serials_mock, \
                 mock.patch.object(public_dump, "stream_stdout",
                                    side_effect=lambda argv, env, dst: dst.write(b"row\n")):
                public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
            serials_mock.assert_called_once_with({}, "corpus")
            self.assertEqual(dump_scan.sequence_resets(gz_path), {"corpus.pages.id"})

    def test_a_new_serial_column_is_reset_without_being_named_anywhere(self):
        """The point of asking the catalog: nothing below mentions the
        table, and its sequence still travels correctly.
        """
        columns = dict(DumpPublicTests.COLUMNS, pages=["document_id", "page_number"])
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(public_dump, "corpus_tables", return_value=list(columns)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                    return_value=columns), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value={"documents": ["id"], "pages": ["id"]}), \
                 mock.patch.object(public_dump, "stream_stdout",
                                    side_effect=lambda argv, env, dst: dst.write(b"row\n")):
                public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
            self.assertEqual(dump_scan.sequence_resets(gz_path),
                             {"corpus.documents.id", "corpus.pages.id"})


class DumpPublicTests(unittest.TestCase):
    COLUMNS = CORPUS_COLUMNS

    def _run(self, tmp, stream_side_effect=None):
        gz_path = Path(tmp) / "01_dump.sql.gz"

        def fake_stream(argv, env, dst):
            if stream_side_effect is not None:
                stream_side_effect(argv, env, dst)
                return
            dst.write(b"-- DDL\n" if argv[0] == "pg_dump" else b"row\n")

        with mock.patch.object(public_dump, "require_classified") as classified_mock, \
             mock.patch.object(public_dump, "corpus_tables",
                                return_value=list(DumpPublicTests.COLUMNS)), \
             mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                return_value=dict(DumpPublicTests.COLUMNS)), \
             mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                return_value=CORPUS_SERIALS), \
             mock.patch.object(public_dump, "stream_stdout", side_effect=fake_stream):
            public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
        return gz_path, classified_mock

    def test_refuses_before_writing_when_a_document_is_unclassified(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified",
                                    side_effect=Unclassified("2026_x")), \
                 mock.patch.object(public_dump, "stream_stdout") as stream_mock:
                with self.assertRaises(Unclassified):
                    public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
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
                 mock.patch.object(public_dump, "corpus_tables",
                                    return_value=list(DumpPublicTests.COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                    return_value=dict(DumpPublicTests.COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(public_dump, "stream_stdout", side_effect=boom):
                with self.assertRaises(RuntimeError) as ctx:
                    public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
            self.assertIn("nope", str(ctx.exception))
            self.assertFalse(gz_path.exists())



if __name__ == "__main__":
    unittest.main()
