"""Unit tests for deploy/corpus_cut.py: WHICH rows of schema corpus travel,
how each column is projected, and the plan that answers both before any
file is opened.

Split out of test_public_dump.py (kb/CLAUDE.md FILE_SIZE) along the seam
the modules keep: that module assembles the file, this one answers what
belongs in it. No live Postgres -- the catalog reads are stubbed, and what
is asserted is the SQL built from their answers and the refusal built from
their gaps.
"""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401
from _dump_fixtures import CORPUS_COLUMNS, CORPUS_SERIALS

import copy_writer
import corpus_columns
import corpus_cut
import public_dump
from manifest_contract import CitationMode


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

    def test_an_unclassified_column_leaves_no_file_either(self):
        """The refusal the corpus half acquired most recently, and the one
        the plan exists for. ColumnUnclassified is not a CommandFailed, so
        raised from inside the open gzip it goes past the handler that
        unlinks -- and this schema is written FIRST, so what stays on disk
        is the preamble, the whole DDL and however many blocks came before
        the offending one, under a docstring promising nothing was written.
        """
        columns = dict(CORPUS_COLUMNS)
        columns["documents"] = [*columns["documents"], "annotation"]
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(corpus_cut, "corpus_tables",
                                    return_value=list(columns)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                    return_value=columns), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(public_dump, "stream_stdout") as stream_mock:
                with self.assertRaises(corpus_columns.ColumnUnclassified) as caught:
                    public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
            self.assertIn("corpus.documents.annotation", str(caught.exception))
            self.assertFalse(gz_path.exists())
            stream_mock.assert_not_called()

    def test_every_corpus_block_is_resolved_before_the_file_is_opened(self):
        """The structural half of the promise: gzip.open is reached only
        once every statement exists. Asserted as an ORDER, because the
        refusals above are only the ones known today -- a question moved
        back inside the file context passes both of them and reinstates the
        truncated dump for the next one.
        """
        seen = []
        real_open, real_select = gzip.open, corpus_cut.copy_select

        def watched_open(*args, **kwargs):
            seen.append("gzip.open")
            return real_open(*args, **kwargs)

        def watched_select(table, columns):
            seen.append(f"select:{table}")
            return real_select(table, columns)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(corpus_cut, "corpus_tables",
                                    return_value=list(CORPUS_COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_columns",
                                    return_value=dict(CORPUS_COLUMNS)), \
                 mock.patch.object(corpus_cut.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(corpus_cut, "copy_select", side_effect=watched_select), \
                 mock.patch.object(gzip, "open", side_effect=watched_open), \
                 mock.patch.object(public_dump, "stream_stdout",
                                    side_effect=lambda argv, env, dst: dst.write(b"-- DDL\n")), \
                 mock.patch.object(copy_writer, "stream_stdout",
                                    side_effect=lambda argv, env, dst: dst.write(b"row\n")):
                public_dump.dump_public({}, gz_path, citation_mode=CitationMode.NONE)
        self.assertEqual(seen[-1], "gzip.open", seen)
        self.assertEqual(len(seen), len(CORPUS_COLUMNS) + 1, seen)

    def test_the_column_read_excludes_generated_columns(self):
        # tsv and source_path are GENERATED: including them would make the
        # dump either unrestorable or (worse) carry text that no longer
        # matches body.
        self.assertIn("a.attgenerated = ''", corpus_cut.schema_catalog._COLUMNS_SQL)

    def test_the_module_no_longer_reads_the_catalog_itself(self):
        """One implementation of "what does this schema hold", asked with a
        schema argument -- not one per schema.
        """
        source = Path(public_dump.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pg_attribute", source)
        self.assertNotIn("pg_namespace", source)


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


if __name__ == "__main__":
    unittest.main()
