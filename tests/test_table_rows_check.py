"""deploy/table_rows_check.py: every table a manifest block declares is in
the dump with exactly that many rows, and nothing of that schema is in the
dump the manifest does not name.

Asked of BOTH classified schemas from one engine, which is what this module
is written against: the corpus cases and the citation cases go through the
same function with the schema as an argument, so a polarity that decayed in
one copy would have to decay in both at once.

Same discipline as its neighbours: a real dump_scan.scan() pass over
hand-built COPY blocks (_artifact_fixtures.citation_scan), so a check is
held to actual bytes rather than to a mocked scan.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import column_class_checks
import table_rows_check
from _artifact_fixtures import (
    citation_copy_block as _copy_block,
    citation_scan as _scan,
)
from manifest_keys import Key
from manifest_contract import CitationMode, Profile

CITATION = "citation"
CORPUS = "corpus"


class CheckEveryDeclaredTableShippedTests(unittest.TestCase):
    """The manifest names every citation table the dump carries, and the
    dump carries every table the manifest names -- with the same row count.

    Without this, the recipient learned nothing about crawl_step,
    public_policy or schema_backfill: the checks that read the journal find
    no rows, report a green nought and certify a package that never shipped
    the table their whole subject is.
    """

    JOURNAL = ["id", "frontier_key"]

    def _scans(self, tables: dict[str, list[list[str]]]) -> dict:
        dump = "CREATE TABLE corpus.documents (id text);\n"
        for table, rows in tables.items():
            columns = ["id", "key"] if table in ("work",) else self.JOURNAL
            dump += _copy_block(f"citation.{table}", columns, rows)
        scans, _facts = _scan(dump)
        return scans

    def _manifest(self, declared) -> dict:
        return {Key.PROFILE: Profile.PUBLIC,
                Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY,
                               Key.TABLE_ROWS: declared}}

    def test_every_declared_table_present_with_its_count_passes(self):
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"], ["2", "k1"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 2}), scans, CITATION)
        self.assertTrue(ok, detail)

    def test_a_declared_table_the_dump_never_carried_is_the_whole_point(self):
        """The journal is declared and absent -- which is exactly the
        package on which check_journal_names_nothing_cut() reported zero
        names and passed.
        """
        scans = self._scans({"work": [["1", "k1"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 604}), scans, CITATION)
        self.assertFalse(ok)
        self.assertIn("citation.crawl_step", detail)
        self.assertIn("604", detail)

    def test_a_row_count_that_does_not_match_the_declaration_fails(self):
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 2}), scans, CITATION)
        self.assertFalse(ok)
        self.assertIn("1 строк против 2", detail)

    def test_a_table_in_the_dump_that_the_manifest_does_not_name_fails(self):
        """The other direction: a table shipped without being described is
        a slice of the schema nothing on this side can hold to anything.
        """
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"work": 1}), scans, CITATION)
        self.assertFalse(ok)
        self.assertIn("citation.crawl_step", detail)

    def test_a_shipping_mode_declaring_no_table_at_all_is_refused(self):
        for declared in (None, {}, "нет"):
            with self.subTest(declared=declared):
                ok, detail = table_rows_check.check_every_declared_table_shipped(
                    self._manifest(declared), self._scans({"work": [["1", "k1"]]}), CITATION)
                self.assertFalse(ok, detail)
                self.assertIn(Key.TABLE_ROWS, detail)

    def test_a_mode_that_ships_nothing_declares_nothing_and_carries_nothing(self):
        manifest = {Key.PROFILE: Profile.PUBLIC,
                    Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE, Key.TABLE_ROWS: {}}}
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            manifest, self._scans({}), CITATION)
        self.assertTrue(ok, detail)

    def test_a_mode_that_ships_nothing_may_not_declare_a_table_either(self):
        manifest = {Key.PROFILE: Profile.PUBLIC,
                    Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                                   Key.TABLE_ROWS: {"work": 1}}}
        ok, _detail = table_rows_check.check_every_declared_table_shipped(
            manifest, self._scans({}), CITATION)
        self.assertFalse(ok)

    def test_a_mode_outside_the_vocabulary_cannot_excuse_the_check(self):
        """Whether the artifact carries the schema is the profile+mode rule,
        not the manifest's own list: a mode nobody has heard of leaves the
        question unanswerable, which is a red row, not an exemption.
        """
        manifest = {Key.PROFILE: Profile.PUBLIC,
                    Key.CITATION: {Key.CITATION_MODE: "half-skeleton", Key.TABLE_ROWS: {}}}
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            manifest, self._scans({}), CITATION)
        self.assertFalse(ok)
        self.assertIn("half-skeleton", detail)


class CorpusHalfTests(unittest.TestCase):
    """The same question of schema corpus, which every profile ships.

    documents_count and pages_count describe two tables; every other corpus
    table was described by nothing at all, so a dump that dropped one
    satisfied every check about it -- the polarity the citation half already
    had, one schema over.
    """

    def _scans(self, tables: dict[str, list[list[str]]]) -> dict:
        dump = ""
        for table, rows in tables.items():
            dump += _copy_block(f"corpus.{table}", ["id", "body"], rows)
        scans, _facts = _scan(dump)
        return scans

    def _manifest(self, declared) -> dict:
        return {Key.PROFILE: Profile.PUBLIC,
                Key.CORPUS: {Key.TABLE_ROWS: declared},
                Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}

    def test_every_declared_corpus_table_present_with_its_count_passes(self):
        scans = self._scans({"documents": [["1", "a"]], "pages": [["1", "b"], ["2", "c"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"documents": 1, "pages": 2}), scans, CORPUS)
        self.assertTrue(ok, detail)

    def test_a_declared_corpus_table_the_dump_never_carried_fails(self):
        """A corpus table declared and absent: the case that used to read
        as "cut correctly" because nothing described it.
        """
        scans = self._scans({"documents": [["1", "a"]], "pages": [["1", "b"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"documents": 1, "pages": 1, "embedding_model": 1}), scans, CORPUS)
        self.assertFalse(ok)
        self.assertIn("corpus.embedding_model", detail)

    def test_a_corpus_table_the_manifest_does_not_name_fails(self):
        scans = self._scans({"documents": [["1", "a"]], "pages": [["1", "b"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"documents": 1}), scans, CORPUS)
        self.assertFalse(ok)
        self.assertIn("corpus.pages", detail)

    def test_a_corpus_row_count_that_does_not_match_fails(self):
        scans = self._scans({"documents": [["1", "a"]]})
        ok, detail = table_rows_check.check_every_declared_table_shipped(
            self._manifest({"documents": 7}), scans, CORPUS)
        self.assertFalse(ok)
        self.assertIn("1 строк против 7", detail)

    def test_declaring_no_corpus_table_at_all_is_refused(self):
        """corpus travels in every profile, so an empty declaration is
        never "nothing to carry" -- it is a manifest this reader cannot
        hold the dump to.
        """
        for declared in (None, {}):
            with self.subTest(declared=declared):
                ok, detail = table_rows_check.check_every_declared_table_shipped(
                    self._manifest(declared), self._scans({"documents": [["1", "a"]]}), CORPUS)
                self.assertFalse(ok, detail)
                self.assertIn(Key.TABLE_ROWS, detail)


class DeclaredSchemasTests(unittest.TestCase):
    def test_both_classified_schemas_are_asked(self):
        """The set is derived from the maps that own their own name, so a
        schema whose columns the recipient classifies is one whose tables
        it counts.
        """
        self.assertEqual(set(table_rows_check.DECLARED_SCHEMAS), {CORPUS, CITATION})
        self.assertEqual(sorted(table_rows_check.DECLARATION_BLOCK),
                         sorted(table_rows_check.DECLARED_SCHEMAS))

    def test_the_set_is_the_column_classification_s_own(self):
        """Not "the same today": the same declaration. Which schemas are
        classified is column_class_checks.CLASSIFIED_SCHEMAS, and the
        manifest-block mapping here is derived through it rather than
        listing the schemas a second time.
        """
        self.assertEqual(sorted(table_rows_check.DECLARATION_BLOCK),
                         sorted(column_class_checks.CLASSIFIED_SCHEMAS))

    def test_a_classified_schema_with_no_manifest_block_cannot_be_derived(self):
        """The positive control, in the form of the defect: a schema
        classified next door and forgotten here raises where the mapping is
        built -- at import, i.e. a package that cannot be certified -- and
        does not resolve to a table count nobody asks for.
        """
        with self.assertRaises(KeyError):
            {schema: table_rows_check._BLOCK[schema]
             for schema in {**column_class_checks.CLASSIFIED_SCHEMAS, "future": object()}}


if __name__ == "__main__":
    unittest.main()
