"""What the manifest DECLARES about the citation rows, held to the rows the
dump turns out to hold (deploy/citation_cut_checks.py).

Three claims, one subject: the schema and the two headline totals under the
declared mode, every table named in table_rows with exactly that many rows,
and the kind census. Split from test_citation_cut_checks.py for module size
(kb/CLAUDE.md FILE_SIZE) along the seam the checked module's own docstring
already draws -- everything there asks whether the rows that shipped name
only what this package carries, everything here asks whether there are as
many of them as the manifest says.

Same discipline: a real dump_scan.scan() pass over hand-built COPY blocks,
with the citation visitors attached (_artifact_fixtures.citation_scan), so
a check is held to actual bytes rather than to a mocked scan.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_cut_checks
from _artifact_fixtures import (
    citation_copy_block as _copy_block,
    citation_scan as _scan,
)
from manifest_keys import Key
from manifest_contract import CitationMode


class CheckCitationSchemaMatchesModeTests(unittest.TestCase):
    def test_none_mode_passes_when_dump_carries_neither_table(self):
        scans, _facts = _scan("CREATE TABLE corpus.documents (id text);\n")
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}
        ok, detail = citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertTrue(ok, detail)

    def test_none_mode_fails_when_a_citation_table_leaked_in(self):
        dump = _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}
        ok, detail = citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertFalse(ok)
        self.assertIn("citation.work", detail)

    def test_shipping_mode_requires_both_tables_with_matching_counts(self):
        dump = (
            _copy_block("citation.work", ["id", "key"], [["1", "k1"], ["2", "k2"]])
            + _copy_block("citation.cites", ["citing", "cited"], [["1", "2"]])
        )
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {
            Key.CITATION_MODE: CitationMode.FULL_SKELETON, Key.WORK_COUNT: 2, Key.CITES_COUNT: 1,
        }}
        ok, detail = citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertTrue(ok, detail)

    def test_shipping_mode_fails_on_a_row_count_mismatch(self):
        dump = (
            _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
            + _copy_block("citation.cites", ["citing", "cited"], [])
        )
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {
            Key.CITATION_MODE: CitationMode.FULL_SKELETON, Key.WORK_COUNT: 5, Key.CITES_COUNT: 0,
        }}
        ok, detail = citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertFalse(ok)

    def test_shipping_mode_fails_when_a_table_is_entirely_absent(self):
        dump = _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {
            Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY, Key.WORK_COUNT: 1, Key.CITES_COUNT: 0,
        }}
        ok, _detail = citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertFalse(ok)


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
        return {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY,
                               Key.TABLE_ROWS: declared}}

    def test_every_declared_table_present_with_its_count_passes(self):
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"], ["2", "k1"]]})
        ok, detail = citation_cut_checks.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 2}), scans)
        self.assertTrue(ok, detail)

    def test_a_declared_table_the_dump_never_carried_is_the_whole_point(self):
        """The journal is declared and absent -- which is exactly the
        package on which check_journal_names_nothing_cut() reported zero
        names and passed.
        """
        scans = self._scans({"work": [["1", "k1"]]})
        ok, detail = citation_cut_checks.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 604}), scans)
        self.assertFalse(ok)
        self.assertIn("citation.crawl_step", detail)
        self.assertIn("604", detail)

    def test_a_row_count_that_does_not_match_the_declaration_fails(self):
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"]]})
        ok, detail = citation_cut_checks.check_every_declared_table_shipped(
            self._manifest({"work": 1, "crawl_step": 2}), scans)
        self.assertFalse(ok)
        self.assertIn("1 строк против 2", detail)

    def test_a_table_in_the_dump_that_the_manifest_does_not_name_fails(self):
        """The other direction: a table shipped without being described is
        a slice of the schema nothing on this side can hold to anything.
        """
        scans = self._scans({"work": [["1", "k1"]], "crawl_step": [["1", "k1"]]})
        ok, detail = citation_cut_checks.check_every_declared_table_shipped(
            self._manifest({"work": 1}), scans)
        self.assertFalse(ok)
        self.assertIn("citation.crawl_step", detail)

    def test_a_shipping_mode_declaring_no_table_at_all_is_refused(self):
        for declared in (None, {}, "нет"):
            with self.subTest(declared=declared):
                ok, detail = citation_cut_checks.check_every_declared_table_shipped(
                    self._manifest(declared), self._scans({"work": [["1", "k1"]]}))
                self.assertFalse(ok, detail)
                self.assertIn(Key.TABLE_ROWS, detail)

    def test_a_mode_that_ships_nothing_declares_nothing_and_carries_nothing(self):
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE, Key.TABLE_ROWS: {}}}
        ok, detail = citation_cut_checks.check_every_declared_table_shipped(
            manifest, self._scans({}))
        self.assertTrue(ok, detail)

    def test_a_mode_that_ships_nothing_may_not_declare_a_table_either(self):
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                                   Key.TABLE_ROWS: {"work": 1}}}
        ok, _detail = citation_cut_checks.check_every_declared_table_shipped(
            manifest, self._scans({}))
        self.assertFalse(ok)


class KindCensusTests(unittest.TestCase):
    """manifest.citation.work_by_kind against the work rows the dump holds.

    It was the one number in the block read from the live database beside
    the dump, and the one nothing on this side re-derived -- while three
    docstrings said this module did. sum(work_by_kind) could disagree with
    work_count and with the file, and every bundled check still printed
    [OK].
    """

    COLUMNS = ["id", "key", "kind"]

    def _facts(self, rows):
        dump = _copy_block("citation.work", self.COLUMNS, rows)
        _scans, facts = _scan(dump)
        return facts

    def _manifest(self, declared):
        return {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY,
                               Key.WORK_BY_KIND: declared}}

    def test_the_census_of_the_shipped_rows_certifies(self):
        facts = self._facts([["1", "k1", "our-document"], ["2", "k2", "excluded"],
                             ["3", "k3", "our-document"]])
        ok, detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest({"our-document": 2, "excluded": 1}), facts)
        self.assertTrue(ok, detail)

    def test_a_kind_the_manifest_over_counts_is_named(self):
        facts = self._facts([["1", "k1", "our-document"]])
        ok, detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest({"our-document": 56}), facts)
        self.assertFalse(ok)
        self.assertIn("our-document", detail)

    def test_two_errors_that_cancel_in_the_sum_still_fail(self):
        """Compared kind by kind, which is what a census is for: a total
        would agree with itself here.
        """
        facts = self._facts([["1", "k1", "our-document"], ["2", "k2", "excluded"]])
        declared = {"our-document": 2, "excluded": 0}
        self.assertEqual(sum(declared.values()), 2)
        ok, _detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest(declared), facts)
        self.assertFalse(ok)

    def test_a_kind_the_dump_carries_and_the_manifest_omits_is_a_leak(self):
        facts = self._facts([["1", "k1", "our-document"], ["2", "k2", "indexed"]])
        ok, detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest({"our-document": 1}), facts)
        self.assertFalse(ok)
        self.assertIn("indexed", detail)

    def test_a_work_block_with_no_kind_column_cannot_certify(self):
        """ARTIFACT_SIDE_FAILS_CLOSED: rows the visitor could not classify
        are counted under the wire format's own NULL, which no census a
        packager writes can equal. Left uncounted they would make the
        manifest's census true by shrinking the dump's.
        """
        dump = _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
        _scans, facts = _scan(dump)
        ok, _detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest({"our-document": 1}), facts)
        self.assertFalse(ok)

    def test_a_census_that_is_not_a_dictionary_is_refused(self):
        ok, detail = citation_cut_checks.check_kind_census_matches_manifest(
            self._manifest(None), self._facts([]))
        self.assertFalse(ok)
        self.assertIn("work_by_kind", detail)

    def test_a_profile_that_ships_no_graph_has_no_census_to_check(self):
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                                   Key.WORK_BY_KIND: {}}}
        ok, _detail = citation_cut_checks.check_kind_census_matches_manifest(
            manifest, self._facts([]))
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
