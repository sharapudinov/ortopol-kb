"""Unit tests for deploy/citation_cut_checks.py: no live database, no real
dump -- a real dump_scan.scan() pass over hand-built COPY blocks, same
discipline as test_profile_checks.py's DumpScanTests (the check must be
right about actual bytes, not a mocked scan).

The facts these checks read are collected by citation_content_checks.
attach_visitors() on that same pass, so the scan helper below is the one
next door: a check written against hand-made fact containers would be a
check about a dict this repository never builds.
"""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_content_checks
import citation_cut_checks
import dump_scan
import profile_checks
from _artifact_fixtures import ArtifactBuilder, dump_facts, EXCLUDED_DOC, FULL_DOC
from manifest_keys import Key
from manifest_contract import CitationMode


def _copy_block(table: str, columns: list[str], rows: list[list[str]]) -> str:
    lines = [f"COPY {table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


def _scan(dump_text: str, mode: str = CitationMode.TOPOLOGY_ONLY) -> tuple[dict, dict]:
    """A real scan of `dump_text` under `mode`. Topology-only by default:
    that is the mode the content hunt exists for, and the mode most of
    these tests are about.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        row_visitors: dict = {}
        facts = citation_content_checks.attach_visitors(row_visitors, mode)
        scans = dump_scan.scan(dump_path, row_visitors).tables
    return scans, dump_facts(facts)


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


class CheckWorkDocumentsPresentTests(unittest.TestCase):
    """The citation slice and the corpus slice are cut by two different
    policies (citation.public_policy vs corpus.documents.public_distribution),
    and citation.work.document_id is a FK across that boundary. A work row
    naming a document the dump does not carry breaks the restore outright --
    and, before that, publishes the title of exactly the document the owner
    classified out.
    """

    MANIFEST = {Key.CITATION: {Key.CITATION_MODE: CitationMode.FULL_SKELETON}}

    def test_passes_when_every_named_document_is_in_the_dump(self):
        dump = (
            _copy_block("corpus.documents", ["id", "filename"], [["1997_sm280", "a.pdf"]])
            + _copy_block("citation.work", ["id", "key", "document_id"],
                          [["1", "k1", "1997_sm280"], ["2", "k2", "\\N"]])
        )
        _scans, facts = _scan(dump)
        facts.corpus.documents.add("1997_sm280")
        ok, detail = citation_cut_checks.check_work_documents_are_in_the_dump(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)

    def test_fails_when_a_work_names_a_document_the_dump_does_not_carry(self):
        dump = _copy_block("citation.work", ["id", "key", "document_id"],
                           [["1", "k1", "excluded_doc"]])
        _scans, facts = _scan(dump)
        facts.corpus.documents.add("1997_sm280")
        ok, detail = citation_cut_checks.check_work_documents_are_in_the_dump(
            self.MANIFEST, facts)
        self.assertFalse(ok)
        self.assertIn("excluded_doc", detail)

    def test_none_mode_is_a_trivial_pass(self):
        facts = {"citation_work_documents": {"whatever": {"k1"}}, "documents": set()}
        ok, detail = citation_cut_checks.check_work_documents_are_in_the_dump(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}, facts)
        self.assertTrue(ok, detail)


class CheckEdgesReferenceShippedWorksTests(unittest.TestCase):
    MANIFEST = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}

    def test_passes_when_both_endpoints_are_in_the_dump(self):
        dump = (
            _copy_block("citation.work", ["id", "key"], [["1", "k1"], ["2", "k2"]])
            + _copy_block("citation.cites", ["citing", "cited"], [["1", "2"]])
        )
        _scans, facts = _scan(dump)
        ok, detail = citation_cut_checks.check_edges_reference_shipped_works(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)

    def test_fails_on_an_edge_whose_endpoint_was_cut_away(self):
        dump = (
            _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
            + _copy_block("citation.cites", ["citing", "cited"], [["1", "2"]])
        )
        _scans, facts = _scan(dump)
        ok, detail = citation_cut_checks.check_edges_reference_shipped_works(
            self.MANIFEST, facts)
        self.assertFalse(ok)
        self.assertIn("2", detail)


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


# A journal row as the dump carries one: the three columns the cut matches
# against both vocabularies, plus the id every crawl_step row has.
JOURNAL_COLUMNS = ["id", "frontier_key", "candidate_key", "node_key"]


def _shipping_manifest(excluded: list[str], shipped: list[str]) -> dict:
    """A public manifest whose legal block classifies `excluded` out."""
    return {
        Key.PROFILE: "public",
        Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY},
        Key.LEGAL: {
            Key.DOCUMENTS_BY_DISTRIBUTION: {"full-text": shipped, "excluded": excluded},
            Key.SHIPPED_DISTRIBUTIONS: ["full-text"],
        },
    }


class CheckJournalNamesNothingCutTests(unittest.TestCase):
    """The journal cut, asserted from the file rather than from the WHERE
    clause that produced it.

    citation_profile._CUT_CTES derives cut_documents -> cut_keys ->
    cut_names -> cut_steps and matches THREE columns against TWO
    vocabularies in a three-branch UNION. It was the only cut in the package
    with no counterpart on this side, so a defect in any branch shipped
    journal rows naming documents the owner classified out while
    profile_checks printed a full column of [OK].
    """

    MANIFEST = _shipping_manifest([EXCLUDED_DOC], [FULL_DOC])

    def _facts(self, journal_rows: list[list[str]], work_rows=()) -> dict:
        dump = _copy_block("citation.work", ["id", "key"], list(work_rows))
        dump += _copy_block("citation.crawl_step", JOURNAL_COLUMNS, journal_rows)
        _scans, facts = _scan(dump)
        facts.corpus.documents.add(FULL_DOC)
        return facts

    def test_a_row_naming_an_excluded_document_in_frontier_key_is_a_leak(self):
        facts = self._facts([["1", EXCLUDED_DOC, "\\N", "\\N"]])
        ok, detail = citation_cut_checks.check_journal_names_nothing_cut(
            self.MANIFEST, facts)
        self.assertFalse(ok)
        self.assertIn(EXCLUDED_DOC, detail)

    def test_every_one_of_the_three_columns_is_watched(self):
        """Three separate UNION branches produce the cut, so a defect can
        live in exactly one of them.
        """
        for position in range(3):
            row = ["1", "\\N", "\\N", "\\N"]
            row[position + 1] = EXCLUDED_DOC
            with self.subTest(column=JOURNAL_COLUMNS[position + 1]):
                ok, detail = citation_cut_checks.check_journal_names_nothing_cut(
                    self.MANIFEST, self._facts([row]))
                self.assertFalse(ok, detail)

    def test_a_journal_naming_only_shipped_names_passes(self):
        facts = self._facts([["1", FULL_DOC, "k1", "k1"]], work_rows=[["1", "k1"]])
        ok, detail = citation_cut_checks.check_journal_names_nothing_cut(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)
        self.assertIn("none", detail)

    def test_a_dropped_candidate_is_not_a_leak_and_is_counted_as_undecidable(self):
        """A `drop` row names a candidate that failed tau and was never
        written to citation.work, so "every journal key is a work in the
        dump" would fail on a CORRECT package. The verdict counts those
        names instead of pretending the question was answered.
        """
        facts = self._facts([["1", "k1", "W_never_a_work", "\\N"]],
                            work_rows=[["1", "k1"]])
        ok, detail = citation_cut_checks.check_journal_names_nothing_cut(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)
        self.assertIn("1 naming neither", detail)

    def test_a_mode_that_ships_nothing_is_a_trivial_pass(self):
        manifest = dict(self.MANIFEST)
        manifest[Key.CITATION] = {Key.CITATION_MODE: CitationMode.NONE}
        ok, detail = citation_cut_checks.check_journal_names_nothing_cut(
            manifest, {"citation_journal_keys": {EXCLUDED_DOC}})
        self.assertTrue(ok, detail)


class JournalCutThroughTheWholePassTests(unittest.TestCase):
    """The same leak through profile_checks.run_checks(), on the artifact
    fixture: a real gzipped dump beside a real manifest, which is the input
    a recipient has.
    """

    WORK_COLUMNS = ["id", "key", "title", "abstract", "evidence"]

    def _builder(self, tmp, journal_rows):
        builder = ArtifactBuilder(Path(tmp))
        builder.schemas = ["corpus", "citation"]
        builder.citation = {
            "mode": CitationMode.TOPOLOGY_ONLY, "work_count": 1, "cites_count": 0,
            "work_columns": self.WORK_COLUMNS, "cites_columns": ["citing", "cited"],
            "work": [["1", "k1", "T1", "\\N", "\\N"]], "cites": [],
            "crawl_step_columns": JOURNAL_COLUMNS, "crawl_step": journal_rows,
        }
        return builder

    def _results(self, builder) -> dict:
        return {name: (ok, detail)
                for name, ok, detail in profile_checks.run_checks(builder.write())}

    def test_a_clean_journal_certifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self._results(self._builder(tmp, [["1", "k1", "k1", "k1"]]))
        for name, (ok, detail) in results.items():
            self.assertTrue(ok, f"{name}: {detail}")

    def test_a_journal_row_naming_the_excluded_document_fails_the_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder(tmp, [["1", "k1", "k1", "k1"],
                                          ["2", EXCLUDED_DOC, "k1", "\\N"]])
            results = self._results(builder)
        name = next(n for n in results if n.startswith("citation.crawl_step"))
        ok, detail = results[name]
        self.assertFalse(ok)
        self.assertIn(EXCLUDED_DOC, detail)

    def test_a_package_whose_journal_never_shipped_no_longer_certifies(self):
        """The vacuity, end to end: with no crawl_step block in the dump the
        journal check finds no names, reports a green nought and says the
        cut holds. What refuses the package is the declaration -- the
        manifest says the journal is in there, and it is not.
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder(tmp, [["1", "k1", "k1", "k1"]])
            builder.citation["crawl_step_columns"] = None
            builder.citation["table_rows"] = {"work": 1, "cites": 0, "crawl_step": 1}
            results = self._results(builder)
        journal = next(n for n in results if n.startswith("citation.crawl_step"))
        self.assertTrue(results[journal][0], "проверка журнала и не могла упасть")
        declared = next(n for n in results if "заявленная таблица" in n)
        self.assertFalse(results[declared][0])
        self.assertIn("citation.crawl_step", results[declared][1])

    def test_the_cli_exits_nonzero_on_such_a_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder(tmp, [["2", EXCLUDED_DOC, "\\N", "\\N"]])
            directory = builder.write()
            self.assertEqual(profile_checks.main(["--artifact-dir", str(directory)]), 1)




class FactsAreReadByNameTests(unittest.TestCase):
    """The pass hands the checks a record, so a fact that never arrived
    raises where it is read.

    Read out of a dict with a default, every check here answered "absent
    from the dump: none" over an empty set -- a green row for a question
    nothing had asked. The drift was not hypothetical: the six fact names
    were spelled once in the visitor module and again in each consumer, and
    a visitor that never fired (a COPY header spelling the table
    differently, a checker older than the bundle it travels in) produced
    the same empty containers as a clean dump.
    """

    MANIFEST = {Key.CITATION: {Key.CITATION_MODE: CitationMode.FULL_SKELETON}}
    CHECKS = ("check_work_documents_are_in_the_dump",
              "check_edges_reference_shipped_works",
              "check_journal_names_nothing_cut")

    class _Drifted:
        """The facts record one rename out of date: nothing this check asks
        for is on it."""

    def test_every_check_refuses_a_record_the_facts_are_not_on(self):
        facts = profile_checks.DumpFacts(corpus=self._Drifted(), citation=self._Drifted())
        for name in self.CHECKS:
            with self.subTest(check=name), self.assertRaises(AttributeError):
                getattr(citation_cut_checks, name)(self.MANIFEST, facts)

    def test_a_fact_that_is_present_and_empty_is_still_the_green_answer(self):
        """The complement: a dump that really carried nothing IS a pass.
        What must not be reachable is the same verdict without looking."""
        for name in self.CHECKS:
            with self.subTest(check=name):
                ok, detail = getattr(citation_cut_checks, name)(
                    self.MANIFEST, dump_facts())
                self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()
