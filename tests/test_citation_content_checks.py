"""Unit tests for deploy/citation_content_checks.py: no live database, no
real dump -- a real dump_scan.scan() pass over hand-built COPY blocks, same
discipline as test_profile_checks.py's DumpScanTests (the check must be
right about actual bytes, not a mocked scan).
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

import artifact_bundle
import block_census
import citation_columns
import citation_content_checks
import citation_dump
import citation_profile
import copy_rows
import dump_scan
from _artifact_fixtures import dump_facts
from citation_columns import CENSUS_COLUMN, CENSUS_TABLE, JOURNAL_KEY_COLUMNS
from manifest_keys import Key
from manifest_contract import CitationMode

DEPLOY_DIR = Path(citation_columns.__file__).resolve().parent
OWNER = "citation_columns.py"


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


class CheckTopologyOnlyStripsTests(unittest.TestCase):
    def _sample(self, items):
        facts = dump_facts()
        for item in items:
            facts.citation.leaked.add(item)
        return facts

    def test_a_declared_full_content_mode_is_a_trivial_pass(self):
        ok, detail = citation_content_checks.check_content_is_stripped(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.FULL_SKELETON}},
            self._sample(["x"]),
        )
        self.assertTrue(ok, detail)

    def test_topology_only_passes_when_nothing_leaked(self):
        ok, detail = citation_content_checks.check_content_is_stripped(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}},
            self._sample([]),
        )
        self.assertTrue(ok, detail)

    def test_a_hundred_leaks_are_counted_in_full_and_quoted_in_part(self):
        """citation.crawl_step grows by ~100k rows per depth-2 crawl and
        every offending row used to be formatted into one string and then
        interpolated into one message. The verdict needs the SIZE of the
        breach and enough of it to find the rest.
        """
        ok, detail = citation_content_checks.check_content_is_stripped(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}},
            self._sample([f"citation.crawl_step.reason:{i}" for i in range(100)]),
        )
        self.assertFalse(ok)
        self.assertIn("leaked 100 row(s)", detail)
        self.assertIn(f"first {citation_content_checks.LEAK_SAMPLE}", detail)
        self.assertIn("citation.crawl_step.reason:0", detail)
        self.assertNotIn("citation.crawl_step.reason:99", detail)
        self.assertLess(len(detail), 2000)

    def test_topology_only_fails_when_a_work_abstract_leaked(self):
        dump = _copy_block("citation.work", ["id", "key", "abstract", "evidence"],
                            [["1", "k1", "an abstract", "\\N"]])
        scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_content_is_stripped(
            manifest, facts,
        )
        self.assertFalse(ok)
        self.assertIn("abstract", detail)

    def test_topology_only_fails_when_cites_evidence_leaked(self):
        dump = _copy_block("citation.cites", ["citing", "cited", "evidence"],
                            [["1", "2", '{"src": "openalex"}']])
        _scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_content_is_stripped(
            manifest, facts,
        )
        self.assertFalse(ok)
        self.assertIn("evidence", detail)

    def test_topology_only_fails_when_the_journal_prose_leaked(self):
        """crawl_step is not scanned for facts, only for content -- so the
        visitor has to be registered from the classification map rather than
        by hand, or the one table nobody collects anything from ships its
        prose unchecked.
        """
        dump = _copy_block("citation.crawl_step", ["id", "action", "reason"],
                            [["1", "keep", "kept"]])
        _scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_content_is_stripped(
            manifest, facts,
        )
        self.assertFalse(ok)
        self.assertIn("crawl_step.reason", detail)

    def test_every_content_column_of_every_table_is_watched(self):
        """The checker's coverage IS the map's content set -- no column
        classified content can be one nothing here looks at.
        """
        visitors: dict = {}
        citation_content_checks.attach_visitors(visitors, CitationMode.TOPOLOGY_ONLY)
        for table, columns in citation_columns.CITATION_COLUMN_CLASS.items():
            content = {c for c, kind in columns.items()
                       if kind == citation_columns.CONTENT}
            if content:
                self.assertIn(f"citation.{table}", visitors, table)

    def test_topology_only_passes_when_evidence_is_null(self):
        dump = _copy_block("citation.work", ["id", "key", "abstract", "evidence"],
                            [["1", "k1", "\\N", "\\N"]])
        _scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_content_is_stripped(
            manifest, facts,
        )
        self.assertTrue(ok, detail)


class VisitorsCostOnlyWhatTheModeAsksTests(unittest.TestCase):
    """The content hunt is a no-op under every mode but topology-only, and
    a no-op that is registered is not free: dump_scan builds a dict per row
    for every table that HAS a visitor.

    Three tables are registered whatever the mode, because they carry facts
    the cut checks read -- crawl_step among them, at ~100k rows per depth-2
    crawl, which is exactly the cost the earlier `if stripping` saved and
    exactly why the journal cut had no artifact-side check at all. What the
    mode still decides is the tables registered for the HUNT alone.
    """

    CONTENT_TABLES = sorted(
        f"citation.{table}" for table, columns
        in citation_columns.CITATION_COLUMN_CLASS.items()
        if any(kind == citation_columns.CONTENT for kind in columns.values())
    )

    def test_topology_only_watches_every_table_with_content(self):
        visitors: dict = {}
        citation_content_checks.attach_visitors(visitors, CitationMode.TOPOLOGY_ONLY)
        for table in self.CONTENT_TABLES:
            self.assertIn(table, visitors)

    FACT_TABLES = ["citation.cites", "citation.crawl_step", "citation.work"]
    # A table classified as carrying content but holding no fact any check
    # reads -- the only shape the hunt-only registration is still about, and
    # the schema has none today (all three content tables carry facts). A
    # synthetic one is what keeps the branch tested rather than assumed.
    FUTURE_TABLE = {"note": citation_columns.CONTENT, "id": citation_columns.TOPOLOGY}

    def test_full_skeleton_registers_only_the_three_that_carry_facts(self):
        visitors: dict = {}
        with mock.patch.dict(citation_columns.CITATION_COLUMN_CLASS,
                             {"future_table": self.FUTURE_TABLE}):
            citation_content_checks.attach_visitors(visitors, CitationMode.FULL_SKELETON)
        self.assertEqual(sorted(visitors), self.FACT_TABLES)

    def test_topology_only_also_watches_a_content_table_carrying_no_facts(self):
        """The complement, on the same synthetic table: the hunt-only
        registration is what a table added to the classification gets, and a
        branch nothing exercises is a branch nothing holds.
        """
        visitors: dict = {}
        with mock.patch.dict(citation_columns.CITATION_COLUMN_CLASS,
                             {"future_table": self.FUTURE_TABLE}):
            citation_content_checks.attach_visitors(visitors, CitationMode.TOPOLOGY_ONLY)
        self.assertEqual(sorted(visitors),
                         sorted(self.FACT_TABLES + ["citation.future_table"]))

    def test_the_journal_is_visited_under_every_mode(self):
        """The whole of the journal fix: the largest and most delicately cut
        table in the schema used to get a visitor only when there was
        content to hunt, so under full-skeleton the recipient could learn
        nothing at all about the cut that produced it.
        """
        for mode in (CitationMode.FULL_SKELETON, CitationMode.TOPOLOGY_ONLY,
                     CitationMode.NONE, "a-mode-nobody-declared"):
            visitors: dict = {}
            with self.subTest(mode=mode):
                citation_content_checks.attach_visitors(visitors, mode)
                self.assertIn("citation.crawl_step", visitors)

    def test_full_skeleton_collects_no_leak_facts_while_scanning(self):
        """The rows go past the work/cites visitors either way -- what must
        not happen is `leaked` filling up with findings nobody reads.
        """
        dump = (
            _copy_block("citation.work", ["id", "key", "abstract"],
                        [["1", "k1", "an abstract that ships in this mode"]])
            + _copy_block("citation.cites", ["citing", "cited", "evidence"],
                          [["1", "1", "{cited by p. 5}"]])
        )
        _scans, facts = _scan(dump, CitationMode.FULL_SKELETON)
        self.assertEqual(facts.citation.leaked.total, 0)
        self.assertEqual(facts.citation.work_ids, {"1"})
        self.assertEqual(facts.citation.edge_endpoints, {"1"})

    def test_the_same_dump_under_topology_only_does_report_them(self):
        """The complement, so the quiet mode cannot pass by never looking."""
        dump = _copy_block("citation.work", ["id", "key", "abstract"],
                            [["1", "k1", "an abstract that must not ship"]])
        _scans, facts = _scan(dump, CitationMode.TOPOLOGY_ONLY)
        self.assertEqual(facts.citation.leaked.sample, ["citation.work.abstract:k1"])
        self.assertEqual(facts.citation.leaked.total, 1)


def _bare_strings(path: Path) -> set[str]:
    """Every string LITERAL a module evaluates, docstrings excepted.

    Prose about a column is not a use of it -- the comments and docstrings
    that explain why the journal has three key columns must stay readable
    where they are.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documentation = {ast.get_docstring(node, clean=False)
                     for node in ast.walk(tree)
                     if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef))}
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in documentation}


class JournalKeyColumnsAreDeclaredOnceTests(unittest.TestCase):
    """The producer's cut and the bundled checker answer "which columns does
    a journal row name something in" from ONE declaration.

    Two hand-written copies could only agree by accident, and crawl_step
    grows a column at a time (JOURNAL_FACTS_ARE_COLUMNS): a fourth key
    column added to the UNION alone leaves the recipient blind to the cut
    the check exists to verify, added to the checker alone fails a correct
    package. Either divergence is silent until it fires.
    """

    def test_only_citation_columns_spells_a_key_column(self):
        for path in sorted(DEPLOY_DIR.rglob("*.py")):
            if path.name == OWNER:
                continue
            with self.subTest(module=path.name):
                self.assertFalse(
                    _bare_strings(path) & set(JOURNAL_KEY_COLUMNS),
                    f"{path.name}: имя ключевой колонки журнала набрано руками; "
                    f"импортируйте JOURNAL_KEY_COLUMNS из {OWNER}")

    def test_the_checker_collects_exactly_the_declared_columns(self):
        self.assertIs(citation_content_checks.JOURNAL_KEY_COLUMNS, JOURNAL_KEY_COLUMNS)

    def test_the_cut_joins_once_per_declared_column(self):
        ctes = citation_profile.crawl_step_cut_ctes()
        for column in JOURNAL_KEY_COLUMNS:
            self.assertEqual(ctes.count(f"j.{column} = r.ref"), 1, column)
        self.assertEqual(ctes.count("JOIN cut_names r ON"), len(JOURNAL_KEY_COLUMNS))

    def test_a_column_added_to_the_declaration_reaches_the_cut(self):
        """The declaration is what the SQL is BUILT from, so the branch
        arrives without citation_profile.py being edited at all.
        """
        with mock.patch.object(citation_profile, "JOURNAL_KEY_COLUMNS",
                               (*JOURNAL_KEY_COLUMNS, "successor_key")):
            ctes = citation_profile.crawl_step_cut_ctes()
        self.assertIn("j.successor_key = r.ref", ctes)
        self.assertEqual(ctes.count("JOIN cut_names r ON"), len(JOURNAL_KEY_COLUMNS) + 1)


class CensusColumnIsDeclaredOnceTests(unittest.TestCase):
    """WHICH column the manifest's work_by_kind counts is one declaration,
    read by both sides of the artifact boundary.

    Three modules tally this column -- citation_dump.py for the public
    profile, artifact_bundle.py for the full one, and the bundled
    citation_content_checks.py re-tallying the shipped file -- and only the
    third travels. Re-typed on the producer's side alone, the checker counts
    every work row under the wire format's NULL and fails a correct package;
    re-typed on the checker's side alone, it certifies a census nothing
    wrote. Neither failure says which of the two copies moved.
    """

    def test_only_citation_columns_spells_the_census_column(self):
        for path in sorted(DEPLOY_DIR.rglob("*.py")):
            if path.name == OWNER:
                continue
            with self.subTest(module=path.name):
                self.assertNotIn(
                    CENSUS_COLUMN, _bare_strings(path),
                    f"{path.name}: имя колонки переписи набрано руками; "
                    f"импортируйте CENSUS_COLUMN из {OWNER}")

    def test_every_tally_reads_the_same_declaration(self):
        self.assertIs(citation_dump.CENSUS_COLUMN, CENSUS_COLUMN)
        self.assertIs(citation_dump.CENSUS_TABLE, CENSUS_TABLE)
        self.assertIs(artifact_bundle.CENSUS_COLUMN, CENSUS_COLUMN)
        self.assertIs(artifact_bundle.CENSUS_TABLE, CENSUS_TABLE)
        self.assertIs(citation_content_checks.CENSUS_COLUMN, CENSUS_COLUMN)

    def test_the_census_column_is_topology_and_therefore_always_shipped(self):
        """A content column would be blanked under topology-only, and a
        census of blanks is one no manifest number can equal.
        """
        self.assertEqual(
            citation_columns.citation_column_class(CENSUS_TABLE, CENSUS_COLUMN),
            citation_columns.TOPOLOGY)

    def test_the_declared_block_is_the_one_the_full_profile_counts(self):
        """The full profile recognises its census block by the qualified
        name in the COPY header, so the schema and the table have to be the
        producer's own two names rather than a third spelling.
        """
        counter = copy_rows.CopyBlockCounter(
            io.BytesIO(),
            block_census.BlockCensus(f"{copy_rows.CITATION_SCHEMA}.{CENSUS_TABLE}",
                                     block_census.FieldTally(CENSUS_COLUMN)))
        counter.write(f"COPY citation.{CENSUS_TABLE} (id, {CENSUS_COLUMN}) "
                      "FROM stdin;\n1\tour-document\n\\.\n".encode())
        counter.finish()
        self.assertEqual(counter.census.tally.counts, {"our-document": 1})


if __name__ == "__main__":
    unittest.main()
