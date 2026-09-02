"""Unit tests for deploy/citation_content_checks.py: no live database, no
real dump -- a real dump_scan.scan() pass over hand-built COPY blocks, same
discipline as test_profile_checks.py's DumpScanTests (the check must be
right about actual bytes, not a mocked scan).
"""
from __future__ import annotations

import gzip
import pathlib
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_columns
import citation_content_checks
import dump_scan
import profile_checks
from manifest_contract import CitationMode, Key, PolicySource, Profile


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
        scans = dump_scan.scan(dump_path, row_visitors)
    return scans, dict(facts)


class CheckCitationSchemaMatchesModeTests(unittest.TestCase):
    def test_none_mode_passes_when_dump_carries_neither_table(self):
        scans, _facts = _scan("CREATE TABLE corpus.documents (id text);\n")
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}
        ok, detail = citation_content_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertTrue(ok, detail)

    def test_none_mode_fails_when_a_citation_table_leaked_in(self):
        dump = _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}}
        ok, detail = citation_content_checks.check_citation_schema_matches_mode(manifest, scans)
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
        ok, detail = citation_content_checks.check_citation_schema_matches_mode(manifest, scans)
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
        ok, detail = citation_content_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertFalse(ok)

    def test_shipping_mode_fails_when_a_table_is_entirely_absent(self):
        dump = _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
        scans, _facts = _scan(dump)
        manifest = {Key.CITATION: {
            Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY, Key.WORK_COUNT: 1, Key.CITES_COUNT: 0,
        }}
        ok, _detail = citation_content_checks.check_citation_schema_matches_mode(manifest, scans)
        self.assertFalse(ok)


class CheckTopologyOnlyStripsTests(unittest.TestCase):
    def _sample(self, items):
        sample = citation_content_checks.LeakSample()
        for item in items:
            sample.add(item)
        return {"citation_leaked": sample}

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
        facts["documents"] = {"1997_sm280"}
        ok, detail = citation_content_checks.check_work_documents_are_in_the_dump(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)

    def test_fails_when_a_work_names_a_document_the_dump_does_not_carry(self):
        dump = _copy_block("citation.work", ["id", "key", "document_id"],
                           [["1", "k1", "excluded_doc"]])
        _scans, facts = _scan(dump)
        facts["documents"] = {"1997_sm280"}
        ok, detail = citation_content_checks.check_work_documents_are_in_the_dump(
            self.MANIFEST, facts)
        self.assertFalse(ok)
        self.assertIn("excluded_doc", detail)

    def test_none_mode_is_a_trivial_pass(self):
        facts = {"citation_work_documents": {"whatever": {"k1"}}, "documents": set()}
        ok, detail = citation_content_checks.check_work_documents_are_in_the_dump(
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
        ok, detail = citation_content_checks.check_edges_reference_shipped_works(
            self.MANIFEST, facts)
        self.assertTrue(ok, detail)

    def test_fails_on_an_edge_whose_endpoint_was_cut_away(self):
        dump = (
            _copy_block("citation.work", ["id", "key"], [["1", "k1"]])
            + _copy_block("citation.cites", ["citing", "cited"], [["1", "2"]])
        )
        _scans, facts = _scan(dump)
        ok, detail = citation_content_checks.check_edges_reference_shipped_works(
            self.MANIFEST, facts)
        self.assertFalse(ok)
        self.assertIn("2", detail)


class VisitorsCostOnlyWhatTheModeAsksTests(unittest.TestCase):
    """The content hunt is a no-op under every mode but topology-only, and
    a no-op that is registered is not free: dump_scan builds a dict per row
    for every table that HAS a visitor, and citation.crawl_step grows by
    ~100k rows per depth-2 crawl.
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

    def test_full_skeleton_registers_only_the_two_that_carry_facts(self):
        visitors: dict = {}
        citation_content_checks.attach_visitors(visitors, CitationMode.FULL_SKELETON)
        self.assertEqual(sorted(visitors), ["citation.cites", "citation.work"])
        extra = [t for t in self.CONTENT_TABLES
                 if t not in ("citation.work", "citation.cites")]
        self.assertTrue(extra, "нет таблицы, на которой видна разница")
        for table in extra:
            self.assertNotIn(table, visitors)

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
        self.assertEqual(facts["citation_leaked"].total, 0)
        self.assertEqual(facts["citation_work_ids"], {"1"})
        self.assertEqual(facts["citation_edge_endpoints"], {"1"})

    def test_the_same_dump_under_topology_only_does_report_them(self):
        """The complement, so the quiet mode cannot pass by never looking."""
        dump = _copy_block("citation.work", ["id", "key", "abstract"],
                            [["1", "k1", "an abstract that must not ship"]])
        _scans, facts = _scan(dump, CitationMode.TOPOLOGY_ONLY)
        self.assertEqual(facts["citation_leaked"].sample, ["citation.work.abstract:k1"])
        self.assertEqual(facts["citation_leaked"].total, 1)


class PolicySourceTests(unittest.TestCase):
    """Who decided the citation mode, asked of the package rather than of
    its filename. An override build is otherwise indistinguishable from an
    owner-classified one to every consumer: same profile, same schemas,
    same counts, a consistent dump -- and the name is not part of the file.

    The answer is profile-dependent: only public applies a policy, so only
    public may name the owner, and a full artifact must say so rather than
    assert a decision the packager never read.
    """

    def _manifest(self, profile=Profile.PUBLIC, **citation):
        base = {Key.CITATION_MODE: CitationMode.FULL_SKELETON,
                Key.WORK_COUNT: 1, Key.CITES_COUNT: 0}
        base.update(citation)
        return {Key.PROFILE: profile, Key.CITATION: base}

    # (profile, policy_source) -> may this artifact be certified.
    COMBINATIONS = {
        (Profile.PUBLIC, PolicySource.OWNER): True,
        (Profile.PUBLIC, PolicySource.OVERRIDE): False,
        (Profile.PUBLIC, PolicySource.NOT_APPLICABLE): False,
        (Profile.PUBLIC, None): False,
        (Profile.PUBLIC, "somebody-else"): False,
        (Profile.FULL, PolicySource.NOT_APPLICABLE): True,
        (Profile.FULL, PolicySource.OWNER): False,
        (Profile.FULL, PolicySource.OVERRIDE): False,
        (Profile.FULL, None): False,
        (Profile.FULL, "somebody-else"): False,
    }

    def test_every_profile_and_source_pair_is_judged(self):
        for (profile, source), expected in self.COMBINATIONS.items():
            with self.subTest(profile=profile, source=source):
                citation = {} if source is None else {
                    Key.CITATION_POLICY_SOURCE: source}
                ok, detail = citation_content_checks.check_policy_is_the_owners(
                    self._manifest(profile, **citation))
                self.assertEqual(ok, expected, detail)

    def test_a_full_artifact_does_not_claim_the_owner_decided(self):
        """The packager never reads citation.public_policy for full, so
        naming the owner there is a provenance nobody supplied.
        """
        ok, detail = citation_content_checks.check_policy_is_the_owners(
            self._manifest(Profile.FULL,
                           **{Key.CITATION_POLICY_SOURCE: PolicySource.OWNER}))
        self.assertFalse(ok)
        self.assertIn(PolicySource.NOT_APPLICABLE, detail)

    def test_a_public_artifact_may_not_duck_the_question(self):
        ok, detail = citation_content_checks.check_policy_is_the_owners(
            self._manifest(Profile.PUBLIC,
                           **{Key.CITATION_POLICY_SOURCE:
                              PolicySource.NOT_APPLICABLE}))
        self.assertFalse(ok)
        self.assertIn(PolicySource.OWNER, detail)

    def test_an_override_build_is_refused_and_says_why(self):
        ok, detail = citation_content_checks.check_policy_is_the_owners(
            self._manifest(**{Key.CITATION_POLICY_SOURCE: PolicySource.OVERRIDE}))
        self.assertFalse(ok)
        self.assertIn("--policy-override", detail)
        self.assertIn("публиковать нельзя", detail)

    def test_an_override_is_refused_even_when_it_shipped_no_graph(self):
        """`--policy-override none` carries no citation schema at all, so
        nothing leaked -- but the refusal is about whose decision it was.
        """
        ok, detail = citation_content_checks.check_policy_is_the_owners({
            Key.PROFILE: Profile.PUBLIC,
            Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                           Key.CITATION_POLICY_SOURCE: PolicySource.OVERRIDE}})
        self.assertFalse(ok)
        self.assertIn("--policy-override", detail)

    def test_a_shipping_manifest_with_no_source_is_refused_not_defaulted(self):
        """Read with a default, an artifact predating the field would be
        certified as owner-classified -- which is the one outcome the flag
        must never be able to produce.
        """
        ok, detail = citation_content_checks.check_policy_is_the_owners(self._manifest())
        self.assertFalse(ok)
        self.assertIn("None", detail)

    def test_an_artifact_carrying_no_graph_has_no_policy_to_certify(self):
        ok, detail = citation_content_checks.check_policy_is_the_owners(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}})
        self.assertTrue(ok, detail)
        ok, detail = citation_content_checks.check_policy_is_the_owners({})
        self.assertTrue(ok, detail)

    def test_profile_checks_runs_it(self):
        """The check is only worth anything if the pass that certifies an
        artifact actually calls it -- profile_checks.run_checks() is what
        both the standalone CLI and smoke_test.py go through.
        """
        source = pathlib.Path(profile_checks.__file__).read_text(encoding="utf-8")
        self.assertIn("check_policy_is_the_owners", source)


if __name__ == "__main__":
    unittest.main()
