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
from manifest_contract import CitationMode, Key, PolicySource


def _copy_block(table: str, columns: list[str], rows: list[list[str]]) -> str:
    lines = [f"COPY {table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


def _scan(dump_text: str) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        row_visitors: dict = {}
        facts = citation_content_checks.attach_visitors(row_visitors)
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
    def test_non_topology_only_mode_is_a_trivial_pass(self):
        ok, detail = citation_content_checks.check_topology_only_strips_content(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.FULL_SKELETON}}, {"citation_leaked": ["x"]},
        )
        self.assertTrue(ok, detail)

    def test_topology_only_passes_when_nothing_leaked(self):
        ok, detail = citation_content_checks.check_topology_only_strips_content(
            {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}, {"citation_leaked": []},
        )
        self.assertTrue(ok, detail)

    def test_topology_only_fails_when_a_work_abstract_leaked(self):
        dump = _copy_block("citation.work", ["id", "key", "abstract", "evidence"],
                            [["1", "k1", "an abstract", "\\N"]])
        scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_topology_only_strips_content(
            manifest, facts,
        )
        self.assertFalse(ok)
        self.assertIn("abstract", detail)

    def test_topology_only_fails_when_cites_evidence_leaked(self):
        dump = _copy_block("citation.cites", ["citing", "cited", "evidence"],
                            [["1", "2", '{"src": "openalex"}']])
        _scans, facts = _scan(dump)
        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.TOPOLOGY_ONLY}}
        ok, detail = citation_content_checks.check_topology_only_strips_content(
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
        ok, detail = citation_content_checks.check_topology_only_strips_content(
            manifest, facts,
        )
        self.assertFalse(ok)
        self.assertIn("crawl_step.reason", detail)

    def test_every_content_column_of_every_table_is_watched(self):
        """The checker's coverage IS the map's content set -- no column
        classified content can be one nothing here looks at.
        """
        visitors: dict = {}
        citation_content_checks.attach_visitors(visitors)
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
        ok, detail = citation_content_checks.check_topology_only_strips_content(
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


class PolicySourceTests(unittest.TestCase):
    """Who decided the citation mode, asked of the package rather than of
    its filename. An override build is otherwise indistinguishable from an
    owner-classified one to every consumer: same profile, same schemas,
    same counts, a consistent dump -- and the name is not part of the file.
    """

    def _manifest(self, **citation):
        base = {Key.CITATION_MODE: CitationMode.FULL_SKELETON,
                Key.WORK_COUNT: 1, Key.CITES_COUNT: 0}
        base.update(citation)
        return {Key.CITATION: base}

    def test_the_owners_decision_passes(self):
        ok, detail = citation_content_checks.check_policy_is_the_owners(
            self._manifest(**{Key.CITATION_POLICY_SOURCE: PolicySource.OWNER}))
        self.assertTrue(ok, detail)

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

    def test_an_unknown_source_is_refused_too(self):
        ok, _detail = citation_content_checks.check_policy_is_the_owners(
            self._manifest(**{Key.CITATION_POLICY_SOURCE: "somebody-else"}))
        self.assertFalse(ok)

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
