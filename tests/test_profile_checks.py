"""Unit tests for deploy/profile_checks.py -- the static half of artifact
verification: no Docker, no Postgres, no network. The dump READER those
checks run on has its own module next door (test_dump_scan.py).

Every case builds a real gzipped dump (COPY blocks in Postgres' text format)
plus a real manifest.json in a temp directory, then asks the checks about it.
That is the same input a recipient has, so a check that passes here passes
for the same reason there.

The leak cases matter more than the happy path: a public artifact that ships
a blob for a publisher-licensed paper must FAIL, and it is this module that
has to notice.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import corpus_content_checks
import citation_policy_check
import dump_scan
import profile_checks
from manifest_classes import check_profile_is_known
from manifest_keys import Key
from manifest_contract import CitationMode, PolicySource, Profile
from _artifact_fixtures import (
    ArtifactBuilder,
    EXCLUDED_DOC,
    FULL_DOC,
    INTERNAL_DOC,
    META_DOC,
    PAGE_COLUMNS,
    _citation_copy_block,
    _document_row,
    _page_row,
)

def _results(builder: ArtifactBuilder) -> dict[str, tuple[bool, str]]:
    directory = builder.write()
    return {name: (ok, detail) for name, ok, detail in profile_checks.run_checks(directory)}


class OnePassTests(unittest.TestCase):
    """run_checks() inflates the dump exactly once.

    The schema names and the COPY headers are the same lines, and the full
    profile's dump carries every source PDF as hex, so a second pass for
    the schema question doubled the cost of verifying the artifact the
    checker exists for. Counted at the only place the file is actually
    opened.
    """

    def test_the_dump_is_decompressed_once(self):
        opens = []
        real_open = dump_scan.gzip.open

        def counting_open(path, *args, **kwargs):
            opens.append(Path(path).name)
            return real_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            with mock.patch.object(dump_scan.gzip, "open", counting_open):
                results = profile_checks.run_checks(directory)
        self.assertTrue(all(ok for _name, ok, _detail in results), results)
        self.assertEqual(opens, ["01_dump.sql.gz"])


class PublicProfileChecksTests(unittest.TestCase):
    def test_a_well_formed_public_artifact_passes_every_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _results(ArtifactBuilder(Path(tmp)))
        for name, (ok, detail) in results.items():
            self.assertTrue(ok, f"{name}: {detail}")

    def test_a_leaked_blob_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents[1] = _document_row(META_DOC, "metadata-only", "\\x2550")
            results = _results(builder)
        ok, detail = results["metadata-only: ни блоба, ни текста"]
        self.assertFalse(ok)
        self.assertIn(META_DOC, detail)

    def test_leaked_page_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.pages[1] = _page_row(META_DOC, 1, "первая страница статьи")
            results = _results(builder)
        ok, detail = results["metadata-only: ни блоба, ни текста"]
        self.assertFalse(ok)
        self.assertIn(META_DOC, detail)

    def test_missing_content_for_a_full_text_document_fails(self):
        # The opposite error, and just as much a defect: the public artifact
        # must actually carry what it is allowed to carry.
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents[0] = _document_row(FULL_DOC, "full-text", "\\N")
            results = _results(builder)
        ok, detail = results["full-text: блоб и текст на месте"]
        self.assertFalse(ok)
        self.assertIn(FULL_DOC, detail)

    def test_page_without_an_embedding_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.pages[1] = _page_row(META_DOC, 1, "", embedding="\\N")
            results = _results(builder)
        ok, detail = results["векторы у всех страниц"]
        self.assertFalse(ok)
        self.assertIn("1 without an embedding", detail)

    def test_page_count_disagreeing_with_the_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.pages.append(_page_row(FULL_DOC, 2, "ещё страница"))
            directory = builder.write()
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["pages_count"] -= 1
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
            results = {n: (ok, d) for n, ok, d in profile_checks.run_checks(directory)}
        self.assertFalse(results["векторы у всех страниц"][0])

    def test_measurements_schema_in_a_public_dump_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.extra_sql = "CREATE SCHEMA measurements;\n"
            results = _results(builder)
        name = next(n for n in results if n.startswith("профиль"))
        ok, detail = results[name]
        self.assertFalse(ok)
        self.assertIn("measurements", detail)

    def test_a_manifest_may_not_declare_a_schema_the_rule_forbids(self):
        """The dump and the declaration agree with each other and with
        nothing else -- one build's decision wrote both.

        A public artifact carries schema corpus alone; here it carries
        measurements as well and says so honestly, which is exactly the
        shape a comparison of the dump against the manifest's own claim
        certifies. The rule the producer resolved by is re-derived on this
        side instead (manifest_contract.schemas_for).
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.schemas = ["corpus", "measurements"]
            builder.extra_sql = "CREATE SCHEMA measurements;\n"
            results = _results(builder)
        name = next(n for n in results if n.startswith("профиль"))
        ok, detail = results[name]
        self.assertFalse(ok, detail)
        self.assertIn("measurements", detail)

    def test_a_citation_mode_outside_the_vocabulary_is_a_row_not_a_raise(self):
        """The rule cannot be derived from a mode this reader does not
        know, and the pass still has to report that as a red line: a
        traceback out of run_checks() leaves a caller that extends its own
        list (smoke_test.py) with no results at all.
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.citation = {"mode": "half-skeleton", "work_count": 0, "cites_count": 0}
            results = _results(builder)
        name = next(n for n in results if n.startswith("профиль"))
        ok, detail = results[name]
        self.assertFalse(ok, detail)
        self.assertIn("half-skeleton", detail)

    def test_a_corpus_table_declared_but_never_shipped_fails(self):
        """The corpus half of the per-table declaration, end to end: a
        table the manifest promises and the dump does not carry used to be
        indistinguishable from a table correctly cut away, because only
        documents and pages were described at all.
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.corpus_table_rows = {"documents": 3, "pages": 4, "embedding_model": 1}
            results = _results(builder)
        ok, detail = results["corpus: каждая заявленная таблица приехала целиком"]
        self.assertFalse(ok)
        self.assertIn("corpus.embedding_model", detail)

    def test_tsv_in_the_page_copy_columns_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.page_columns = PAGE_COLUMNS + ["tsv"]
            builder.pages = [row + ["'текст':1"] for row in builder.pages]
            results = _results(builder)
        self.assertFalse(results["нет generated-колонок в дампе"][0])

    def test_unclassified_document_in_the_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.unclassified = 1
            results = _results(builder)
        self.assertFalse(results["правовая классификация полна"][0])

    def test_document_missing_from_every_class_list_fails(self):
        # The completeness predicate: an id in the dump that no class claims
        # (or a class list that claims one twice) must not pass silently.
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.by_distribution["metadata-only"] = []
            results = _results(builder)
        ok, detail = results["правовая классификация полна"]
        self.assertFalse(ok)
        self.assertIn("3 document row(s)", detail)

    def test_excluded_document_is_counted_out_not_missing(self):
        # The manifest lists four documents and the dump carries three: that
        # is the excluded one, and it must read as correct rather than as an
        # incomplete artifact.
        with tempfile.TemporaryDirectory() as tmp:
            results = _results(ArtifactBuilder(Path(tmp)))
        ok, detail = results["правовая классификация полна"]
        self.assertTrue(ok, detail)
        self.assertIn("4 id(s)", detail)
        self.assertIn("3 of them shipped", detail)

    def test_a_leaked_row_for_an_excluded_document_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents.append(_document_row(EXCLUDED_DOC, "excluded", "\\N"))
            results = _results(builder)
        ok, detail = results["excluded: ни строки документа, ни страниц"]
        self.assertFalse(ok)
        self.assertIn(EXCLUDED_DOC, detail)

    def test_a_leaked_page_for_an_excluded_document_fails(self):
        # Even a body-less page row is a leak here: it carries the vector,
        # which is what makes the work findable in the first place.
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.pages.append(_page_row(EXCLUDED_DOC, 1, ""))
            results = _results(builder)
        ok, detail = results["excluded: ни строки документа, ни страниц"]
        self.assertFalse(ok)
        self.assertIn(EXCLUDED_DOC, detail)

    def test_a_manifest_without_shipped_distributions_is_refused(self):
        # An artifact that cannot say which classes it carries cannot be
        # verified at all -- and must not be read as carrying everything.
        # The legal-vocabulary gate stops the pass on it before a check
        # about the dump runs (tests/test_manifest_classes.py); asked of the
        # check that reads the block, the refusal is the same one.
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.shipped = None
            results = _results(builder)
            manifest = json.loads((builder.directory / "manifest.json").read_text())
        ok, detail = results["правовой словарь манифеста известен"]
        self.assertFalse(ok)
        self.assertIn("shipped_distributions", detail)
        ok, detail = corpus_content_checks.check_classification_complete(manifest, {})
        self.assertFalse(ok)
        self.assertIn("shipped_distributions", detail)


class ProfileVocabularyGateTests(unittest.TestCase):
    """The profile string is the switch every strictness in the pass reads.

    `!= Profile.PUBLIC` is how expected_ids(), content_expectation() and
    the citation policy check each pick what to demand, so one unvalidated
    field turned the whole certification lenient at once: a manifest whose
    profile is missing, misspelt or hand-edited printed a column of passes
    about a package nothing had been verified against.
    """

    def _results_with_profile(self, profile):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.profile = profile
            return _results(builder)

    def test_an_unknown_profile_stops_the_pass_instead_of_relaxing_it(self):
        for profile in ("staging", "Public", "", None):
            with self.subTest(profile=profile):
                results = self._results_with_profile(profile)
                ok, detail = results["манифест называет известный профиль"]
                self.assertFalse(ok)
                self.assertIn(repr(profile), detail)
                # Nothing below the gate ran: a pass printed under it reads
                # as a verified package.
                self.assertEqual(len(results), 2)

    def test_the_policy_check_refuses_an_unknown_profile_on_its_own(self):
        """It is called from run_checks() above the gate today; the refusal
        belongs to the check as well, because it is the one that chooses
        between demanding the owner's row and excusing the profile.
        """
        manifest = {Key.PROFILE: "staging",
                    Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                                   Key.CITATION_POLICY_SOURCE: PolicySource.NOT_APPLICABLE}}
        ok, detail = citation_policy_check.check_policy_is_the_owners(manifest)
        self.assertFalse(ok, detail)
        self.assertIn("staging", detail)

    def test_a_declared_profile_still_passes_the_gate(self):
        for profile in Profile.ALL:
            with self.subTest(profile=profile):
                ok, _detail = check_profile_is_known({Key.PROFILE: profile})
                self.assertTrue(ok)


class CitationContentChecksIntegrationTests(unittest.TestCase):
    """The two citation checks (citation_content_checks.py) as run through
    profile_checks.run_checks() end to end -- unit coverage of the
    predicates themselves lives in test_citation_content_checks.py.
    """

    WORK_COLUMNS = ["id", "key", "kind", "title", "abstract", "evidence"]
    CITES_COLUMNS = ["citing", "cited", "evidence"]

    def _builder_with_citation(self, tmp, mode, work_rows, cites_rows):
        builder = ArtifactBuilder(Path(tmp))
        # Every mode this helper is called with ships the schema, so the
        # artifact declares it: manifest.schemas is held to the rule now,
        # not merely to the dump.
        builder.schemas = ["corpus", "citation"]
        builder.citation = {
            "mode": mode, "work_count": len(work_rows), "cites_count": len(cites_rows),
            "work_columns": self.WORK_COLUMNS, "cites_columns": self.CITES_COLUMNS,
            "work": work_rows, "cites": cites_rows,
        }
        return builder

    def test_a_build_that_ships_no_graph_still_certifies_the_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _results(ArtifactBuilder(Path(tmp)))
        self.assertTrue(results["citation: схема/счётчики совпадают с манифестом"][0])
        self.assertTrue(results["citation: content-колонки вырезаны вне full-skeleton"][0])
        self.assertTrue(results["citation: режим — решение владельца, не --policy-override"][0])

    def test_a_manifest_with_no_citation_block_is_refused(self):
        """The hole the defensive reads used to swallow: no block, no mode,
        nothing shipped, "nothing to check" -- and a clean certification
        with no statement about who decided the citation policy.

        Refused by the shape gate now, which runs in front of the dump pass
        (a `citation` field that is not a mapping would otherwise raise
        through _visit before any result exists) and carries the same
        verdict one row earlier.
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.citation_block = False
            results = _results(builder)
        ok, detail = results["манифест несёт блок citation словарём"]
        self.assertFalse(ok, detail)

    def test_full_skeleton_with_matching_counts_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder_with_citation(
                tmp, "full-skeleton",
                work_rows=[["1", "k1", "indexed", "T1", "an abstract", '{"src": "openalex"}']],
                cites_rows=[],
            )
            results = _results(builder)
        ok, detail = results["citation: схема/счётчики совпадают с манифестом"]
        self.assertTrue(ok, detail)
        ok, detail = results["citation: content-колонки вырезаны вне full-skeleton"]
        self.assertTrue(ok, detail)  # not topology-only -- nothing to strip

    def test_topology_only_with_a_leaked_abstract_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder_with_citation(
                tmp, "topology-only",
                work_rows=[["1", "k1", "indexed", "T1", "an abstract", "\\N"]],
                cites_rows=[],
            )
            results = _results(builder)
        ok, detail = results["citation: content-колонки вырезаны вне full-skeleton"]
        self.assertFalse(ok)
        self.assertIn("abstract", detail)

    def test_topology_only_properly_stripped_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder_with_citation(
                tmp, "topology-only",
                work_rows=[["1", "k1", "indexed", "T1", "\\N", "\\N"]],
                cites_rows=[["1", "2", "\\N"]],
            )
            results = _results(builder)
        ok, detail = results["citation: схема/счётчики совпадают с манифестом"]
        self.assertTrue(ok, detail)
        ok, detail = results["citation: content-колонки вырезаны вне full-skeleton"]
        self.assertTrue(ok, detail)

    def test_none_mode_with_a_leaked_table_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.citation = {"mode": "none", "work_count": 0, "cites_count": 0}
            # Leak a citation.work COPY block directly, bypassing write()'s
            # own citation-writing path (which "none" never triggers) --
            # simulates a packager bug that shipped the table anyway.
            builder.extra_sql = _citation_copy_block(
                "work", self.WORK_COLUMNS, [["1", "k1", "indexed", "T1", "\\N", "\\N"]])
            results = _results(builder)
        ok, detail = results["citation: схема/счётчики совпадают с манифестом"]
        self.assertFalse(ok)
        self.assertIn("citation.work", detail)


class FullProfileChecksTests(unittest.TestCase):
    def _full_builder(self, directory: Path) -> ArtifactBuilder:
        builder = ArtifactBuilder(directory)
        builder.profile = "full"
        builder.schemas = ["corpus", "measurements"]
        builder.extra_sql = "CREATE SCHEMA measurements;\n"
        # The full artifact carries every blob and every body, whatever the
        # class -- including the documents the public profile omits: it is
        # the owner's own backup, and the legal cut is the public profile's
        # job alone.
        builder.documents = [
            _document_row(FULL_DOC, "full-text", "\\x2550"),
            _document_row(META_DOC, "metadata-only", "\\x2552"),
            _document_row(INTERNAL_DOC, "internal", "\\x2551"),
            _document_row(EXCLUDED_DOC, "excluded", "\\x2553"),
        ]
        builder.pages = [
            _page_row(FULL_DOC, 1, "текст"),
            _page_row(META_DOC, 1, "полный текст статьи"),
            _page_row(META_DOC, 2, "продолжение"),
            _page_row(INTERNAL_DOC, 1, "наш индекс"),
            _page_row(EXCLUDED_DOC, 1, "текст статьи ВМЖ"),
        ]
        return builder

    def test_full_artifact_with_everything_present_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _results(self._full_builder(Path(tmp)))
        for name, (ok, detail) in results.items():
            self.assertTrue(ok, f"{name}: {detail}")

    def test_full_artifact_keeps_the_documents_the_public_one_omits(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _results(self._full_builder(Path(tmp)))
        ok, detail = results["excluded: ни строки документа, ни страниц"]
        self.assertTrue(ok, detail)
        self.assertIn("0 excluded document(s)", detail)

    def test_full_artifact_missing_a_blob_fails_regardless_of_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._full_builder(Path(tmp))
            builder.documents[1] = _document_row(META_DOC, "metadata-only", "\\N")
            results = _results(builder)
        ok, detail = results["full-text: блоб и текст на месте"]
        self.assertFalse(ok)
        self.assertIn(META_DOC, detail)

    def test_full_artifact_must_declare_the_measurements_schema_it_carries(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._full_builder(Path(tmp))
            builder.schemas = ["corpus"]  # lying manifest
            results = _results(builder)
        name = next(n for n in results if n.startswith("профиль"))
        self.assertFalse(results[name][0])


class CliTests(unittest.TestCase):
    def test_exit_0_on_a_clean_artifact_and_1_on_a_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            self.assertEqual(profile_checks.main(["--artifact-dir", str(directory)]), 0)

        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents[1] = _document_row(META_DOC, "metadata-only", "\\x2550")
            directory = builder.write()
            self.assertEqual(profile_checks.main(["--artifact-dir", str(directory)]), 1)

    def test_missing_manifest_is_exit_2_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            from unittest import mock
            with mock.patch("sys.stderr"):
                self.assertEqual(profile_checks.main(["--artifact-dir", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
