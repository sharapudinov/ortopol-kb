"""Unit tests for deploy/dump_scan.py and deploy/profile_checks.py -- the
static half of artifact verification: no Docker, no Postgres, no network.

Every case builds a real gzipped dump (COPY blocks in Postgres' text format)
plus a real manifest.json in a temp directory, then asks the checks about it.
That is the same input a recipient has, so a check that passes here passes
for the same reason there.

The leak cases matter more than the happy path: a public artifact that ships
a blob for a publisher-licensed paper must FAIL, and it is this module that
has to notice.
"""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_policy_check
import dump_scan
import profile_checks
from manifest_classes import check_profile_is_known
from manifest_keys import Key
from manifest_contract import CitationMode, PolicySource, Profile
from _artifact_fixtures import (
    ArtifactBuilder,
    DOCUMENT_COLUMNS,
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


class DumpScanTests(unittest.TestCase):
    def test_counts_rows_and_nulls_per_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            scans = dump_scan.scan(directory / "01_dump.sql.gz")
        documents = scans["corpus.documents"]
        self.assertEqual(documents.rows, 3)
        self.assertEqual(documents.columns, DOCUMENT_COLUMNS)
        self.assertEqual(documents.nulls["source_blob"], 1)  # the metadata-only one
        self.assertEqual(scans["corpus.pages"].nulls["body"], 2)

    def test_schema_names_sees_ddl_and_copy_statements(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.extra_sql = "CREATE TABLE measurements.run (id integer);\n"
            directory = builder.write()
            self.assertEqual(
                dump_scan.schema_names(directory / "01_dump.sql.gz"), {"corpus", "measurements"},
            )

    def test_row_with_the_wrong_field_count_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.documents.append([FULL_DOC, "only-two-fields"])
            directory = builder.write()
            with self.assertRaises(ValueError):
                dump_scan.scan(directory / "01_dump.sql.gz")

    def test_truncated_copy_block_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "cut.sql.gz"
            with gzip.open(dump_path, "wt", encoding="utf-8") as f:
                f.write("COPY corpus.pages (document_id) FROM stdin;\n2009_isu34\n")
            with self.assertRaises(ValueError) as ctx:
                dump_scan.scan(dump_path)
        self.assertIn("truncated", str(ctx.exception))


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
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.shipped = None
            results = _results(builder)
        ok, detail = results["правовая классификация полна"]
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

    WORK_COLUMNS = ["id", "key", "title", "abstract", "evidence"]
    CITES_COLUMNS = ["citing", "cited", "evidence"]

    def _builder_with_citation(self, tmp, mode, work_rows, cites_rows):
        builder = ArtifactBuilder(Path(tmp))
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
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.citation_block = False
            results = _results(builder)
        ok, detail = results["citation: режим — решение владельца, не --policy-override"]
        self.assertFalse(ok, detail)

    def test_full_skeleton_with_matching_counts_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder_with_citation(
                tmp, "full-skeleton",
                work_rows=[["1", "k1", "T1", "an abstract", '{"src": "openalex"}']],
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
                work_rows=[["1", "k1", "T1", "an abstract", "\\N"]],
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
                work_rows=[["1", "k1", "T1", "\\N", "\\N"]],
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
                "work", self.WORK_COLUMNS, [["1", "k1", "T1", "\\N", "\\N"]])
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
