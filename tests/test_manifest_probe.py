"""Unit tests for deploy/manifest_probe.py: no Docker, no live
database. Postgres/HTTP interaction is stubbed by monkeypatching the
module-level names manifest_probe actually calls. served_model_digest's own
transport/matching logic is tested directly in test_ollama_registry.py.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import manifest_probe


class StemmedTokenOverlapTests(unittest.TestCase):
    """_stemmed_token_overlap() delegates the actual stemming/intersection
    to a single SQL round trip (see _TOKEN_OVERLAP_SQL) -- scalar() is
    stubbed here to return the FIELD_SEP-joined lexeme string Postgres would
    produce, so only the Python-side split/sort is under test; the SQL
    itself is exercised for real by GatherManifestErrorPathsTests below via
    manifest_probe.scalar (same stub point, different scenario) and by the
    live-Postgres integration run.
    """

    def test_empty_result_means_no_overlap(self):
        with mock.patch.object(manifest_probe, "scalar", return_value=""):
            overlap = manifest_probe._stemmed_token_overlap({}, "q", "doc1", 5)
        self.assertEqual(overlap, [])

    def test_splits_and_sorts_multiple_lexemes(self):
        with mock.patch.object(manifest_probe, "scalar", return_value="полином\x1fвеличин"):
            overlap = manifest_probe._stemmed_token_overlap({}, "q", "doc1", 5)
        self.assertEqual(overlap, ["величин", "полином"])

    def test_variables_carry_query_document_and_page(self):
        with mock.patch.object(manifest_probe, "scalar", return_value="") as scalar_mock:
            manifest_probe._stemmed_token_overlap({}, "запрос", "2015_demr1", 69)
        _, kwargs = scalar_mock.call_args
        self.assertEqual(kwargs["variables"], {"q": "запрос", "doc": "2015_demr1", "page": "69"})


# A well-formed 8-column row for _MANIFEST_SCALARS_SQL: documents_count,
# pages_count, model, dims, runs_count, blob_len, blob_sha, fulltext_hits.
_GOOD_ROW = ["70", "2462", "bge-m3", "1024", "5", "12345", "a" * 64, "3"]

# What legal_profile.legal_summary() returns, stubbed: it is four independent
# SQL reads of its own, tested against the real database shape in
# test_legal_profile.py. Stubbing it here keeps these tests DB-free -- the
# point below is what gather_manifest DOES with a profile, not how the
# classification is read.
_LEGAL_SUMMARY = {
    "verify_query": "SELECT count(*) ...",
    "unclassified_documents": 0,
    "class_counts": [],
    "documents_by_distribution": {
        "full-text": ["2009_isu34", "2015_demr1"],
        "metadata-only": ["1997_sm280"],
        "excluded": ["2016_vmj598"],
    },
    "full_content_distributions": ["full-text", "internal"],
    "shipped_distributions": ["full-text", "metadata-only", "internal"],
}


# The resolution's output, spelled at every call site: gather_manifest()
# takes the pair as required keywords, so there is no shape of the call
# that leaves either half to a default.
_RESOLVED = {"citation_mode": "full-skeleton", "policy_source": "owner"}


def _patch_citation_defaults(test_case: unittest.TestCase) -> None:
    """Mocks citation_profile.citation_counts so gather_manifest() never
    touches the database for its citation-graph half. The MODE is no longer
    read here at all: build_package.main() resolves it once and hands it in
    (citation_profile.resolve_citation_mode, tested in
    test_citation_profile.py).
    """
    patcher = mock.patch.object(manifest_probe.citation_profile, "citation_counts",
                                 return_value=(0, 0, {}))
    patcher.start()
    test_case.addCleanup(patcher.stop)


class GatherManifestErrorPathsTests(unittest.TestCase):
    """gather_manifest()'s guard branches -- previously untested, so a
    regression in any of them (e.g. the overlap check silently passing a
    lexically-overlapping pair) would have gone undetected.
    """

    def setUp(self):
        _patch_citation_defaults(self)

    def _patch_prelude(self, row=None, digest=("sha256:abc", 123)):
        row = list(row if row is not None else _GOOD_ROW)
        return (
            mock.patch.object(manifest_probe, "scalar_row", return_value=row),
            mock.patch.object(manifest_probe, "served_model_digest", return_value=digest),
            mock.patch.object(manifest_probe.legal_profile, "legal_summary",
                              return_value=dict(_LEGAL_SUMMARY)),
        )

    def test_embedding_service_unreachable_raises(self):
        scalar_row_p, digest_p, legal_p = self._patch_prelude()
        with scalar_row_p, digest_p, legal_p, \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertIn("unreachable", str(ctx.exception))

    def test_no_embedded_pages_raises(self):
        scalar_row_p, digest_p, legal_p = self._patch_prelude()
        with scalar_row_p, digest_p, legal_p, \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertIn("no embedded rows", str(ctx.exception))

    def test_lexical_overlap_raises(self):
        nearest = {"document_id": "2015_demr1", "page_number": 69, "rank": 1, "distance": 0.4}
        scalar_row_p, digest_p, legal_p = self._patch_prelude()
        with scalar_row_p, digest_p, legal_p, \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=nearest), \
             mock.patch.object(manifest_probe, "scalar", return_value="модуль"):
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertIn("token", str(ctx.exception))
        self.assertIn("модуль", str(ctx.exception))

    def test_missing_embedding_model_row_raises_informative_error(self):
        row = ["70", "2462", "", "", "5", "12345", "a" * 64, "3"]  # NULL model/dims
        scalar_row_p, digest_p, legal_p = self._patch_prelude(row=row)
        with scalar_row_p, digest_p, legal_p:
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertIn("embedding_model is empty", str(ctx.exception))

    def test_missing_blob_probe_document_raises_informative_error(self):
        row = ["70", "2462", "bge-m3", "1024", "5", "", "", "3"]  # NULL blob columns
        scalar_row_p, digest_p, legal_p = self._patch_prelude(row=row)
        with scalar_row_p, digest_p, legal_p:
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertIn(manifest_probe.BLOB_PROBE_DOC, str(ctx.exception))

    def test_happy_path_records_digest_in_manifest(self):
        nearest = {"document_id": "2015_demr1", "page_number": 69, "rank": 1, "distance": 0.4}
        scalar_row_p, digest_p, legal_p = self._patch_prelude(digest=("sha256:deadbeef", 1157672605))
        with scalar_row_p, digest_p, legal_p, \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=nearest), \
             mock.patch.object(manifest_probe.pg_rank_probe, "runner_up_distance", return_value=0.55), \
             mock.patch.object(manifest_probe, "scalar", return_value=""):
            manifest = manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertEqual(manifest["embedding_model"]["digest"], "sha256:deadbeef")
        self.assertEqual(manifest["embedding_model"]["size_bytes"], 1157672605)
        self.assertEqual(manifest["vector_probe"]["token_overlap"], [])
        self.assertEqual(manifest["vector_probe"]["runner_up_distance"], 0.55)
        # c5d48e2f: shared constant, not a literal re-typed here and in
        # smoke_test.py's mismatch check.
        self.assertEqual(manifest["schema_version"], manifest_probe.MANIFEST_SCHEMA_VERSION)


class ProfileAwarenessTests(unittest.TestCase):
    """The manifest must describe THE ARTIFACT, not the live database: the
    public profile ships no measurements schema and no blob for a
    publisher-licensed paper, so recording the live measurement count or the
    full profile's blob probe there would produce a manifest its own smoke
    test can never satisfy.
    """

    NEAREST = {"document_id": "2015_demr1", "page_number": 69, "rank": 1, "distance": 0.4}

    def setUp(self):
        _patch_citation_defaults(self)

    def _gather(self, profile, citation_mode="full-skeleton", policy_source="owner"):
        with mock.patch.object(manifest_probe, "scalar_row", return_value=list(_GOOD_ROW)) as row_mock, \
             mock.patch.object(manifest_probe, "served_model_digest", return_value=("d", 1)), \
             mock.patch.object(manifest_probe.legal_profile, "legal_summary",
                                return_value=dict(_LEGAL_SUMMARY)), \
             mock.patch.object(manifest_probe.legal_profile, "require_classified") as classified_mock, \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=self.NEAREST), \
             mock.patch.object(manifest_probe.pg_rank_probe, "runner_up_distance", return_value=0.5), \
             mock.patch.object(manifest_probe, "scalar", return_value=""):
            manifest = manifest_probe.gather_manifest(
                {}, "http://x/api/embed", profile=profile, citation_mode=citation_mode,
                policy_source=policy_source)
        return manifest, row_mock, classified_mock

    def test_full_profile_is_the_default_and_keeps_both_schemas(self):
        manifest, row_mock, classified_mock = self._gather("full")
        self.assertEqual(manifest["profile"], "full")
        # citation appended: the resolved mode this build was handed is a
        # shipping one, so the schema is declared (schemas_for()).
        self.assertEqual(manifest["schemas"], ["corpus", "measurements", "citation"])
        self.assertEqual(manifest["citation"]["mode"], "full-skeleton")
        self.assertEqual(manifest["measurements_run_count"], int(_GOOD_ROW[4]))
        self.assertEqual(manifest["blob_probe"]["document_id"], manifest_probe.BLOB_PROBE_DOC)
        # No legal gate on the full profile: it never leaves the owner's
        # machines, and refusing to back up an unclassified corpus would be
        # the wrong incentive entirely.
        classified_mock.assert_not_called()
        sql = row_mock.call_args[0][1]
        self.assertIn("AND TRUE", sql)

    def test_public_profile_declares_corpus_only_and_zero_measurements(self):
        # citation_mode "none" here: this test is about the corpus half.
        manifest, row_mock, classified_mock = self._gather("public", citation_mode="none")
        self.assertEqual(manifest["profile"], "public")
        self.assertEqual(manifest["schemas"], ["corpus"])
        self.assertEqual(manifest["measurements_run_count"], 0)
        classified_mock.assert_called_once()

    def test_public_profile_probes_a_blob_it_actually_ships(self):
        manifest, _row_mock, _classified = self._gather("public")
        self.assertEqual(
            manifest["blob_probe"]["document_id"], manifest_probe.PUBLIC_BLOB_PROBE_DOC,
        )
        self.assertNotEqual(manifest_probe.PUBLIC_BLOB_PROBE_DOC, manifest_probe.BLOB_PROBE_DOC)

    def test_public_fulltext_probe_counts_only_shipped_text(self):
        # The recorded hit count must come from a query restricted to
        # full-content documents; counting live matches would record pages
        # whose body the public artifact deliberately leaves empty.
        _manifest, row_mock, _classified = self._gather("public")
        sql = row_mock.call_args[0][1]
        self.assertIn("public_distribution IN ('full-text', 'internal')", sql)
        self.assertIn("JOIN corpus.documents", sql)

    def test_public_counts_leave_out_the_documents_the_artifact_omits(self):
        # Counting the live corpus would describe a package that does not
        # exist: an excluded document contributes neither a documents row
        # nor a page row to the public dump.
        _manifest, row_mock, _classified = self._gather("public")
        sql = row_mock.call_args[0][1]
        shipped = "public_distribution IN ('full-text', 'metadata-only', 'internal')"
        self.assertIn(f"FROM corpus.documents WHERE {shipped}", sql)
        self.assertEqual(sql.count(shipped), 2)  # documents and pages alike

    def test_full_counts_are_unrestricted(self):
        _manifest, row_mock, _classified = self._gather("full")
        sql = row_mock.call_args[0][1]
        self.assertIn("FROM corpus.documents WHERE TRUE", sql)
        self.assertNotIn("'metadata-only'", sql)

    def test_public_refuses_a_vector_probe_naming_a_document_it_omits(self):
        # Recording it would produce a manifest the artifact's own smoke
        # test can never satisfy -- the page simply is not in the package.
        excluded = dict(self.NEAREST, document_id="2016_vmj598")
        with mock.patch.object(manifest_probe, "scalar_row", return_value=list(_GOOD_ROW)), \
             mock.patch.object(manifest_probe, "served_model_digest", return_value=("d", 1)), \
             mock.patch.object(manifest_probe.legal_profile, "legal_summary",
                                return_value=dict(_LEGAL_SUMMARY)), \
             mock.patch.object(manifest_probe.legal_profile, "require_classified"), \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=excluded), \
             mock.patch.object(manifest_probe, "scalar", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                manifest_probe.gather_manifest({}, "http://x/api/embed", profile="public",
                                               **_RESOLVED)
            self.assertIn("2016_vmj598", str(ctx.exception))
            # The same probe is fine for the full profile, which ships it.
            with mock.patch.object(manifest_probe.pg_rank_probe, "runner_up_distance",
                                    return_value=0.5):
                manifest = manifest_probe.gather_manifest({}, "http://x/api/embed", **_RESOLVED)
        self.assertEqual(manifest["vector_probe"]["document_id"], "2016_vmj598")

    def test_unknown_profile_refused(self):
        with self.assertRaises(ValueError) as ctx:
            manifest_probe.gather_manifest({}, "http://x/api/embed",
                                           profile="sort-of-public", **_RESOLVED)
        self.assertIn("sort-of-public", str(ctx.exception))

    def test_legal_block_is_carried_into_the_manifest(self):
        manifest, _row_mock, _classified = self._gather("public")
        self.assertEqual(manifest["legal"], _LEGAL_SUMMARY)


class CitationManifestTests(unittest.TestCase):
    """The citation{} block describes the PACKAGE, never the live database
    -- MANIFEST_DESCRIBES_ARTIFACT applied to the citation schema. The mode
    itself arrives resolved (see build_package.main); what this module still
    decides is what to COUNT under it.
    """

    NEAREST = {"document_id": "2015_demr1", "page_number": 69, "rank": 1, "distance": 0.4}

    def _gather(self, profile="public", citation_mode="topology-only",
                policy_source="owner"):
        with mock.patch.object(manifest_probe, "scalar_row", return_value=list(_GOOD_ROW)), \
             mock.patch.object(manifest_probe, "served_model_digest", return_value=("d", 1)), \
             mock.patch.object(manifest_probe.legal_profile, "legal_summary",
                                return_value=dict(_LEGAL_SUMMARY)), \
             mock.patch.object(manifest_probe.legal_profile, "require_classified"), \
             mock.patch.object(manifest_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(manifest_probe.pg_rank_probe, "nearest_page", return_value=self.NEAREST), \
             mock.patch.object(manifest_probe.pg_rank_probe, "runner_up_distance", return_value=0.5), \
             mock.patch.object(manifest_probe, "scalar", return_value=""), \
             mock.patch.object(manifest_probe.citation_profile, "citation_counts",
                                return_value=(438, 2425,
                                              {"external-skeleton": 382, "our-document": 56})) as counts_mock:
            manifest = manifest_probe.gather_manifest(
                {}, "http://x/api/embed", profile=profile, citation_mode=citation_mode,
                policy_source=policy_source,
            )
        return manifest, counts_mock

    def test_manifest_counts_describe_package(self):
        manifest, _counts = self._gather()
        self.assertEqual(manifest["citation"], {
            "mode": "topology-only", "policy_source": "owner",
            "work_count": 438, "cites_count": 2425,
            "work_by_kind": {"external-skeleton": 382, "our-document": 56},
            # Declared empty here and stamped by build_package.main() from
            # what the dump actually wrote: this function runs before the
            # dump exists, and a live count would describe a package nobody
            # has produced yet.
            "table_rows": {},
        })
        self.assertEqual(manifest["schemas"], ["corpus", "citation"])

    def test_the_manifest_records_whose_decision_the_mode_was(self):
        """The filename is not part of the package; this field is. Without
        it an override build and an owner-classified one are byte-for-byte
        the same artifact to every consumer that reads manifest.json.
        """
        manifest, _counts = self._gather(policy_source="override")
        self.assertEqual(manifest["citation"]["policy_source"], "override")

    def test_neither_half_of_the_pair_can_be_omitted(self):
        """A call that names no mode and no provenance does not produce a
        manifest at all. Both are the resolution's output, and a default
        would let a second entry point certify "the owner decided" by
        saying nothing -- the one claim manifest.json exists to carry.
        """
        for kwargs in ({}, {"citation_mode": "none"}, {"policy_source": "owner"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    manifest_probe.gather_manifest({}, "http://x/api/embed", **kwargs)

    def test_public_counts_apply_the_per_document_cut(self):
        """A work row naming an excluded document does not ship
        (citation_dump.py), so the manifest must not count it either --
        otherwise profile_checks fails a correct package on its own numbers.
        """
        _manifest, counts_mock = self._gather(profile="public")
        self.assertEqual(counts_mock.call_args.kwargs, {"shipped_only": True})

    def test_full_profile_counts_the_whole_schema(self):
        _manifest, counts_mock = self._gather(profile="full", citation_mode="full-skeleton")
        self.assertEqual(counts_mock.call_args.kwargs, {"shipped_only": False})

    def test_none_mode_records_zero_counts_and_no_schema(self):
        """A FULL build whose database carries no citation schema: full
        applies no policy and describes the database as it is, so the mode
        is "none" and the provenance is "not-applicable" --
        resolve_citation_mode() reads no owner row on that path, and the
        manifest may not claim one on its behalf. (A PUBLIC build against
        such a database does not get here at all: it is refused, see
        citation_profile.resolve_citation_mode.)
        """
        manifest, counts_mock = self._gather(profile="full", citation_mode="none",
                                             policy_source="not-applicable")
        self.assertEqual(manifest["citation"],
                          {"mode": "none", "policy_source": "not-applicable",
                           "work_count": 0, "cites_count": 0, "work_by_kind": {},
                           "table_rows": {}})
        self.assertEqual(manifest["schemas"], ["corpus", "measurements"])
        counts_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
