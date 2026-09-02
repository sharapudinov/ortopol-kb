"""Unit tests for deploy/smoke_checks.py's manifest-comparison
predicates (check_counts, check_fulltext, check_measurements_run,
check_embedding_model_dims, check_embedding_model_digest): no Docker, no
live database. Postgres/HTTP interaction is stubbed by monkeypatching the
module-level `scalar`/`scalar_row` names the check functions actually call
(both re-exported here from pg_common.py -- see that module for the real
psql round trip). check_bundled_files has its own dedicated
test_smoke_checks_bundled_files.py (module size); check_vector has its own
test_vector_probe_check.py (module size, mirrors the vector_probe_check.py
split).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import smoke_checks


class ManifestComparisonTests(unittest.TestCase):
    def test_check_counts_match(self):
        # One round trip now (see smoke_checks._COUNTS_SQL) -- both counts
        # come back as a single scalar_row, not two separate run_sql calls.
        with mock.patch.object(smoke_checks, "scalar_row", return_value=["3", "10"]):
            ok, detail = smoke_checks.check_counts({}, {"documents_count": 3, "pages_count": 10})
        self.assertTrue(ok)

    def test_check_counts_mismatch(self):
        with mock.patch.object(smoke_checks, "scalar_row", return_value=["3", "9"]):
            ok, detail = smoke_checks.check_counts({}, {"documents_count": 3, "pages_count": 10})
        self.assertFalse(ok)
        self.assertIn("pages=9", detail)

    def test_check_fulltext_requires_exact_match_not_just_at_least_one(self):
        manifest = {"fulltext_probe": {"query": "повторные средние", "hits": 5}}
        with mock.patch.object(smoke_checks, "scalar", return_value="1"):
            ok, _ = smoke_checks.check_fulltext({}, manifest)
        # A partial restore that still matches once must FAIL, not pass on ">=1".
        self.assertFalse(ok)

    def test_check_fulltext_exact_match_passes(self):
        manifest = {"fulltext_probe": {"query": "повторные средние", "hits": 5}}
        with mock.patch.object(smoke_checks, "scalar", return_value="5"):
            ok, _ = smoke_checks.check_fulltext({}, manifest)
        self.assertTrue(ok)

    def test_check_measurements_run_complete(self):
        # One round trip now (see smoke_checks._MEASUREMENTS_RUN_SQL).
        with mock.patch.object(smoke_checks, "scalar_row", return_value=["42", "0"]):
            ok, _ = smoke_checks.check_measurements_run({}, {"measurements_run_count": 42})
        self.assertTrue(ok)

    def test_check_measurements_run_missing_required_fields(self):
        with mock.patch.object(smoke_checks, "scalar_row", return_value=["42", "2"]):
            ok, detail = smoke_checks.check_measurements_run({}, {"measurements_run_count": 42})
        self.assertFalse(ok)
        self.assertIn("missing required fields=2", detail)


class CitationProjectionTests(unittest.TestCase):
    """check_citation_projection: SKIP for an artifact that ships no
    citation mode (older artifact, or CitationMode.NONE), otherwise the same
    |V|=work/|E|=cites comparison `pg_graph.py project --check` makes.
    """

    def test_skips_when_manifest_has_no_citation_block(self):
        ok, detail = smoke_checks.check_citation_projection({}, {})
        self.assertIsNone(ok)
        self.assertIn("None", detail)

    def test_skips_under_none_mode(self):
        manifest = {"citation": {"mode": "none", "work_count": 0, "cites_count": 0}}
        with mock.patch.object(smoke_checks.pg_graph_common, "graph_exists") as exists_mock:
            ok, _detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertIsNone(ok)
        exists_mock.assert_not_called()

    def test_fails_when_graph_was_never_projected(self):
        manifest = {"citation": {"mode": "full-skeleton", "work_count": 5, "cites_count": 3}}
        with mock.patch.object(smoke_checks.pg_graph_common, "graph_exists", return_value=False):
            ok, detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertFalse(ok)
        self.assertIn("02_project_graph.sql", detail)

    def test_matching_counts_pass(self):
        manifest = {"citation": {"mode": "topology-only", "work_count": 438, "cites_count": 2425}}
        with mock.patch.object(smoke_checks.pg_graph_common, "graph_exists", return_value=True), \
             mock.patch.object(smoke_checks.pg_graph_common, "graph_counts", return_value=(438, 2425)):
            ok, detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertTrue(ok, detail)

    def test_mismatched_counts_fail(self):
        manifest = {"citation": {"mode": "full-skeleton", "work_count": 438, "cites_count": 2425}}
        with mock.patch.object(smoke_checks.pg_graph_common, "graph_exists", return_value=True), \
             mock.patch.object(smoke_checks.pg_graph_common, "graph_counts", return_value=(437, 2425)):
            ok, detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertFalse(ok)
        self.assertIn("diff -1", detail)


class CheckEmbeddingModelDimsTests(unittest.TestCase):
    # check_embedding_model_dims folded its two round trips (declared
    # model/dims, then the corpus.pages aggregate) into one -- see
    # smoke_checks._EMBEDDING_MODEL_DIMS_SQL -- so every case below mocks a
    # single 5-column scalar_row return instead of two side_effect calls.

    def test_check_embedding_model_dims_consistent(self):
        with mock.patch.object(
            smoke_checks, "scalar_row",
            return_value=["bge-m3", "1024", "1", "1024", "500"],
        ):
            ok, detail = smoke_checks.check_embedding_model_dims({})
        self.assertTrue(ok)
        self.assertIn("1 distinct vector_dims", detail)

    def test_check_embedding_model_dims_mixed_widths_fails(self):
        # This is the exact failure mode the aggregate check exists to catch:
        # a partially re-embedded corpus, where a single sampled row could
        # accidentally happen to have the right width.
        with mock.patch.object(
            smoke_checks, "scalar_row",
            return_value=["bge-m3", "1024", "2", "1024", "500"],
        ):
            ok, _ = smoke_checks.check_embedding_model_dims({})
        self.assertFalse(ok)

    def test_check_embedding_model_dims_no_embedded_rows_fails(self):
        with mock.patch.object(
            smoke_checks, "scalar_row",
            return_value=["bge-m3", "1024", "0", "", "0"],
        ):
            ok, _ = smoke_checks.check_embedding_model_dims({})
        self.assertFalse(ok)

    def test_check_embedding_model_dims_empty_table(self):
        # corpus.embedding_model has no id = 1 row: both scalar subselects
        # for model/dims come back NULL (empty string via psql -t -A), but
        # the outer query -- five independent scalar subselects, no shared
        # FROM clause -- still returns exactly one row, same as
        # manifest_probe._MANIFEST_SCALARS_SQL.
        with mock.patch.object(
            smoke_checks, "scalar_row",
            return_value=["", "", "0", "", "0"],
        ):
            ok, detail = smoke_checks.check_embedding_model_dims({})
        self.assertFalse(ok)
        self.assertIn("empty", detail)

    def test_check_embedding_model_dims_sql_filters_by_id_1(self):
        # a0d70129: must match pg_search.embed_query's WHERE id = 1, not
        # read the table unfiltered.
        self.assertIn("WHERE id = 1", smoke_checks._EMBEDDING_MODEL_DIMS_SQL)


class CheckEmbeddingModelDigestTests(unittest.TestCase):
    """check_embedding_model_digest() now delegates the actual HTTP/tags
    lookup to ollama_registry.served_model_digest (see test_ollama_registry.py
    for that function's own transport/matching tests) -- this class only
    covers check_embedding_model_digest's own comparison logic, so the
    served_model_digest boundary is stubbed directly rather than re-testing
    urllib through it.
    """
    _MANIFEST = {"embedding_model": {"model": "bge-m3", "dims": 1024, "digest": "sha256:abc"}}

    def test_matching_digest_passes(self):
        with mock.patch.object(smoke_checks, "served_model_digest", return_value=("sha256:abc", 123)):
            ok, detail = smoke_checks.check_embedding_model_digest(self._MANIFEST, "http://x/api/embed")
        self.assertTrue(ok)

    def test_different_digest_fails_hard(self):
        # The whole point of this check: same name+dims, different build.
        with mock.patch.object(smoke_checks, "served_model_digest",
                                return_value=("sha256:def-a-different-build", 123)):
            ok, detail = smoke_checks.check_embedding_model_digest(self._MANIFEST, "http://x/api/embed")
        self.assertFalse(ok)

    def test_no_manifest_digest_fails(self):
        ok, detail = smoke_checks.check_embedding_model_digest(
            {"embedding_model": {"model": "bge-m3", "dims": 1024}}, "http://x/api/embed",
        )
        self.assertFalse(ok)
        self.assertIn("no embedding_model.digest", detail)

    def test_transport_failure_fails_cleanly(self):
        with mock.patch.object(smoke_checks, "served_model_digest",
                                side_effect=RuntimeError("could not query http://x/api/tags: down")):
            ok, detail = smoke_checks.check_embedding_model_digest(self._MANIFEST, "http://x/api/embed")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
