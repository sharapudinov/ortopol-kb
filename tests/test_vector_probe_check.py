"""Unit tests for deploy/vector_probe_check.py's check_vector():
no Docker, no live database. Postgres/HTTP interaction is stubbed by
monkeypatching the module-level `pg_search`/`pg_rank_probe` names check_vector
actually calls. Split out of test_smoke_checks.py (module size) alongside the
vector_probe_check.py/smoke_checks.py split it mirrors.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import vector_probe_check


class CheckVectorTests(unittest.TestCase):
    _MANIFEST = {"vector_probe": {
        "query": "q", "document_id": "doc", "page_number": 1, "rank": 1, "distance": 0.4,
    }}

    def test_nonempty_token_overlap_fails_fast_without_touching_ollama(self):
        # 54126ad6: manifest_probe.gather_manifest records token_overlap and
        # refuses to build a manifest where it is non-empty, but nothing on
        # the verify side ever read it back -- --artifact-dir explicitly
        # lets this script point at an older/pre-guard artifact whose
        # build-time invariant may not hold. A non-empty value means the
        # whole premise of this check ("a match proves vector search, not
        # word overlap") was never true for THIS artifact.
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "token_overlap": ["полином"]}}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query") as embed_mock:
            ok, detail = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertFalse(ok)
        self.assertIn("полином", detail)
        embed_mock.assert_not_called()

    def test_empty_token_overlap_proceeds_normally(self):
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "token_overlap": []}}
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, _ = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertTrue(ok)

    def test_missing_token_overlap_key_proceeds_normally(self):
        # Older manifest schema, before token_overlap existed at all --
        # .get() must not treat "absent" the same as a real overlap.
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, _ = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertTrue(ok)

    def test_runner_up_distance_margin_is_reported_when_present(self):
        # c52984a9: the manifest may carry runner_up_distance (rnk = 2 from
        # the same ranked CTE) -- when it does, check_vector's detail must
        # surface the margin, not just the manifest/observed distances.
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "runner_up_distance": 0.4657}}
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertTrue(ok)
        self.assertIn("margin to runner-up=0.065700", detail)

    def test_missing_runner_up_distance_omits_margin_without_failing(self):
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertTrue(ok)
        self.assertNotIn("margin", detail)

    def test_ambiguous_runner_up_margin_fails_even_with_good_rank_and_distance(self):
        # The rank tolerance's own justification (see VECTOR_PROBE_RANK_
        # TOLERANCE's docstring) covers exactly two cases: an exact tie
        # (margin == 0) or a margin clearly past the distance tolerance.
        # Anything strictly between the two is neither, and used to pass
        # silently because the margin was only ever printed, never asserted.
        margin = vector_probe_check.VECTOR_PROBE_DISTANCE_TOLERANCE / 2
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "runner_up_distance": 0.4 + margin}}
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertFalse(ok)
        self.assertIn("AMBIGUOUS", detail)

    def test_exact_tie_runner_up_margin_still_passes(self):
        # margin == 0.0 exactly is the documented, designed-for case --
        # the boundary of the ambiguous band, not inside it.
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "runner_up_distance": 0.4}}
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertTrue(ok)

    def test_margin_past_tolerance_passes(self):
        # The other side of the same boundary: clearly separated (margin >
        # tolerance) is the second case the tolerance's own justification
        # covers, not an ambiguous one.
        margin = vector_probe_check.VECTOR_PROBE_DISTANCE_TOLERANCE * 2
        manifest = {"vector_probe": {**self._MANIFEST["vector_probe"], "runner_up_distance": 0.4 + margin}}
        hit = {"rank": 1, "distance": 0.4}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, manifest, "http://x")
        self.assertTrue(ok)
        self.assertNotIn("AMBIGUOUS", detail)

    def test_value_error_from_embed_query_is_caught(self):
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", side_effect=ValueError("dims mismatch")):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("dims mismatch", detail)

    def test_runtime_error_from_embed_query_is_caught_not_propagated(self):
        # embed_query's own internal run_sql (model/dims lookup) can raise a
        # bare RuntimeError on any psql failure; previously only ValueError
        # was caught here, so this would have crashed smoke_test.py's main()
        # before teardown/reporting instead of yielding a clean FAIL line.
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", side_effect=RuntimeError("psql failed")):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("psql failed", detail)

    def test_within_distance_tolerance_passes(self):
        hit = {"rank": 1, "distance": 0.4 + vector_probe_check.VECTOR_PROBE_DISTANCE_TOLERANCE / 2}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, _ = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertTrue(ok)

    def test_rank_exactly_at_tolerance_boundary_passes(self):
        # 2bce654a: rank_ok's own boundary was untested -- every prior test
        # held hit["rank"] == probe["rank"] == 1, making rank_ok trivially
        # True regardless of what VECTOR_PROBE_RANK_TOLERANCE actually was.
        # Distance held well within its own tolerance so this isolates the
        # rank predicate specifically.
        hit = {
            "rank": self._MANIFEST["vector_probe"]["rank"] + vector_probe_check.VECTOR_PROBE_RANK_TOLERANCE,
            "distance": 0.4,
        }
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, _ = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertTrue(ok)

    def test_rank_one_past_tolerance_boundary_fails_even_with_good_distance(self):
        hit = {
            "rank": self._MANIFEST["vector_probe"]["rank"] + vector_probe_check.VECTOR_PROBE_RANK_TOLERANCE + 1,
            "distance": 0.4,
        }
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("rank=", detail)

    def test_distance_beyond_tolerance_fails_even_with_matching_rank(self):
        # Pins the actual review gap this closes: rank alone used to be the
        # only predicate, so a distance that drifted far past the manifest
        # value (a materially different embedding space) still passed as
        # long as the ordinal rank happened to match.
        hit = {"rank": 1, "distance": 0.4 + 10 * vector_probe_check.VECTOR_PROBE_DISTANCE_TOLERANCE}
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=hit):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("delta=", detail)

    def test_no_longer_embedded_fails(self):
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", return_value=None):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("no longer embedded", detail)

    def test_runtime_error_from_page_rank_is_caught_not_propagated(self):
        # page_rank's own run_sql call can fail on a transient psql hiccup
        # exactly like embed_query's above; only the embed_query call used
        # to be guarded, so this would have crashed smoke_test.py's main()
        # before teardown/reporting instead of yielding a clean FAIL line.
        with mock.patch.object(vector_probe_check.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(vector_probe_check.pg_rank_probe, "page_rank", side_effect=RuntimeError("psql failed")):
            ok, detail = vector_probe_check.check_vector({}, self._MANIFEST, "http://x")
        self.assertFalse(ok)
        self.assertIn("psql failed", detail)


if __name__ == "__main__":
    unittest.main()
