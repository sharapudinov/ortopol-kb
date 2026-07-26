"""Unit tests for deploy/drift_probe.py: no live database, no
network. pg_search.resolve_model/embed_with/embed_query and pg_rank_probe.
page_rank/nearest_page are stubbed at drift_probe's own module level, the
same pattern test_manifest_probe.py uses for the same two collaborators.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import drift_probe
from pg_common import PostgresUnavailable


_PROBE = {"query": "q", "document_id": "1997_sm280", "page_number": 7, "rank": 1, "distance": 0.4}
_RESOLVED = ("bge-m3", 1024)


class MeasureDriftTests(unittest.TestCase):
    def test_zero_drift_across_identical_repeats(self):
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=_RESOLVED), \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "page_rank",
                                return_value={"rank": 1, "distance": 0.4}):
            report = drift_probe.measure_drift({}, "http://x", _PROBE, 3)
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["rank_shifts"], [0, 0, 0])
        self.assertEqual(report["distance_deltas"], [0.0, 0.0, 0.0])
        self.assertEqual(report["max_rank_shift"], 0)
        self.assertEqual(report["max_rank_increase"], 0)
        self.assertEqual(report["max_distance_delta"], 0.0)

    def test_reports_the_worst_observed_shift_and_delta_across_repeats(self):
        hits = [
            {"rank": 1, "distance": 0.4},
            {"rank": 2, "distance": 0.4006},
            {"rank": 1, "distance": 0.4001},
        ]
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=_RESOLVED), \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "page_rank", side_effect=hits):
            report = drift_probe.measure_drift({}, "http://x", _PROBE, 3)
        self.assertEqual(report["rank_shifts"], [0, 1, 0])
        self.assertAlmostEqual(report["max_distance_delta"], 0.0006, places=6)
        self.assertEqual(report["max_rank_shift"], 1)
        self.assertEqual(report["max_rank_increase"], 1)

    def test_all_negative_shifts_report_a_positive_magnitude_not_a_negative_number(self):
        # 1cc63c59: a reference at rank > 1 (not today's actual usage, but a
        # real point in measure_drift's documented contract -- any rank) can
        # legitimately see every repeat rank BETTER than the reference,
        # making every shift negative. max(rank_shifts) would then be the
        # LEAST negative value -- reading as "less drift than none" -- which
        # is exactly the bug this pins closed. max_rank_shift must stay a
        # true worst-case magnitude, matching distance_deltas' own
        # abs()-based semantics.
        probe = {**_PROBE, "rank": 3}
        hits = [{"rank": 1, "distance": 0.4}, {"rank": 2, "distance": 0.4}]
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=_RESOLVED), \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "page_rank", side_effect=hits):
            report = drift_probe.measure_drift({}, "http://x", probe, 2)
        self.assertEqual(report["rank_shifts"], [-2, -1])
        self.assertEqual(report["max_rank_shift"], 2)
        # The signed, one-sided companion: no repeat got WORSE than the
        # reference, so this stays non-positive rather than being folded
        # into the magnitude above.
        self.assertEqual(report["max_rank_increase"], -1)

    def test_zero_or_negative_n_rejected(self):
        with self.assertRaises(ValueError):
            drift_probe.measure_drift({}, "http://x", _PROBE, 0)

    def test_model_resolved_once_regardless_of_repeat_count(self):
        # 3860966a: corpus.embedding_model carries exactly one CHECK-
        # constrained row, so the (model, dims) pair cannot change mid-run --
        # resolving it once and reusing it n times, instead of paying one
        # extra psql round trip per repeat, is the whole point of the split
        # between pg_search.resolve_model and pg_search.embed_with.
        with mock.patch.object(drift_probe.pg_search, "resolve_model",
                                return_value=_RESOLVED) as resolve_mock, \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "page_rank",
                                return_value={"rank": 1, "distance": 0.4}):
            drift_probe.measure_drift({}, "http://x", _PROBE, 5)
        resolve_mock.assert_called_once()

    def test_embedding_model_empty_raises(self):
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                drift_probe.measure_drift({}, "http://x", _PROBE, 1)
        self.assertIn("embedding_model", str(ctx.exception))

    def test_unreachable_embedding_service_raises(self):
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=_RESOLVED), \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                drift_probe.measure_drift({}, "http://x", _PROBE, 1)
        self.assertIn("unreachable", str(ctx.exception))

    def test_reference_pair_no_longer_embedded_raises(self):
        with mock.patch.object(drift_probe.pg_search, "resolve_model", return_value=_RESOLVED), \
             mock.patch.object(drift_probe.pg_search, "embed_with", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "page_rank", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                drift_probe.measure_drift({}, "http://x", _PROBE, 1)
        self.assertIn("no longer embedded", str(ctx.exception))


class EstablishReferenceTests(unittest.TestCase):
    def test_builds_a_measure_drift_shaped_probe_from_a_fresh_nearest_page(self):
        nearest = {"document_id": "2015_demr1", "page_number": 69, "distance": 0.2, "rank": 1}
        with mock.patch.object(drift_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "nearest_page", return_value=nearest):
            reference = drift_probe.establish_reference({}, "http://x", "запрос")
        self.assertEqual(reference, {
            "query": "запрос", "document_id": "2015_demr1", "page_number": 69,
            "distance": 0.2, "rank": 1,
        })

    def test_unreachable_embedding_service_raises(self):
        with mock.patch.object(drift_probe.pg_search, "embed_query", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                drift_probe.establish_reference({}, "http://x", "запрос")
        self.assertIn("unreachable", str(ctx.exception))

    def test_no_embedded_pages_raises(self):
        with mock.patch.object(drift_probe.pg_search, "embed_query", return_value="[0.1]"), \
             mock.patch.object(drift_probe.pg_rank_probe, "nearest_page", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                drift_probe.establish_reference({}, "http://x", "запрос")
        self.assertIn("no embedded rows", str(ctx.exception))


class MainPostgresUnavailableTests(unittest.TestCase):
    """Mirrors test_build_package.py's MainPostgresUnavailableTests: no
    Postgres, no network -- drift_probe.main() previously had zero coverage
    of its own control flow (argument parsing, the PostgresUnavailable ->
    print+return 1 path, wiring establish_reference()'s result into
    measure_drift()), unlike build_package.py's main(), which has exactly
    this pair of test classes already.
    """

    def test_returns_1_and_prints_error_without_touching_the_rest(self):
        with mock.patch.object(drift_probe, "load_pgenv",
                                side_effect=PostgresUnavailable("no .pgenv")), \
             mock.patch.object(drift_probe, "establish_reference") as establish_mock, \
             mock.patch("sys.stderr"):
            exit_code = drift_probe.main(["--pgenv", "unused.pgenv"])
        self.assertEqual(exit_code, 1)
        establish_mock.assert_not_called()


class MainHappyPathTests(unittest.TestCase):
    def test_happy_path_wires_the_established_reference_into_measure_drift(self):
        reference = {"query": "q", "document_id": "doc", "page_number": 1, "rank": 1, "distance": 0.4}
        report = {
            "n": 2, "rank_shifts": [0, 0], "distance_deltas": [0.0, 0.0],
            "max_rank_shift": 0, "max_rank_increase": 0, "max_distance_delta": 0.0,
        }
        with mock.patch.object(drift_probe, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
             mock.patch.object(drift_probe, "establish_reference",
                                return_value=reference) as establish_mock, \
             mock.patch.object(drift_probe, "measure_drift", return_value=report) as measure_mock:
            exit_code = drift_probe.main(["--pgenv", "unused.pgenv", "-n", "2"])
        self.assertEqual(exit_code, 0)
        establish_mock.assert_called_once()
        (env, url, ref_arg, n), _kwargs = measure_mock.call_args
        self.assertEqual(env, {"PGUSER": "ortopol"})
        self.assertEqual(ref_arg, reference)
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
