"""The projection's CONTENT reading: fingerprints, not just cardinality.

Separate file from test_pg_graph.py purely for size (kb/CLAUDE.md
FILE_SIZE). The pure part -- what projection_faults() calls a fault --
always runs; the live part skips (not fails) when Postgres is unreachable,
same convention as test_pg_graph.py, because AGE's label tables are a
live-instance fact no stub stands in for.
"""
from __future__ import annotations

import contextlib
import io
import time
import unittest
from unittest import mock

import _pathfix  # noqa: F401

import pg_graph_common
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FAITHFUL = pg_graph_common.Projection(
    work_n=3, cites_n=2, vertex_n=3, edge_n=2,
    work_digest="w", graph_work_digest="w",
    cites_digest="c", graph_cites_digest="c",
)


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class ProjectionFaultsTests(unittest.TestCase):
    """Pure function: what counts as "the projection is not the tables"."""

    def test_matching_counts_and_digests_are_no_fault(self):
        self.assertEqual(pg_graph_common.projection_faults(FAITHFUL), [])

    def test_a_count_gap_is_reported(self):
        faults = pg_graph_common.projection_faults(FAITHFUL._replace(vertex_n=2))
        self.assertEqual(len(faults), 1)
        self.assertIn("diff -1", faults[0])

    def test_equal_counts_with_a_different_vertex_digest_is_still_stale(self):
        faults = pg_graph_common.projection_faults(
            FAITHFUL._replace(graph_work_digest="other"))
        self.assertEqual(len(faults), 1)
        self.assertIn("content fingerprint differs", faults[0])
        self.assertIn("вершины", faults[0])

    def test_equal_counts_with_a_different_edge_digest_is_still_stale(self):
        faults = pg_graph_common.projection_faults(
            FAITHFUL._replace(graph_cites_digest="other"))
        self.assertEqual(len(faults), 1)
        self.assertIn("content fingerprint differs", faults[0])
        self.assertIn("рёбра", faults[0])


class ProjectionReadTests(unittest.TestCase):
    """projection_diff() carries the digests alongside the counts, and the
    three renderings of it all consult projection_faults().
    """

    def test_the_digests_travel_with_the_counts(self):
        with mock.patch.object(pg_graph_common, "graph_exists", return_value=True), \
             mock.patch.object(pg_graph_common, "scalar", side_effect=["438", "2425"]), \
             mock.patch.object(pg_graph_common, "graph_counts", return_value=(438, 2425)), \
             mock.patch.object(pg_graph_common, "content_fingerprints",
                               return_value=("a", "a", "b", "b")):
            seen = pg_graph_common.projection_diff({})
        self.assertEqual(seen, (438, 2425, 438, 2425, "a", "a", "b", "b"))
        self.assertEqual(pg_graph_common.projection_faults(seen), [])

    def test_check_fails_on_a_content_gap_alone(self):
        stale = FAITHFUL._replace(graph_work_digest="other")
        stderr = io.StringIO()
        with mock.patch.object(pg_graph_common, "projection_diff", return_value=stale), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(pg_graph_common.check({}), 1)
        self.assertIn("content fingerprint differs", stderr.getvalue())

    def test_citation_checks_renders_the_same_fault(self):
        import citation_checks

        stale = FAITHFUL._replace(graph_cites_digest="other")
        with mock.patch.object(pg_graph_common, "projection_diff", return_value=stale):
            problems = citation_checks._projection_stale({})
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("PROJECTION STALE: "))
        self.assertIn("content fingerprint differs", problems[0])

    def test_smoke_check_fails_on_a_content_gap_alone(self):
        import _pathfix_deploy  # noqa: F401

        import smoke_checks
        from manifest_contract import CitationMode, Key

        manifest = {Key.CITATION: {Key.CITATION_MODE: CitationMode.FULL_SKELETON,
                                   Key.WORK_COUNT: 3, Key.CITES_COUNT: 2}}
        with mock.patch.object(pg_graph_common, "projection_diff", return_value=FAITHFUL):
            ok, _detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertTrue(ok)
        stale = FAITHFUL._replace(graph_work_digest="other")
        with mock.patch.object(pg_graph_common, "projection_diff", return_value=stale):
            ok, detail = smoke_checks.check_citation_projection({}, manifest)
        self.assertFalse(ok)
        self.assertIn("content fingerprint differs", detail)


class ContentFingerprintLiveTests(unittest.TestCase):
    """A property-only change is invisible to the counts and must not be
    invisible to the check: citations/store.py updates title/kind/year on
    rows that already exist, which is exactly this shape.
    """

    PREFIX = "test:pg_graph:fingerprint:"

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _cleanup(self) -> None:
        run_sql(self.env, f"DELETE FROM citation.work WHERE key LIKE '{self.PREFIX}%';")
        pg_graph_common.project(self.env)

    def _check(self) -> tuple[int, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pg_graph_common.check(self.env)
        return code, stdout.getvalue() + stderr.getvalue()

    def test_a_retitled_row_without_reprojection_is_stale(self):
        self.addCleanup(self._cleanup)
        run_sql(
            self.env,
            "INSERT INTO citation.work (key, title, source, kind) VALUES "
            f"('{self.PREFIX}a', 'before', 'manual', 'external-skeleton');",
        )
        pg_graph_common.project(self.env)
        code, _out = self._check()
        self.assertEqual(code, 0)

        run_sql(self.env, "UPDATE citation.work SET title = 'after' "
                          f"WHERE key = '{self.PREFIX}a';")
        code, out = self._check()
        self.assertEqual(code, 1, "равные счётчики скрыли изменившееся содержание")
        self.assertIn("content fingerprint differs", out)

        pg_graph_common.project(self.env)
        code, out = self._check()
        self.assertEqual(code, 0, out)

    def test_the_four_digests_are_read_in_one_cheap_round_trip(self):
        started = time.monotonic()
        digests = pg_graph_common.content_fingerprints(self.env)
        elapsed = time.monotonic() - started
        self.assertEqual(len(digests), 4)
        for digest in digests:
            self.assertRegex(digest, r"^[0-9a-f]{32}$")
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(digests[2], digests[3])
        self.assertLess(elapsed, 1.0, f"отпечатки заняли {elapsed:.3f} с")


if __name__ == "__main__":
    unittest.main()
