"""Tests for pg_graph.py and the citation.cypher_literal / citation.
project_graph SQL functions it wraps (pg_schema_citation.sql).

compare_counts is pure Python and always runs. Everything that touches the
live graph skips (not fails) when Postgres is unreachable, same convention
as test_pg_semantic.py -- the citation schema depends on the AGE extension
being present, which is a live-instance fact, not something a stub can
stand in for.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"


def _expected_cypher_literal(raw: str) -> str:
    """Mirrors citation.cypher_literal()'s five replace() calls, in the
    same order, so the live test asserts against independently-derived
    expected output rather than a hand-transcribed string.
    """
    return (
        raw.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class CompareCountsTests(unittest.TestCase):
    """Pure function, no database: the arithmetic project() --check reports on."""

    def test_matching_counts_are_zero_diff(self):
        self.assertEqual(pg_graph.compare_counts(3, 2, 3, 2), (0, 0))

    def test_check_reports_count_mismatch(self):
        # Fewer vertices than work rows -- the graph is missing something.
        self.assertEqual(pg_graph.compare_counts(5, 2, 3, 2), (-2, 0))
        # More edges than cites rows -- the graph has something the relational
        # tables no longer do (e.g. a stale projection after a DELETE).
        self.assertEqual(pg_graph.compare_counts(3, 2, 3, 5), (0, 3))


class CitationGraphLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _scalar(self, sql, variables=None):
        return run_sql(self.env, sql, variables=variables, extra_args=["-t", "-A"]).stdout.strip()

    def test_cypher_literal_escapes_quotes_backslashes_newlines(self):
        raw = "a'b\"c\\d\ne\rf"
        got = self._scalar("SELECT citation.cypher_literal(:'raw');", variables={"raw": raw})
        self.assertEqual(got, _expected_cypher_literal(raw))

    def test_init_is_idempotent(self):
        pg_graph.init_schema(self.env)
        pg_graph.init_schema(self.env)  # must not raise the second time

    def test_kind_constraints(self):
        # our-document without document_id: the FK to corpus.documents is the
        # whole point of that kind, an unresolvable NULL there is a bug, not
        # a legitimate state.
        with self.assertRaises(RuntimeError):
            run_sql(
                self.env,
                "INSERT INTO citation.work (key, source, kind) "
                "VALUES ('test:pg_graph:kind-our-doc', 'manual', 'our-document');",
            )
        # excluded without exclusion_reason: the reason IS the value of the row
        # for this kind -- see EXCLUDED comment in pg_schema_citation.sql.
        with self.assertRaises(RuntimeError):
            run_sql(
                self.env,
                "INSERT INTO citation.work (key, source, kind) "
                "VALUES ('test:pg_graph:kind-excluded', 'manual', 'excluded');",
            )
        n = self._scalar(
            "SELECT count(*) FROM citation.work WHERE key LIKE 'test:pg_graph:kind-%';"
        )
        self.assertEqual(n, "0", "нарушающая CHECK строка всё же осталась в таблице")

    def test_edge_endpoints_must_exist_fk(self):
        with self.assertRaises(RuntimeError):
            run_sql(
                self.env,
                "INSERT INTO citation.cites (citing, cited, source) "
                "VALUES (-1, -2, 'manual');",
            )

    def _delete_and_reproject(self, prefix: str) -> None:
        run_sql(self.env, f"DELETE FROM citation.work WHERE key LIKE '{prefix}%';")
        pg_graph.project(self.env)

    def test_project_is_idempotent_on_live_db(self):
        prefix = "test:pg_graph:project:"
        self.addCleanup(self._delete_and_reproject, prefix)
        run_sql(
            self.env,
            f"""
            INSERT INTO citation.work (key, title, source, kind) VALUES
              ('{prefix}a', 'A', 'manual', 'external-skeleton'),
              ('{prefix}b', 'B', 'manual', 'external-skeleton'),
              ('{prefix}c', 'C', 'manual', 'external-skeleton');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT a.id, b.id, 'manual' FROM citation.work a, citation.work b
            WHERE a.key = '{prefix}a' AND b.key = '{prefix}b';
            INSERT INTO citation.cites (citing, cited, source)
            SELECT a.id, b.id, 'manual' FROM citation.work a, citation.work b
            WHERE a.key = '{prefix}b' AND b.key = '{prefix}c';
            """,
        )
        first = pg_graph.project(self.env)
        self.assertEqual(pg_graph.check(self.env), 0)
        second = pg_graph.project(self.env)
        self.assertEqual(first, second, "повторная проекция дала другие |V|/|E|")
        self.assertEqual(pg_graph.check(self.env), 0)


if __name__ == "__main__":
    unittest.main()
