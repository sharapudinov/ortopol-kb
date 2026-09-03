"""What the citation graph does on a LIVE instance.

Split off test_pg_graph.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that module asks what the plumbing and the schema TEXT say,
without a database; this one asks the AGE-backed graph itself. Skipped, not
failed, when Postgres is unreachable -- the citation schema depends on the
AGE extension being present, which is a live-instance fact no stub stands
in for.

Every case that writes cleans up after itself and reprojects, so the graph
this leaves behind is the one it found (GRAPH_IS_PROJECTION).
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401

import pg_graph
import pg_graph_common
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
        pg_graph_common.init_schema(self.env)
        pg_graph_common.init_schema(self.env)  # must not raise the second time

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
        pg_graph_common.project(self.env)

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
        first = pg_graph_common.project(self.env)
        self.assertEqual(pg_graph.check(self.env), 0)
        second = pg_graph_common.project(self.env)
        self.assertEqual(first, second, "повторная проекция дала другие |V|/|E|")
        self.assertEqual(pg_graph.check(self.env), 0)

    def test_projection_carries_properties_through_agtype_unmangled(self):
        """A third-party title can hold quotes, backslashes and newlines.
        The projection writes properties as jsonb cast to agtype (Postgres's
        own JSON writer does the escaping); the consumers read them back
        through Cypher with citation.cypher_literal(). Both halves must agree
        on the same string.
        """
        prefix = "test:pg_graph:props:"
        raw_key = prefix + "a'b\\c\"d"
        raw_title = "Ряд \"Фурье\"\\Чебышёва, o'зна\nчение"
        self.addCleanup(self._delete_and_reproject, prefix)
        run_sql(
            self.env,
            "INSERT INTO citation.work (key, title, year, source, kind) "
            "VALUES (:'key', :'title', 1997, 'manual', 'external-skeleton');",
            variables={"key": raw_key, "title": raw_title},
        )
        pg_graph_common.project(self.env)
        escaped = self._scalar("SELECT citation.cypher_literal(:'raw');", variables={"raw": raw_key})
        row = run_sql(
            self.env,
            pg_graph_common.AGE_PREAMBLE
            + "SELECT t::text, y::text, k::text FROM ag_catalog.cypher('citation_graph', "
            + f"$CYPHERQ$MATCH (w:Work {{key: '{escaped}'}}) RETURN w.title, w.year, w.kind$CYPHERQ$) "
            + "AS (t agtype, y agtype, k agtype);",
            extra_args=["-t", "-A", "-F", FIELD_SEP],
        ).stdout.strip()
        self.assertTrue(row, "спроецированный узел не найден по своему же ключу")
        # agtype's ::text cast strips the JSON-style quoting a bare agtype
        # column prints (the same cast pg_graph_candidates relies on), so these
        # are the plain strings, newline and all.
        title, year, kind = row.split(FIELD_SEP)
        self.assertEqual(title, raw_title)
        self.assertEqual(year, "1997")
        self.assertEqual(kind, "external-skeleton")

    def test_projection_omits_absent_year_and_title(self):
        """jsonb_strip_nulls, matching what the Cypher form emitted: a node
        with no year carries no `year` property at all, rather than a null
        one -- `MATCH (w:Work) WHERE w.year IS NULL` and property-count
        answers differ between the two.
        """
        prefix = "test:pg_graph:sparse:"
        self.addCleanup(self._delete_and_reproject, prefix)
        run_sql(
            self.env,
            f"INSERT INTO citation.work (key, source, kind) "
            f"VALUES ('{prefix}a', 'manual', 'external-skeleton');",
        )
        pg_graph_common.project(self.env)
        sparse = self._scalar(
            'SELECT count(*) FROM citation_graph."Work" '
            "WHERE (properties::text)::jsonb->>'key' = :'key' "
            "AND NOT ((properties::text)::jsonb ? 'year') "
            "AND NOT ((properties::text)::jsonb ? 'title');",
            variables={"key": f"{prefix}a"},
        )
        self.assertEqual(sparse, "1", "у узла без года/названия появились лишние свойства")


if __name__ == "__main__":
    unittest.main()
