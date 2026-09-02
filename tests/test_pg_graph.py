"""Tests for pg_graph_common.py (the plumbing pg_graph.py's CLI drives) and
the citation.cypher_literal / citation.project_graph SQL functions it wraps
(pg_schema_citation.sql).

compare_counts is pure Python and always runs, and so does the check that
the CLI layer stayed a CLI layer. Everything that touches the live graph
skips (not fails) when Postgres is unreachable, same convention as
test_pg_semantic.py -- the citation schema depends on the AGE extension
being present, which is a live-instance fact, not something a stub can
stand in for.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_checks
import citation_profile
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


class CompareCountsTests(unittest.TestCase):
    """Pure function, no database: the arithmetic project() --check reports on."""

    def test_matching_counts_are_zero_diff(self):
        self.assertEqual(pg_graph_common.compare_counts(3, 2, 3, 2), (0, 0))

    def test_check_reports_count_mismatch(self):
        # Fewer vertices than work rows -- the graph is missing something.
        self.assertEqual(pg_graph_common.compare_counts(5, 2, 3, 2), (-2, 0))
        # More edges than cites rows -- the graph has something the relational
        # tables no longer do (e.g. a stale projection after a DELETE).
        self.assertEqual(pg_graph_common.compare_counts(3, 2, 3, 5), (0, 3))


class CliLayerTests(unittest.TestCase):
    """pg_graph.py parses arguments, dispatches and prints; the plumbing its
    consumers need lives in pg_graph_common.py. The one function-level
    import it is allowed is the optional paths.py shim (pg_search.py's main()
    carries the same one for the same reason: paths.py is not bundled into
    the deploy artifact). A deferred import of a graph module would mean the
    cycle that forced one is back.
    """

    SOURCE = Path(pg_graph.__file__).read_text(encoding="utf-8")
    TREE = ast.parse(SOURCE)

    def _function_level_imports(self) -> set[str]:
        names = set()
        for node in ast.walk(self.TREE):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    names |= {alias.name for alias in inner.names}
                elif isinstance(inner, ast.ImportFrom):
                    names.add(inner.module or "")
        return names

    def test_no_import_lives_inside_a_function(self):
        self.assertEqual(self._function_level_imports(), set())

    def test_the_query_layer_is_imported_at_module_level(self):
        imported = {
            alias.name
            for node in self.TREE.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("pg_graph_queries", imported)
        self.assertIn("pg_graph_common", imported)

    def test_the_plumbing_is_not_defined_here(self):
        defined = {node.name for node in self.TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("graph_sql", "split_records", "graph_exists", "graph_counts",
                     "compare_counts", "project", "init_schema"):
            self.assertNotIn(name, defined, f"{name}() belongs to pg_graph_common.py")
            self.assertTrue(hasattr(pg_graph_common, name))


class SharedCitationPlumbingTests(unittest.TestCase):
    """citation_schema_exists() and the kind census answer questions three
    consumers ask, so they are read from one place. Written verbatim in two
    modules once, together with FIELD_SEP, they are exactly the drift
    pg_graph_common.py exists to prevent -- hence the guard below, in the
    shape CliLayerTests uses on pg_graph.py.
    """

    CONSUMERS = (Path(citation_checks.__file__), Path(citation_profile.__file__))

    def test_schema_exists_reads_to_regclass(self):
        with mock.patch.object(pg_graph_common, "scalar", return_value="t") as scalar_mock:
            self.assertTrue(pg_graph_common.citation_schema_exists({}))
        self.assertIn("to_regclass('citation.work')", scalar_mock.call_args[0][1])
        with mock.patch.object(pg_graph_common, "scalar", return_value="f"):
            self.assertFalse(pg_graph_common.citation_schema_exists({}))

    def test_kind_census_parses_the_separated_rows(self):
        text = FIELD_SEP.join(("external-skeleton", "382")) + "\n" \
            + FIELD_SEP.join(("our-document", "56")) + "\n"
        with mock.patch.object(pg_graph_common, "run_sql", return_value=mock.Mock(stdout=text)):
            self.assertEqual(pg_graph_common.kind_counts({}),
                             {"external-skeleton": 382, "our-document": 56})

    def test_kind_census_takes_the_callers_narrowing_clause(self):
        with mock.patch.object(pg_graph_common, "run_sql",
                                return_value=mock.Mock(stdout="")) as run_mock:
            self.assertEqual(pg_graph_common.kind_counts({}, " WHERE w.year > 2000"), {})
        self.assertIn(" WHERE w.year > 2000", run_mock.call_args[0][1])

    def _definitions(self, path: Path) -> tuple[set[str], set[str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        return functions, assigned

    def test_neither_consumer_redeclares_the_shared_primitives(self):
        for path in self.CONSUMERS:
            functions, assigned = self._definitions(path)
            self.assertNotIn("citation_schema_exists", functions, path.name)
            self.assertNotIn("kind_counts", functions, path.name)
            self.assertNotIn("counts_by_kind", functions, path.name)
            self.assertNotIn("FIELD_SEP", assigned, f"{path.name}: import it from pg_graph_common")

    def test_neither_consumer_carries_its_own_copy_of_the_queries(self):
        for path in self.CONSUMERS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("to_regclass('citation.work')", text, path.name)
            self.assertNotIn("count(*) FROM citation.work w{where}", text, path.name)

    def test_both_consumers_reach_the_shared_layer(self):
        for path in self.CONSUMERS:
            imported = {
                alias.name
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.ImportFrom) and node.module == "pg_graph_common"
                for alias in node.names
            }
            self.assertIn("citation_schema_exists", imported, path.name)
            self.assertIn("kind_counts", imported, path.name)


class CrawlStepIndexTests(unittest.TestCase):
    """crawl_step is the largest table in the schema and the public cut
    matches two of its columns by equality, once per name it removes.
    """

    SCHEMA = pg_graph_common.SCHEMA_PATH.read_text(encoding="utf-8")

    def test_the_cut_columns_are_indexed_idempotently(self):
        for column in ("frontier_key", "candidate_key"):
            self.assertIn(
                f"CREATE INDEX IF NOT EXISTS crawl_step_{column}_idx "
                f"ON citation.crawl_step ({column});",
                self.SCHEMA,
            )


class ProjectionShapeTests(unittest.TestCase):
    """The projection's COST is a property of its shape, and the shape is
    readable without a database: two bulk INSERT ... SELECT statements into
    the AGE label tables, not one Cypher command per row. The row-by-row
    form this replaced re-planned a command per row AND resolved each edge's
    endpoints with `MATCH (a:Work {key: ...})`, a sequential scan of the
    whole vertex label per edge (AGE indexes no property by itself).
    """

    SCHEMA = pg_graph_common.SCHEMA_PATH.read_text(encoding="utf-8")

    def test_labels_are_filled_by_bulk_insert(self):
        self.assertIn('INSERT INTO citation_graph."Work" (id, properties)', self.SCHEMA)
        self.assertIn(
            'INSERT INTO citation_graph."CITES" (id, start_id, end_id, properties)',
            self.SCHEMA,
        )

    def test_no_per_row_cypher_command_is_assembled(self):
        body = self.SCHEMA[self.SCHEMA.index("CREATE OR REPLACE FUNCTION citation.project_graph()"):]
        self.assertNotIn("FOR w IN", body)
        self.assertNotIn("FOR c IN", body)
        self.assertNotIn("MATCH (a:Work", body)
        self.assertNotIn("CREATE (:Work", body)

    def test_edge_endpoints_are_computed_from_the_relational_id(self):
        # _graphid(<Work label>, citation.work.id) -- so an edge's endpoints
        # are arithmetic, not a lookup by key.
        self.assertIn("ag_catalog._graphid($1, ci.citing)", self.SCHEMA)
        self.assertIn("ag_catalog._graphid($1, ci.cited)", self.SCHEMA)


def _backfill_statement() -> str:
    """The one-time journal backfill, taken from the schema file itself.

    Read out of pg_schema_citation.sql rather than restated here: a copy
    would let the test keep passing over a statement the schema no longer
    applies.
    """
    schema = pg_graph_common.SCHEMA_PATH.read_text(encoding="utf-8")
    start = schema.index("UPDATE citation.crawl_step SET")
    return schema[start:schema.index(";", start) + 1]


class JournalBackfillTests(unittest.TestCase):
    """604 journal rows were written before node_key/score/tau existed, with
    those three facts inside `reason`. The schema parses them back into the
    columns -- once, only into NULLs, and inside a rolled-back transaction
    here so the live journal is not touched by the test.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    FIXTURE = """
    INSERT INTO citation.crawl_step (crawl_id, depth, frontier_key, candidate_key,
                                     action, reason)
    VALUES ('test:backfill', 1, 'W_F', 'W_C', 'keep',
            'kept; score=0.6123 tau=0.5000 relation=cites node=W_NODE'),
           ('test:backfill', 1, 'W_F', 'W_D', 'drop',
            'below-threshold; score=0.4001 tau=0.5000 relation=referenced'),
           ('test:backfill', 0, '2019_rm9846', 'W_EN', 'keep',
            'twin-of=2019_rm9846 seed=W_RU'),
           ('test:backfill', 1, 'W_F', NULL, 'fetch', NULL);
    """

    def _backfilled(self, repeats: int = 1) -> list[str]:
        out = run_sql(
            self.env,
            "BEGIN;\n" + self.FIXTURE + _backfill_statement() * repeats + "\n"
            "SELECT action, coalesce(node_key, '-'), coalesce(score::text, '-'), "
            "coalesce(tau::text, '-') FROM citation.crawl_step "
            "WHERE crawl_id = 'test:backfill' ORDER BY id;\nROLLBACK;\n",
            extra_args=["-t", "-A", "-F", FIELD_SEP],
        ).stdout
        return [line for line in out.splitlines() if line.strip()]

    def test_prose_becomes_columns_row_by_row(self):
        self.assertEqual(self._backfilled(), [
            FIELD_SEP.join(["keep", "W_NODE", "0.6123", "0.5"]),
            FIELD_SEP.join(["drop", "-", "0.4001", "0.5"]),
            FIELD_SEP.join(["keep", "W_RU", "-", "-"]),
            FIELD_SEP.join(["fetch", "-", "-", "-"]),
        ])

    def test_running_it_again_changes_nothing(self):
        self.assertEqual(self._backfilled(repeats=2), self._backfilled())

    def test_the_live_journal_has_no_unparsed_score_left(self):
        """The columns are filled wherever the prose ever carried them --
        the state the migration is supposed to leave the base in."""
        left = run_sql(
            self.env,
            "SELECT count(*) FROM citation.crawl_step "
            "WHERE reason ~ 'score=' AND score IS NULL;",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(left, "0")


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
        self.assertEqual(pg_graph_common.check(self.env), 0)
        second = pg_graph_common.project(self.env)
        self.assertEqual(first, second, "повторная проекция дала другие |V|/|E|")
        self.assertEqual(pg_graph_common.check(self.env), 0)

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
        # column prints (the same cast pg_graph_queries relies on), so these
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
