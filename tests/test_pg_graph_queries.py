"""Tests for pg_graph_queries.py (citers/candidates/cocitation/hybrid).

Offline tests (no database) cover the pure/text-generation logic: depth
validation, Cypher/SQL text shape, VOSviewer export format, ranking. Live
tests, same convention as test_pg_graph.py / test_pg_search_units.py, skip
(not fail) when Postgres is unreachable -- the citation schema and AGE
graph are live-instance facts a stub cannot stand in for. The projection
may legitimately be empty (no crawl has run yet); the live tests here build
their own tiny fixture rather than assuming any pre-existing data.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import pg_graph_common
import pg_graph_queries as pgq
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql


# Enough to defeat the default --limit where a live test has to find one
# specific fixture pair in a graph whose real pairs outrank it.
ALL_PAIRS = 100_000


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class DepthBoundsTests(unittest.TestCase):
    def test_accepts_one_to_three(self):
        for depth in (1, 2, 3):
            self.assertEqual(pgq.validate_depth(depth), depth)

    def test_rejects_zero_and_four(self):
        with self.assertRaises(ValueError):
            pgq.validate_depth(0)
        with self.assertRaises(ValueError):
            pgq.validate_depth(4)


class CitersQueryTests(unittest.TestCase):
    """build_citers_sql() takes an ALREADY-escaped seed key (citers() is
    what calls citation.cypher_literal() to produce it) -- these tests
    exercise only the text splicing and depth handling, not the escaping
    algorithm itself, which is pg_schema_citation.sql's job and is tested
    live against the real SQL function in test_pg_graph.py.
    """
    def test_depth_bound_enforced(self):
        with self.assertRaises(ValueError):
            pgq.build_citers_sql("seed", 0)
        with self.assertRaises(ValueError):
            pgq.build_citers_sql("seed", 4)

    def test_embeds_escaped_key_and_depth_verbatim(self):
        sql = pgq.build_citers_sql(r"it\'s", 2)
        self.assertIn("key: 'it\\'s'}", sql)
        self.assertIn("*1..2", sql)
        self.assertIn("ag_catalog.cypher('citation_graph'", sql)
        self.assertIn("$CYPHERQ$", sql)

    def test_delimiter_collision_raises(self):
        with self.assertRaises(ValueError):
            pgq.build_citers_sql("x$CYPHERQ$y", 1)


class CitersOrderTests(unittest.TestCase):
    """Two runs over unchanged data print the same table. AGE hands back the
    label table's physical order, which a reprojection changes.
    """

    ROWS = [
        {"key": "Wz", "year": 2020, "title": "", "kind": "external-skeleton"},
        {"key": "Wa", "year": 2020, "title": "", "kind": "external-skeleton"},
        {"key": "Wb", "year": None, "title": "", "kind": "external-skeleton"},
        {"key": "Wc", "year": 1998, "title": "", "kind": "external-skeleton"},
    ]

    def test_oldest_first_undated_last_ties_by_key(self):
        self.assertEqual([r["key"] for r in pgq.sort_citers(self.ROWS)],
                         ["Wc", "Wa", "Wz", "Wb"])

    def test_the_input_order_does_not_survive_into_the_answer(self):
        shuffled = list(reversed(self.ROWS))
        self.assertEqual(pgq.sort_citers(shuffled), pgq.sort_citers(self.ROWS))


class CandidatesSqlTests(unittest.TestCase):
    """The ranking is the DATABASE's job: an HNSW-ordered top-K and a
    pre-aggregated link count, not every external-skeleton row shipped
    through psql for Python to sort.
    """

    SQL = pgq.build_candidates_sql(":'vec'::vector", 0)

    def test_excludes_our_documents_by_kind(self):
        self.assertIn("w.kind = 'external-skeleton'", self.SQL)

    def test_top_k_is_ordered_and_limited_over_the_bare_table(self):
        # Nothing joined below the LIMIT: that is what makes
        # work_embedding_hnsw plannable at all.
        nearest = self.SQL[self.SQL.index("nearest AS ("):self.SQL.index("LIMIT :top")]
        self.assertIn("ORDER BY w.embedding <=>", nearest)
        self.assertNotIn("JOIN", nearest)
        self.assertLess(self.SQL.index("LIMIT :top"), self.SQL.index("LEFT JOIN links"))

    def test_link_count_is_a_grouped_aggregate_not_a_per_row_subquery(self):
        # The OR across two columns (citing = w.id OR cited = w.id) is what
        # no index can serve; UNION ALL of the two directions, aggregated
        # once, is what replaced it.
        self.assertIn("UNION ALL", self.SQL)
        self.assertIn("GROUP BY e.id", self.SQL)
        self.assertNotIn("c.citing = w.id OR c.cited = w.id", self.SQL)

    def test_min_links_is_a_join_above_the_top_k_not_a_lookup_per_row(self):
        # Measured with EXPLAIN on this instance (enable_seqscan=off): any
        # membership test against `links` INSIDE `nearest` -- a correlated
        # subquery, an IN, or a join -- makes the planner drop
        # work_embedding_hnsw and sort instead. Above the LIMIT the index
        # scan survives and `links` is hashed once.
        sql = pgq.build_candidates_sql(":'vec'::vector", 2)
        self.assertIn("JOIN links l ON l.id = n.id AND l.n >= :min_links", sql)
        self.assertGreater(sql.index(":min_links"), sql.index("LIMIT :top"))
        nearest = sql[sql.index("nearest AS ("):sql.index("LIMIT :top")]
        self.assertNotIn("links", nearest)
        self.assertNotIn("JOIN", nearest)

    def test_no_min_links_keeps_every_top_k_row_with_its_count(self):
        self.assertNotIn(":min_links", self.SQL)
        self.assertIn("LEFT JOIN links l ON l.id = n.id", self.SQL)


class CandidatesRankingTests(unittest.TestCase):
    """rank_candidates() stays a pure tie-break over the already-limited
    set: SQL orders by distance, this settles equal scores deterministically.
    """

    def test_sorts_by_score_descending(self):
        rows = [
            {"key": "a", "score": 0.5, "links": 0},
            {"key": "b", "score": 0.9, "links": 0},
            {"key": "c", "score": 0.7, "links": 0},
        ]
        ranked = pgq.rank_candidates(rows)
        self.assertEqual([r["key"] for r in ranked], ["b", "c", "a"])

    def test_min_links_filters_out_weak_nodes(self):
        rows = [
            {"key": "a", "score": 0.9, "links": 0},
            {"key": "b", "score": 0.5, "links": 3},
        ]
        ranked = pgq.rank_candidates(rows, min_links=1)
        self.assertEqual([r["key"] for r in ranked], ["b"])

    def test_rows_without_score_are_dropped(self):
        rows = [{"key": "a", "score": None, "links": 5}]
        self.assertEqual(pgq.rank_candidates(rows), [])

    def test_top_caps_result_length(self):
        rows = [{"key": str(i), "score": float(i), "links": 0} for i in range(5)]
        ranked = pgq.rank_candidates(rows, top=2)
        self.assertEqual([r["key"] for r in ranked], ["4", "3"])

    def test_ties_break_on_links_then_key_for_determinism(self):
        rows = [
            {"key": "z", "score": 1.0, "links": 1},
            {"key": "a", "score": 1.0, "links": 2},
        ]
        ranked = pgq.rank_candidates(rows)
        self.assertEqual([r["key"] for r in ranked], ["a", "z"])


class CocitationSqlTests(unittest.TestCase):
    """The self-join is bounded on BOTH sides: which citers may generate
    pairs at all, and how many pairs come back.
    """

    SQL = pgq._COCITATION_SQL

    def test_out_degree_cap_is_applied_before_the_self_join(self):
        citers = self.SQL[self.SQL.index("WITH citers AS ("):self.SQL.index("pairs AS (")]
        self.assertIn("HAVING count(*) <= :max_out_degree", citers)
        self.assertLess(self.SQL.index(":max_out_degree"),
                        self.SQL.index("JOIN citation.cites c2"))

    def test_the_self_join_only_sees_capped_citers(self):
        self.assertIn("JOIN citers ON citers.citing = c1.citing", self.SQL)

    def test_result_is_limited_and_ordered_deterministically(self):
        self.assertIn("ORDER BY p.n DESC, wa.key, wb.key", self.SQL)
        self.assertIn("LIMIT :limit", self.SQL)
        self.assertLess(self.SQL.index("ORDER BY p.n DESC"), self.SQL.index("LIMIT :limit"))

    def test_defaults_are_named_once_and_reach_the_query(self):
        seen = {}

        def fake_graph_sql(env, sql, variables=None, extra_args=None):
            seen.update(variables or {})
            return mock.Mock(stdout="")

        with mock.patch.object(pgq.pg_graph_common, "graph_sql", side_effect=fake_graph_sql):
            pgq.cocitation({})
        self.assertEqual(seen["max_out_degree"], str(pgq.MAX_OUT_DEGREE))
        self.assertEqual(seen["limit"], str(pgq.COCITATION_LIMIT))


class VosviewerExportTests(unittest.TestCase):
    """Format: https://app.vosviewer.com/docs/file-types/map-and-network-file-type/
    -- tab-delimited; map file has a header row and one row per distinct
    item; network file has NO header, one 'id1\\tid2\\tweight' row per link.
    """
    PAIRS = [
        {"a_key": "k:a", "a_title": "A", "b_key": "k:b", "b_title": "B", "count": 3},
        {"a_key": "k:a", "a_title": "A", "b_key": "k:c", "b_title": "C", "count": 1},
    ]

    def test_map_has_header_and_one_row_per_distinct_node(self):
        map_lines, _ = pgq.build_vosviewer_export(self.PAIRS)
        self.assertEqual(map_lines[0], "id\tlabel")
        self.assertEqual(len(map_lines), 1 + 3, "3 distinct nodes: k:a, k:b, k:c")

    def test_network_has_no_header_and_one_row_per_pair(self):
        _, network_lines = pgq.build_vosviewer_export(self.PAIRS)
        self.assertEqual(len(network_lines), len(self.PAIRS))
        for line in network_lines:
            self.assertEqual(len(line.split("\t")), 3)

    def test_node_ids_are_sequential_integers_from_one(self):
        map_lines, network_lines = pgq.build_vosviewer_export(self.PAIRS)
        ids = sorted(int(line.split("\t")[0]) for line in map_lines[1:])
        self.assertEqual(ids, [1, 2, 3])
        # every id referenced in the network file must appear in the map
        map_ids = {line.split("\t")[0] for line in map_lines[1:]}
        for line in network_lines:
            a, b, _weight = line.split("\t")
            self.assertIn(a, map_ids)
            self.assertIn(b, map_ids)

    def test_write_creates_both_files_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path, network_path, n_nodes, n_edges = pgq.write_vosviewer_export(self.PAIRS, Path(tmp))
            self.assertTrue(map_path.is_file())
            self.assertTrue(network_path.is_file())
            self.assertEqual(n_nodes, 3)
            self.assertEqual(n_edges, 2)


class HybridSqlTests(unittest.TestCase):
    """The AGE+pgvector demonstration stays exactly that -- cypher() in a
    FROM clause, JOINed against citation.work -- but the traversal is
    bounded by the seeds it will actually be joined to, so its cost is
    proportional to `top` and not to |E|.
    """

    SEEDS = [("k1", "k1", "0.742"), ("k2", "k2", "0.5")]
    SQL = pgq.build_hybrid_sql(SEEDS)

    def test_cypher_sits_in_a_from_clause(self):
        self.assertIn("FROM ag_catalog.cypher(", self.SQL)

    def test_joins_the_cypher_output_with_citation_work(self):
        self.assertIn("JOIN citation.work w ON w.key = e.cited_key", self.SQL)
        self.assertIn("JOIN citation.work w ON w.key = e.citing_key", self.SQL)

    def test_the_seeds_are_carried_in_not_searched_for_again(self):
        """One hybrid call, one nearest-neighbour scan. This statement used
        to re-run the identical top-K over the HNSW index, with the 1024-
        float question vector serialised into a second psql script.
        """
        self.assertNotIn("<=>", self.SQL, "второй запрос всё ещё ищет по вектору")
        self.assertNotIn(":'vec'", self.SQL)
        self.assertIn("WITH nearest(key, score) AS (\n    VALUES ", self.SQL)
        self.assertIn("(E'k1', 0.742::double precision)", self.SQL)

    def test_the_one_vector_scan_uses_the_question_embedding(self):
        self.assertIn("ORDER BY embedding <=> :'vec'::vector", pgq._NEAREST_SEEDS_SQL)
        self.assertIn("citation.cypher_literal(key)", pgq._NEAREST_SEEDS_SQL)
        self.assertIn("LIMIT :top", pgq._NEAREST_SEEDS_SQL)

    def test_traversal_is_bounded_by_the_seed_keys(self):
        self.assertIn("a.key IN ['k1', 'k2']", self.SQL)
        self.assertIn("b.key IN ['k1', 'k2']", self.SQL)
        # ... and no longer asks for the whole edge set.
        self.assertNotIn("MATCH (a:Work)-[:CITES]->(b:Work)\n        RETURN", self.SQL)

    def test_keys_are_spliced_already_escaped_not_re_escaped_here(self):
        sql = pgq.build_hybrid_sql([("it's", r"it\'s", "0.1")])
        self.assertIn(r"['it\'s']", sql, "ключ для Cypher переэкранирован здесь")
        self.assertIn("E'it\\'s'", sql, "ключ для SQL не проведён через sql_literal")

    def test_a_score_that_is_not_a_number_never_reaches_the_statement(self):
        with self.assertRaises(ValueError):
            pgq.build_hybrid_sql([("k1", "k1", "0.5); DROP TABLE citation.work; --")])

    def test_delimiter_collision_raises(self):
        with self.assertRaises(ValueError):
            pgq.build_hybrid_sql([("x", "x$CYPHERQ$y", "0.5")])

    def test_no_seeds_means_no_statement_to_run(self):
        self.assertIsNone(pgq.build_hybrid_sql([]))


class LiveConsumersTests(unittest.TestCase):
    """Fixture: 3 works ('test:pgq:' prefix), 2 cites --
    a=our-document(document_id='INDEX'), b/c=external-skeleton,
    b cites a AND b cites c. That single shape exercises both queries at
    once: citers(a) must find b (b cites a); cocitation must find the pair
    (a, c) -- both cited by the same citing work b.
    """
    PREFIX = "test:pgq:"

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _cleanup(self):
        run_sql(self.env, f"DELETE FROM citation.work WHERE key LIKE '{self.PREFIX}%';")
        pg_graph_common.project(self.env)

    def test_citers_finds_citing_work_and_cocitation_finds_shared_target_pair(self):
        self.addCleanup(self._cleanup)
        run_sql(
            self.env,
            f"""
            INSERT INTO citation.work (key, title, source, kind, document_id) VALUES
              ('{self.PREFIX}a', 'Seed A', 'manual', 'our-document', 'INDEX');
            INSERT INTO citation.work (key, title, source, kind) VALUES
              ('{self.PREFIX}b', 'Citer B', 'manual', 'external-skeleton'),
              ('{self.PREFIX}c', 'Cited C', 'manual', 'external-skeleton');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}b' AND y.key = '{self.PREFIX}a';
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}b' AND y.key = '{self.PREFIX}c';
            """,
        )
        pg_graph_common.project(self.env)

        citer_keys = {r["key"] for r in pgq.citers(self.env, "INDEX", depth=1)}
        self.assertIn(f"{self.PREFIX}b", citer_keys)

        # ALL_PAIRS: the fixture's pair is co-cited once, so it sits at the
        # very end of the count-ordered answer -- the default --limit is a
        # human-sized table, not a set to search for one row in.
        pairs = pgq.cocitation(self.env, min_count=1, limit=ALL_PAIRS)
        found = {(p["a_key"], p["b_key"]) for p in pairs} | {(p["b_key"], p["a_key"]) for p in pairs}
        self.assertIn((f"{self.PREFIX}a", f"{self.PREFIX}c"), found)

    def test_a_citer_past_the_out_degree_cap_generates_no_pairs(self):
        """The fixture's one citer cites three works, so it would produce
        three pairs; under a cap of two it produces none, and the pairs its
        under-cap sibling produces are unaffected.
        """
        self.addCleanup(self._cleanup)
        run_sql(
            self.env,
            f"""
            INSERT INTO citation.work (key, title, source, kind) VALUES
              ('{self.PREFIX}fat', 'Bibliography', 'manual', 'external-skeleton'),
              ('{self.PREFIX}thin', 'Ordinary citer', 'manual', 'external-skeleton'),
              ('{self.PREFIX}x', 'X', 'manual', 'external-skeleton'),
              ('{self.PREFIX}y', 'Y', 'manual', 'external-skeleton'),
              ('{self.PREFIX}z', 'Z', 'manual', 'external-skeleton');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}fat'
              AND y.key IN ('{self.PREFIX}x', '{self.PREFIX}y', '{self.PREFIX}z');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}thin'
              AND y.key IN ('{self.PREFIX}x', '{self.PREFIX}y');
            """,
        )
        ours = {f"{self.PREFIX}x", f"{self.PREFIX}y", f"{self.PREFIX}z"}

        def pairs_under(cap):
            return {(p["a_key"], p["b_key"])
                    for p in pgq.cocitation(self.env, min_count=1, max_out_degree=cap,
                                            limit=ALL_PAIRS)
                    if p["a_key"] in ours and p["b_key"] in ours}

        self.assertEqual(len(pairs_under(3)), 3, "все три пары «жирного» цитирующего")
        self.assertEqual(pairs_under(2), {(f"{self.PREFIX}x", f"{self.PREFIX}y")},
                         "под кэпом остаётся только пара от цитирующего в пределах кэпа")

    def test_limit_caps_the_answer_at_the_most_co_cited_pairs(self):
        self.addCleanup(self._cleanup)
        run_sql(
            self.env,
            f"""
            INSERT INTO citation.work (key, title, source, kind) VALUES
              ('{self.PREFIX}c1', 'Citer', 'manual', 'external-skeleton'),
              ('{self.PREFIX}x', 'X', 'manual', 'external-skeleton'),
              ('{self.PREFIX}y', 'Y', 'manual', 'external-skeleton'),
              ('{self.PREFIX}z', 'Z', 'manual', 'external-skeleton');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}c1'
              AND y.key IN ('{self.PREFIX}x', '{self.PREFIX}y', '{self.PREFIX}z');
            """,
        )
        self.assertEqual(len(pgq.cocitation(self.env, min_count=1, limit=2)), 2)

    def test_candidates_and_hybrid_answer_from_the_same_fixture(self):
        """candidates() must find an external-skeleton node linked to one of
        our own documents, ranked by the embedding it was given; hybrid()
        must find that node's neighbour through the graph. Both now do their
        top-K in SQL, so the fixture also covers "the LIMIT did not cut the
        answer away".
        """
        self.addCleanup(self._cleanup)
        vec = "[" + ",".join(["0.1"] * 1024) + "]"
        run_sql(
            self.env,
            f"""
            INSERT INTO citation.work (key, title, source, kind, document_id, embedding) VALUES
              ('{self.PREFIX}a', 'Seed A', 'manual', 'our-document', 'INDEX', :'vec');
            INSERT INTO citation.work (key, title, source, kind, embedding) VALUES
              ('{self.PREFIX}b', 'Citer B', 'manual', 'external-skeleton', :'vec'),
              ('{self.PREFIX}c', 'Cited C', 'manual', 'external-skeleton', :'vec');
            INSERT INTO citation.cites (citing, cited, source)
            SELECT x.id, y.id, 'manual' FROM citation.work x, citation.work y
            WHERE x.key = '{self.PREFIX}b' AND y.key = '{self.PREFIX}a';
            """,
            variables={"vec": vec},
        )
        pg_graph_common.project(self.env)

        ranked = pgq.candidates(self.env, top=400, min_links=1)
        by_key = {r["key"]: r for r in ranked}
        self.assertIn(f"{self.PREFIX}b", by_key, "узел со связью с нашим документом не найден")
        self.assertEqual(by_key[f"{self.PREFIX}b"]["links"], 1)
        self.assertNotIn(f"{self.PREFIX}c", by_key, "--min-links не отфильтровал узел без связей")

        rows = pgq.hybrid(self.env, "тестовый вопрос", top=400)
        if rows:  # ollama may be unavailable; hybrid() then returns []
            self.assertTrue(all(r["neighbor_key"] for r in rows))


if __name__ == "__main__":
    unittest.main()
