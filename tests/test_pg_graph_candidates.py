"""Offline tests for the candidates consumer (pg_graph_candidates.py): the
shape of the ranking SQL, and the pure tie-break over the rows it returns.

The ranking is the DATABASE's job, so most of what matters here is the
statement's shape -- an HNSW-ordered top-K with nothing joined below the
LIMIT, and the target read once. The live fixture that checks the planner
still agrees is test_pg_graph_consumers_live.py.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph_candidates as pgcand


class CandidatesSqlTests(unittest.TestCase):
    """The ranking is the DATABASE's job: an HNSW-ordered top-K and a
    pre-aggregated link count, not every external-skeleton row shipped
    through psql for Python to sort.
    """

    SQL = pgcand.build_candidates_sql(":'vec'::vector", 0)

    def test_excludes_our_documents_by_kind(self):
        self.assertIn("w.kind = 'external-skeleton'", self.SQL)

    def test_top_k_is_ordered_and_limited_over_the_bare_table(self):
        # Nothing but the single-row target is read below the LIMIT: that is
        # what makes work_embedding_hnsw plannable at all.
        nearest = self.SQL[self.SQL.index("nearest AS ("):self.SQL.index("LIMIT :top")]
        self.assertIn("ORDER BY w.embedding <=> t.v", nearest)
        self.assertIn("CROSS JOIN LATERAL", nearest)
        self.assertNotIn("links", nearest)
        self.assertNotIn("JOIN citation.", nearest)
        self.assertLess(self.SQL.index("LIMIT :top"), self.SQL.index("LEFT JOIN links"))

    def test_the_target_is_evaluated_once_per_statement(self):
        """Spliced into both the score and the ORDER BY, the centroid was
        two textually distinct subqueries -- two aggregations over every
        our-document embedding per call -- and the query vector's 1024
        floats were serialised into the statement twice.
        """
        for expr in (":'vec'::vector", pgcand._CENTROID_EXPR):
            sql = pgcand.build_candidates_sql(expr, 0)
            self.assertEqual(sql.count(expr), 1, expr)
            self.assertIn("target AS MATERIALIZED", sql)
            self.assertEqual(sql.count("t.v"), 2, "счёт и порядок читают одно и то же")

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
        sql = pgcand.build_candidates_sql(":'vec'::vector", 2)
        self.assertIn("JOIN links l ON l.id = n.id AND l.n >= :min_links", sql)
        self.assertGreater(sql.index(":min_links"), sql.index("LIMIT :top"))
        nearest = sql[sql.index("nearest AS ("):sql.index("LIMIT :top")]
        self.assertNotIn("links", nearest)
        self.assertNotIn("JOIN citation.", nearest)

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
        ranked = pgcand.rank_candidates(rows)
        self.assertEqual([r["key"] for r in ranked], ["b", "c", "a"])

    def test_min_links_filters_out_weak_nodes(self):
        rows = [
            {"key": "a", "score": 0.9, "links": 0},
            {"key": "b", "score": 0.5, "links": 3},
        ]
        ranked = pgcand.rank_candidates(rows, min_links=1)
        self.assertEqual([r["key"] for r in ranked], ["b"])

    def test_rows_without_score_are_dropped(self):
        rows = [{"key": "a", "score": None, "links": 5}]
        self.assertEqual(pgcand.rank_candidates(rows), [])

    def test_top_caps_result_length(self):
        rows = [{"key": str(i), "score": float(i), "links": 0} for i in range(5)]
        ranked = pgcand.rank_candidates(rows, top=2)
        self.assertEqual([r["key"] for r in ranked], ["4", "3"])

    def test_ties_break_on_links_then_key_for_determinism(self):
        rows = [
            {"key": "z", "score": 1.0, "links": 1},
            {"key": "a", "score": 1.0, "links": 2},
        ]
        ranked = pgcand.rank_candidates(rows)
        self.assertEqual([r["key"] for r in ranked], ["a", "z"])


if __name__ == "__main__":
    unittest.main()
