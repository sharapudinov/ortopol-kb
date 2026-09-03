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
    """The ranking is the DATABASE's job: an HNSW-ordered top-K and, per
    row of it, two index-served counts -- not every external-skeleton row
    shipped through psql for Python to sort, and not an aggregate over
    every edge in the graph to decorate at most :top of them.
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
        self.assertLess(self.SQL.index("LIMIT :top"), self.SQL.index("LEFT JOIN LATERAL"))

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

    def test_the_two_directions_are_counted_per_row_not_over_the_whole_table(self):
        """Each half is an index lookup -- the primary key (citing, cited,
        source) downward, cites_cited_idx upward -- driven from ONE top-K
        row. The shapes this replaced both read every edge in the graph:
        the OR across two columns (`citing = w.id OR cited = w.id`), which
        no single index scan can serve, and then a grouped CTE over the
        whole of citation.cites joined on `l.id = n.id`, a qualifier the
        planner cannot push down into the aggregate.
        """
        self.assertIn("LEFT JOIN LATERAL", self.SQL)
        self.assertIn("WHERE c.citing = n.id", self.SQL)
        self.assertIn("WHERE c.cited = n.id", self.SQL)
        self.assertNotIn("GROUP BY", self.SQL)
        self.assertNotIn("c.citing = w.id OR c.cited = w.id", self.SQL)

    def test_the_count_is_of_our_own_documents_not_of_edge_rows(self):
        """Whichever direction the edge runs, the OTHER endpoint is what has
        to be one of ours -- and each such document counts ONCE.

        citation.cites is keyed (citing, cited, source) on purpose: the same
        pair attested by two crawl sources is two rows. Summed row counts
        therefore reported a candidate tied to one document as `links = 2`
        and let it through --min-links 2, which is documented as "at least N
        of our own documents"; a mutual pair double-counted the same way.
        """
        self.assertIn("count(DISTINCT e.other)", self.SQL)
        self.assertIn("o.id = e.other AND o.kind = 'our-document'", self.SQL)
        self.assertNotIn("count(*)", self.SQL)

    def test_both_directions_reach_that_one_count_through_a_union(self):
        """UNION ALL rather than two counts added: the addition is what
        counted a document twice, and the union keeps each direction its own
        index-driven scan (the shape the LATERAL exists for).
        """
        lateral = self.SQL[self.SQL.index("LEFT JOIN LATERAL"):]
        self.assertIn("UNION ALL", lateral)
        self.assertEqual(lateral.count("FROM citation.cites c"), 2)
        self.assertNotIn(") AS n\n         + (", self.SQL)

    def test_min_links_is_a_filter_above_the_top_k_not_a_lookup_per_row(self):
        # Measured with EXPLAIN on this instance (enable_seqscan=off): any
        # membership test INSIDE `nearest` -- a correlated subquery, an IN,
        # or a join -- makes the planner drop work_embedding_hnsw and sort
        # instead. Above the LIMIT the index scan survives.
        sql = pgcand.build_candidates_sql(":'vec'::vector", 2)
        self.assertIn("WHERE links.n >= :min_links", sql)
        self.assertGreater(sql.index(":min_links"), sql.index("LIMIT :top"))
        nearest = sql[sql.index("nearest AS ("):sql.index("LIMIT :top")]
        self.assertNotIn("links", nearest)
        self.assertNotIn("JOIN citation.", nearest)

    def test_no_min_links_keeps_every_top_k_row_with_its_count(self):
        self.assertNotIn(":min_links", self.SQL)
        self.assertIn("LEFT JOIN LATERAL", self.SQL)
        self.assertIn("coalesce(links.n, 0)", self.SQL)


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
