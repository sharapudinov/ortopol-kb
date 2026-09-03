"""Offline tests for the two Cypher consumers: citers and hybrid.

Both are exercised through pg_graph_cypher, the module that defines them --
the relational pair next door re-exports nothing, so a test importing one
of these names from another module would be asserting over a facade.

No database here: depth validation, the shape of the Cypher and SQL text,
and the splicing rules the untrusted keys go through. The live fixture that
runs all four consumers against the real graph is
test_pg_graph_consumers_live.py.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph_cypher as pgc


class DepthBoundsTests(unittest.TestCase):
    def test_accepts_one_to_three(self):
        for depth in (1, 2, 3):
            self.assertEqual(pgc.validate_depth(depth), depth)

    def test_rejects_zero_and_four(self):
        with self.assertRaises(ValueError):
            pgc.validate_depth(0)
        with self.assertRaises(ValueError):
            pgc.validate_depth(4)


class CitersQueryTests(unittest.TestCase):
    """build_citers_sql() takes an ALREADY-escaped seed key (citers() is
    what calls citation.cypher_literal() to produce it) -- these tests
    exercise only the text splicing and depth handling, not the escaping
    algorithm itself, which is pg_schema_citation.sql's job and is tested
    live against the real SQL function in test_pg_graph_live.py.
    """
    def test_depth_bound_enforced(self):
        with self.assertRaises(ValueError):
            pgc.build_citers_sql("seed", 0)
        with self.assertRaises(ValueError):
            pgc.build_citers_sql("seed", 4)

    def test_embeds_escaped_key_and_depth_verbatim(self):
        sql = pgc.build_citers_sql(r"it\'s", 2)
        self.assertIn("key: 'it\\'s'}", sql)
        self.assertIn("*1..2", sql)
        self.assertIn("ag_catalog.cypher('citation_graph'", sql)
        self.assertIn("$CYPHERQ$", sql)

    def test_delimiter_collision_raises(self):
        with self.assertRaises(ValueError):
            pgc.build_citers_sql("x$CYPHERQ$y", 1)


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
        self.assertEqual([r["key"] for r in pgc.sort_citers(self.ROWS)],
                         ["Wc", "Wa", "Wz", "Wb"])

    def test_the_input_order_does_not_survive_into_the_answer(self):
        shuffled = list(reversed(self.ROWS))
        self.assertEqual(pgc.sort_citers(shuffled), pgc.sort_citers(self.ROWS))


class HybridSqlTests(unittest.TestCase):
    """The AGE+pgvector demonstration stays exactly that -- cypher() in a
    FROM clause, JOINed against citation.work -- with the MATCH restricted
    to the seeds it will actually be joined to. What the restriction buys
    is measured beside the statement itself: on AGE 1.7 the label scan
    happens either way, and it is the agtype materialisation and the joins
    below that shrink, not the traversal.
    """

    SEEDS = [("k1", "k1", "0.742"), ("k2", "k2", "0.5")]
    SQL = pgc.build_hybrid_sql(SEEDS)

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
        self.assertIn("ORDER BY w.embedding <=> q.v", pgc._NEAREST_SEEDS_SQL)
        self.assertIn("citation.cypher_literal(w.key)", pgc._NEAREST_SEEDS_SQL)
        self.assertIn("LIMIT :top", pgc._NEAREST_SEEDS_SQL)

    def test_the_question_vector_is_written_into_the_statement_once(self):
        """psql expands a script variable textually, so a second :'vec'
        puts the 1024 floats into the script again and casts them again.
        The sibling nearest-neighbour query took the target out of both the
        score and the ORDER BY for exactly that reason; this one shares the
        shape rather than keeping its own.
        """
        self.assertEqual(pgc._NEAREST_SEEDS_SQL.count(":'vec'"), 1)
        self.assertIn("WITH q AS MATERIALIZED", pgc._NEAREST_SEEDS_SQL)

    def test_the_match_is_restricted_to_the_seed_keys(self):
        self.assertIn("a.key IN ['k1', 'k2']", self.SQL)
        self.assertIn("b.key IN ['k1', 'k2']", self.SQL)
        # ... and no longer materialises the whole edge set into SQL.
        self.assertNotIn("MATCH (a:Work)-[:CITES]->(b:Work)\n        RETURN", self.SQL)

    def test_keys_are_spliced_already_escaped_not_re_escaped_here(self):
        sql = pgc.build_hybrid_sql([("it's", r"it\'s", "0.1")])
        self.assertIn(r"['it\'s']", sql, "ключ для Cypher переэкранирован здесь")
        self.assertIn("E'it\\'s'", sql, "ключ для SQL не проведён через sql_literal")

    def test_a_score_that_is_not_a_number_never_reaches_the_statement(self):
        with self.assertRaises(ValueError):
            pgc.build_hybrid_sql([("k1", "k1", "0.5); DROP TABLE citation.work; --")])

    def test_delimiter_collision_raises(self):
        with self.assertRaises(ValueError):
            pgc.build_hybrid_sql([("x", "x$CYPHERQ$y", "0.5")])

    def test_no_seeds_means_no_statement_to_run(self):
        self.assertIsNone(pgc.build_hybrid_sql([]))


if __name__ == "__main__":
    unittest.main()
