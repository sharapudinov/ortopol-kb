"""Live tests for the four graph consumers, against the real instance.

One fixture serves several of them at once, which is why they share a file
rather than following the module split: citers and cocitation are checked
over the same three works and two edges, and candidates and hybrid over the
same embedded triple. Same convention as test_pg_graph.py -- skipped, not
failed, when Postgres is unreachable, and the fixture is built here rather
than assumed, since the projection may legitimately be empty (no crawl has
run yet).
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph_candidates as pgcand
import pg_graph_cocitation as pgcoci
import pg_graph_common
import pg_graph_cypher as pgc
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

        citer_keys = {r["key"] for r in pgc.citers(self.env, "INDEX", depth=1)}
        self.assertIn(f"{self.PREFIX}b", citer_keys)

        # ALL_PAIRS: the fixture's pair is co-cited once, so it sits at the
        # very end of the count-ordered answer -- the default --limit is a
        # human-sized table, not a set to search for one row in.
        pairs = pgcoci.cocitation(self.env, min_count=1, limit=ALL_PAIRS)
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
                    for p in pgcoci.cocitation(self.env, min_count=1, max_out_degree=cap,
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
        self.assertEqual(len(pgcoci.cocitation(self.env, min_count=1, limit=2)), 2)

    def test_the_top_k_still_plans_as_an_hnsw_scan(self):
        """The shape's whole purpose, asked of the planner rather than
        argued: with the target bound once and read through a LATERAL, the
        ordered scan is still the index's. enable_seqscan=off because at
        438 works the planner is right to prefer a sort -- the question is
        whether the index REMAINS available as the graph grows.
        """
        vec = "[" + ",".join(["0.1"] * 1024) + "]"
        plan = run_sql(
            self.env,
            "SET enable_seqscan = off;\nEXPLAIN (COSTS OFF)\n"
            + pgcand.build_candidates_sql(":'vec'::vector", 0),
            variables={"vec": vec, "top": "20"},
            extra_args=["-t", "-A"],
        ).stdout
        self.assertIn("work_embedding_hnsw", plan, plan)
        self.assertIn("Order By: (embedding <=> t.v)", plan, plan)

    def test_the_link_counts_are_index_lookups_per_row(self):
        """The other half of the shape: the two directions are counted from
        the top-K row's own id, so each is an index lookup -- the primary
        key downward, cites_cited_idx upward -- and nothing reads the whole
        of citation.cites. A grouped aggregate over every edge would show
        as a HashAggregate with a Seq/Index Scan on cites underneath it,
        and would run whatever --top asked for.
        """
        vec = "[" + ",".join(["0.1"] * 1024) + "]"
        plan = run_sql(
            self.env,
            "SET enable_seqscan = off;\nEXPLAIN (COSTS OFF)\n"
            + pgcand.build_candidates_sql(":'vec'::vector", 0),
            variables={"vec": vec, "top": "20"},
            extra_args=["-t", "-A"],
        ).stdout
        self.assertIn("cites_pkey", plan, plan)
        self.assertIn("cites_cited_idx", plan, plan)
        self.assertIn("Index Cond: (citing = n.id)", plan, plan)
        self.assertIn("Index Cond: (cited = n.id)", plan, plan)
        self.assertNotIn("HashAggregate", plan, plan)

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

        ranked = pgcand.candidates(self.env, top=400, min_links=1)
        by_key = {r["key"]: r for r in ranked}
        self.assertIn(f"{self.PREFIX}b", by_key, "узел со связью с нашим документом не найден")
        self.assertEqual(by_key[f"{self.PREFIX}b"]["links"], 1)
        self.assertNotIn(f"{self.PREFIX}c", by_key, "--min-links не отфильтровал узел без связей")

        rows = pgc.hybrid(self.env, "тестовый вопрос", top=400)
        if rows:  # ollama may be unavailable; hybrid() then returns []
            self.assertTrue(all(r["neighbor_key"] for r in rows))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
