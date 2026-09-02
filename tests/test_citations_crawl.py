"""The BFS itself: what the snowball keeps, drops, journals and writes.

The offline half drives Snowball with a fake OpenAlex and a planned
embedder, so keep/drop, the journal shape and the edge derivation are
asserted exactly. The live half tests the one property no stub can carry --
what real SQL does on conflict -- and skips when Postgres is unreachable,
keying everything under 'test:citations:' and deleting what it wrote.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, unit, work
from citations.crawl import Snowball
from citations.store import DryRunWriter, PostgresWriter
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql, scalar

def build_snowball(writer, *, tau, embedder=None, records=None, citers=None, seeds=None):
    seeds = seeds or {"doc_a": "W_SEED_A"}
    records = records or [work("W_SEED_A", title="Seed Chebyshev")]
    client = FakeClient(records, citers)
    embedder = embedder or PlannedEmbedder({"Seed": unit(0)})
    return client, Snowball(client, embedder, writer, tau=tau, crawl_id="test-crawl",
                            log=lambda *_: None), seeds


class CrawlTests(unittest.TestCase):
    def test_seed_without_match_is_journaled(self):
        writer = DryRunWriter()
        client, snowball, seeds = build_snowball(writer, tau=0.0)
        snowball.seed(["doc_a", "doc_b", "doc_c"], seeds)
        missing = [s for s in writer.steps_seen if s["action"] == "seed-missing"]
        self.assertEqual(sorted(s["frontier_key"] for s in missing), ["doc_b", "doc_c"])
        self.assertTrue(all(s["reason"] == "not in OpenAlex (run 85)" for s in missing))
        # ... and no work row stands for a document the source does not have.
        self.assertEqual(sorted(snowball.registry.nodes), ["W_SEED_A"])

    def test_seed_matched_but_unreturned_is_an_error_row(self):
        writer = DryRunWriter()
        client = FakeClient([])  # OpenAlex returns nothing for the matched id
        snowball = Snowball(client, PlannedEmbedder({}), writer, tau=0.0,
                            crawl_id="c", log=lambda *_: None)
        with self.assertRaises(ValueError):
            snowball.seed(["doc_a"], {"doc_a": "W_GONE"})  # no seeds -> no centroid
        self.assertEqual([s["action"] for s in writer.steps_seen], ["error"])

    def test_dropped_candidate_leaves_journal_row_not_work_row(self):
        writer = DryRunWriter()
        seed = work("W_SEED_A", title="Seed Chebyshev")
        near = work("W_NEAR", title="Near Chebyshev discrete")
        far = work("W_FAR", title="Far unrelated topic")
        client = FakeClient([seed, near, far],
                            citers={"W_SEED_A": [
                                work("W_NEAR", title="Near Chebyshev discrete",
                                     refs=["W_SEED_A"]),
                                work("W_FAR", title="Far unrelated topic",
                                     refs=["W_SEED_A"]),
                            ]})
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Far": unit(500)})
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED_A"})
        kept = snowball.expand(["W_SEED_A"], 1)

        self.assertEqual(kept, ["W_NEAR"])
        self.assertNotIn("W_FAR", {n.key for n in writer.works_seen})
        drops = [s for s in writer.steps_seen if s["action"] == "drop"]
        self.assertEqual([s["candidate_key"] for s in drops], ["W_FAR"])
        self.assertIn("below-threshold; score=0.0000 tau=0.5000", drops[0]["reason"])
        keeps = [s for s in writer.steps_seen if s["action"] == "keep"]
        self.assertEqual([s["candidate_key"] for s in keeps], ["W_NEAR"])
        self.assertIn("kept; score=1.0000 tau=0.5000", keeps[0]["reason"])

    def test_fetch_row_counts_what_a_frontier_node_yielded(self):
        writer = DryRunWriter()
        seed = work("W_SEED_A", title="Seed Chebyshev", refs=["W_REF"])
        client = FakeClient(
            [seed, work("W_REF", title="Seed reference")],
            citers={"W_SEED_A": [work("W_C", title="Seed citer", refs=["W_SEED_A"])]},
        )
        embedder = PlannedEmbedder({"Seed": unit(0)})
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED_A"})
        snowball.expand(["W_SEED_A"], 1)
        fetch = [s for s in writer.steps_seen if s["action"] == "fetch"]
        self.assertEqual(len(fetch), 1)
        self.assertEqual(fetch[0]["frontier_key"], "W_SEED_A")
        self.assertEqual(fetch[0]["n_found"], 2)  # one citer up, one reference down
        self.assertEqual(fetch[0]["n_kept"], 2)

    def test_edges_are_written_between_any_two_known_nodes(self):
        writer = DryRunWriter()
        seed_a = work("W_A", title="Seed one", refs=["W_B"])
        seed_b = work("W_B", title="Seed two")
        client = FakeClient([seed_a, seed_b], citers={})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a", "doc_b"], {"doc_a": "W_A", "doc_b": "W_B"})
        snowball.expand(["W_A", "W_B"], 1)
        self.assertIn(("W_A", "W_B", "referenced", "W_A"), writer.edges_seen)

    def test_depth_two_expands_only_what_depth_one_kept(self):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        near = work("W_NEAR", title="Near Chebyshev", refs=["W_SEED"])
        far = work("W_FAR", title="Far unrelated", refs=["W_SEED"])
        client = FakeClient([seed, near, far], citers={"W_SEED": [near, far]})
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Far": unit(500)})
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        snowball.run(2)
        self.assertEqual(client.cites_batches, [["W_SEED"], ["W_NEAR"]])

    def test_calibrate_scores_every_candidate_and_writes_no_work(self):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        client = FakeClient(
            [seed],
            citers={"W_SEED": [work("W_C1", title="Near Chebyshev", refs=["W_SEED"]),
                               work("W_C2", title="Far unrelated", refs=["W_SEED"])]},
        )
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Far": unit(500)})
        snowball = Snowball(client, embedder, writer, tau=float("inf"), crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        rows = snowball.calibrate()
        self.assertEqual(sorted(r["candidate_key"] for r in rows), ["W_C1", "W_C2"])
        self.assertAlmostEqual(next(r["score"] for r in rows if r["candidate_key"] == "W_C1"), 1.0)
        self.assertAlmostEqual(next(r["score"] for r in rows if r["candidate_key"] == "W_C2"), 0.0)
        self.assertEqual(writer.works_seen, [])

    def test_two_candidates_that_are_one_work_are_written_once(self):
        """The twin union happens on add(), after scoring: without a guard the
        node lands in the write batch twice and the whole upsert aborts with
        "ON CONFLICT DO UPDATE command cannot affect row a second time"."""
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        original = work("W_RU", title="Near Chebyshev original",
                        doi="10.4213/sm723", refs=["W_SEED"])
        translation = work("W_EN", title="Near Chebyshev translation",
                           doi="10.4213/SM723", refs=["W_SEED"])
        client = FakeClient([seed, original, translation],
                            citers={"W_SEED": [original, translation]})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0), "Near": unit(0)}),
                            writer, tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        kept = snowball.expand(["W_SEED"], 1)

        self.assertEqual(kept, ["W_RU"], "двойник по DOI записан вторым узлом")
        self.assertEqual([n.key for n in writer.works_seen].count("W_RU"), 1)
        keeps = [s for s in writer.steps_seen if s["action"] == "keep"]
        self.assertEqual(sorted(s["candidate_key"] for s in keeps), ["W_EN", "W_RU"],
                         "слияние двойников спрятано от журнала")
        self.assertTrue(all("node=W_RU" in s["reason"] for s in keeps))

    def test_titleless_candidate_scores_below_every_threshold(self):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        blank = work("W_BLANK", title="", refs=["W_SEED"])
        blank["display_name"] = ""
        client = FakeClient([seed, blank], citers={"W_SEED": [blank]})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.0, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        snowball.expand(["W_SEED"], 1)
        drops = [s for s in writer.steps_seen if s["action"] == "drop"]
        self.assertEqual([s["candidate_key"] for s in drops], ["W_BLANK"])
        self.assertIn("score=-1.0000 tau=0.0000", drops[0]["reason"])


def _live_env():
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


PREFIX = "test:citations:"


class IdempotencyLiveTests(unittest.TestCase):
    """The one property a stub cannot carry: what real SQL does on conflict."""

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        import pg_graph
        pg_graph.init_schema(cls.env)
        documents = run_sql(
            cls.env,
            "SELECT id FROM corpus.documents WHERE source_dir = 'theory/iis' "
            "AND extraction_state <> 'metadata' ORDER BY id LIMIT 2;",
            extra_args=["-t", "-A"],
        ).stdout.split()
        if len(documents) < 2:
            raise unittest.SkipTest("в базе нет двух документов ИИШ для семян")
        cls.documents = documents

    def _cleanup(self):
        run_sql(self.env, f"DELETE FROM citation.work WHERE key LIKE '{PREFIX}%';")
        run_sql(self.env, "DELETE FROM citation.crawl_step WHERE crawl_id LIKE 'test:%';")

    def _counts(self):
        work_n = scalar(self.env,
                        f"SELECT count(*) FROM citation.work WHERE key LIKE '{PREFIX}%';")
        cites_n = scalar(
            self.env,
            "SELECT count(*) FROM citation.cites c JOIN citation.work w ON w.id = c.citing "
            f"WHERE w.key LIKE '{PREFIX}%';")
        steps_n = scalar(self.env,
                         "SELECT count(*) FROM citation.crawl_step WHERE crawl_id LIKE 'test:%';")
        return int(work_n), int(cites_n), int(steps_n)

    def _crawl(self):
        seed_a = work(f"{PREFIX}A", title="Seed one Chebyshev", refs=[f"{PREFIX}B"])
        seed_b = work(f"{PREFIX}B", title="Seed two Chebyshev")
        citer = work(f"{PREFIX}C", title="Near Chebyshev citer", refs=[f"{PREFIX}A"])
        far = work(f"{PREFIX}D", title="Far unrelated", refs=[f"{PREFIX}A"])
        client = FakeClient([seed_a, seed_b, citer, far],
                            citers={f"{PREFIX}A": [citer, far]})
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Far": unit(500)})
        snowball = Snowball(client, embedder, PostgresWriter(self.env), tau=0.5,
                            crawl_id="test:crawl", log=lambda *_: None)
        snowball.seed(self.documents,
                      {self.documents[0]: f"{PREFIX}A", self.documents[1]: f"{PREFIX}B"})
        snowball.run(1)

    def test_second_run_adds_nothing(self):
        self.addCleanup(self._cleanup)
        self._cleanup()
        self._crawl()
        first_work, first_cites, first_steps = self._counts()
        self.assertGreater(first_work, 0)
        self.assertGreater(first_cites, 0)

        self._crawl()
        second_work, second_cites, second_steps = self._counts()
        self.assertEqual((second_work, second_cites), (first_work, first_cites),
                         "повторный прогон добавил строки work/cites")
        # crawl_step is a journal, not a set: a second pass is a second set of
        # decisions and must be recorded as such, or "why is X here" loses its
        # history the moment the crawl is rerun.
        self.assertEqual(second_steps, 2 * first_steps)

    def test_kept_node_carries_its_embedding_and_evidence(self):
        self.addCleanup(self._cleanup)
        self._cleanup()
        self._crawl()
        missing = scalar(
            self.env,
            f"SELECT count(*) FROM citation.work WHERE key LIKE '{PREFIX}%' "
            "AND embedding IS NULL;")
        self.assertEqual(missing, "0", "узел без вектора заставит pg_embed.py досчитывать")
        evidence = scalar(
            self.env,
            f"SELECT evidence -> 'records' -> 0 ->> 'id' FROM citation.work "
            f"WHERE key = '{PREFIX}C';")
        self.assertEqual(evidence, f"https://openalex.org/{PREFIX}C")

    def test_dropped_candidate_has_no_work_row_but_a_journal_row(self):
        self.addCleanup(self._cleanup)
        self._cleanup()
        self._crawl()
        self.assertEqual(
            scalar(self.env, f"SELECT count(*) FROM citation.work WHERE key = '{PREFIX}D';"),
            "0")
        self.assertEqual(
            scalar(self.env, "SELECT count(*) FROM citation.crawl_step "
                             f"WHERE candidate_key = '{PREFIX}D' AND action = 'drop';"),
            "1")


if __name__ == "__main__":
    unittest.main()
