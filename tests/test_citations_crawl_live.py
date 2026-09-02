"""The live half of the BFS tests: what real SQL does on conflict.

Split from test_citations_crawl.py, which drives Snowball against a fake
OpenAlex and a planned embedder and never needs a database. This half tests
the one property no stub can carry -- a second crawl over the same works
must add no work/cites row and must add a second set of journal rows -- so
it skips (not fails) when Postgres is unreachable, keys everything it writes
under 'test:citations:' / 'test:%' and deletes it again.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, unit, work
from citations.crawl import Snowball
from citations.store import PostgresWriter
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql, scalar

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
        import pg_graph_common
        pg_graph_common.init_schema(cls.env)
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
