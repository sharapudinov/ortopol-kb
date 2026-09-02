"""The BFS itself: what the snowball keeps, drops, journals and writes.

Snowball is driven with a fake OpenAlex and a planned embedder, so keep/drop,
the journal shape and the edge derivation are asserted exactly, and nothing
here needs a database. The one property no stub can carry -- what real SQL
does on conflict -- lives next door in test_citations_crawl_live.py.
"""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import (
    FakeClient,
    PlannedEmbedder,
    build_snowball,
    unit,
    work,
)
from citations import journal, seeding, store, store_sql
from citations.crawl import Snowball
from citations.frontier import EMBED_BATCH
from citations.registry import Node, WorkRegistry, scoring_fields
from citations.store import DryRunWriter, PostgresWriter, Writer
from pg_copy import CopyResult


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
        self.assertEqual((drops[0]["score"], drops[0]["tau"]), (0.0, 0.5))
        keeps = [s for s in writer.steps_seen if s["action"] == "keep"]
        self.assertEqual([s["candidate_key"] for s in keeps], ["W_NEAR"])
        self.assertEqual((keeps[0]["score"], keeps[0]["tau"]), (1.0, 0.5))

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

    def test_a_reference_already_in_the_registry_is_not_requested_again(self):
        """It still counts as found -- the edge is a fact about both nodes --
        but asking OpenAlex for metadata we already hold buys nothing.
        """
        writer = DryRunWriter()
        seed_a = work("W_A", title="Seed one", refs=["W_B"])
        seed_b = work("W_B", title="Seed two")
        client = FakeClient([seed_a, seed_b], citers={})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a", "doc_b"], {"doc_a": "W_A", "doc_b": "W_B"})
        client.id_batches.clear()
        _candidates, found, _hubs, _references = snowball.gather(["W_A"])
        self.assertEqual(client.id_batches, [], "известную ссылку спросили заново")
        self.assertEqual(found["W_A"], 1)

    def test_a_reference_nobody_knows_yet_is_requested(self):
        """The control for the dedup above: the branch is a cut, not a floor."""
        writer = DryRunWriter()
        seed_a = work("W_A", title="Seed one", refs=["W_NEW"])
        client = FakeClient([seed_a, work("W_NEW", title="Unknown reference")],
                            citers={})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_A"})
        client.id_batches.clear()
        _candidates, found, _hubs, _references = snowball.gather(["W_A"])
        self.assertEqual(client.id_batches, [["W_NEW"]])
        self.assertEqual(found["W_A"], 1)

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
        self.assertTrue(all(s["node_key"] == "W_RU" for s in keeps))

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
        self.assertEqual((drops[0]["score"], drops[0]["tau"]), (-1.0, 0.0))


class ResumeTests(unittest.TestCase):
    """--resume hands the crawl the keys it fetched recently
    (citations/inputs.fresh_keys) and the frontier drops them.
    """

    def _crawl(self, skip_keys):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        citer = work("W_X", title="Near Chebyshev", refs=["W_SEED"])
        client = FakeClient([seed, citer], citers={"W_SEED": [citer], "W_X": []})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0), "Near": unit(0)}),
                            writer, tau=0.5, crawl_id="c", log=lambda *_: None,
                            skip_keys=skip_keys)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        client.cites_batches.clear()
        snowball.run(2)
        return client

    def test_a_skipped_key_never_reaches_the_client(self):
        client = self._crawl({"W_X"})
        asked = [key for batch in client.cites_batches for key in batch]
        self.assertNotIn("W_X", asked)
        self.assertNotIn("W_X", [key for batch in client.id_batches for key in batch])

    def test_without_the_skip_the_same_node_is_expanded(self):
        client = self._crawl(frozenset())
        asked = [key for batch in client.cites_batches for key in batch]
        self.assertIn("W_X", asked)


class ScoringMemoryTests(unittest.TestCase):
    """The vector, not the record, is what a level's memory is made of:
    1024 floats per candidate, thousands of candidates at depth 2, and only
    those above tau are ever written. Below-tau vectors are released as soon
    as the score is known, so peak memory follows the KEPT set.
    """

    class BatchRecordingEmbedder:
        """Vectors by title marker, remembering the size of every call.

        The seed shares the kept candidates' axis, so the centroid is that
        axis and the cosine is exactly 1.0 or 0.0.
        """

        NEAR = ("Seed", "Near")

        def __init__(self):
            self.batches: list[int] = []

        def __call__(self, texts):
            self.batches.append(len(texts))
            return [unit(0) if any(m in text for m in self.NEAR) else unit(500)
                    for text in texts]

    def _seeded(self, n_keep: int, n_drop: int):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        citers = [work(f"W_KEEP{i}", title="Near Chebyshev", refs=["W_SEED"])
                  for i in range(n_keep)]
        citers += [work(f"W_DROP{i}", title="Far unrelated", refs=["W_SEED"])
                   for i in range(n_drop)]
        client = FakeClient([seed] + citers, citers={"W_SEED": citers})
        embedder = self.BatchRecordingEmbedder()
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        embedder.batches.clear()  # the seed pass is not the level being measured
        return snowball, embedder

    def _scored(self, n_keep: int, n_drop: int):
        snowball, embedder = self._seeded(n_keep, n_drop)
        candidates, _found, _hubs, _refs = snowball.gather(["W_SEED"])
        return snowball.score(candidates), embedder

    def test_only_the_kept_candidates_still_hold_a_vector(self):
        scored, _embedder = self._scored(n_keep=10, n_drop=90)
        self.assertEqual(len(scored), 100)
        with_vector = [item for item in scored if item["vector"] is not None]
        self.assertEqual(len(with_vector), 10,
                         "вектор отброшенного кандидата остался в памяти")
        self.assertTrue(all(item["score"] >= 0.5 for item in with_vector))
        self.assertTrue(all(item["candidate_key"].startswith("W_KEEP")
                            for item in with_vector))

    def test_the_level_is_embedded_in_batches_not_in_one_call(self):
        _scored, embedder = self._scored(n_keep=10, n_drop=90)
        self.assertEqual(sum(embedder.batches), 100)
        self.assertTrue(all(size <= EMBED_BATCH for size in embedder.batches),
                        embedder.batches)
        self.assertEqual(len(embedder.batches), 7)  # ceil(100 / 16)

    def test_no_candidate_carries_a_reference_list(self):
        """The vector is not the only bulky thing a level holds: a citer's
        referenced_works is most of its bytes, and >51000 citers were
        measured at one depth-2 level. The list travels beside the
        candidates and is freed for everything the filter drops.
        """
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        near = work("W_NEAR", title="Near Chebyshev", refs=["W_SEED", "W_OTHER"])
        far = work("W_FAR", title="Far unrelated", refs=["W_SEED"])
        client = FakeClient([seed, near, far], citers={"W_SEED": [near, far]})
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Far": unit(500)})
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        candidates, _found, _hubs, references = snowball.gather(["W_SEED"])
        self.assertTrue(all("referenced_works" not in record
                            for record, _relation, _source in candidates), candidates)
        self.assertEqual(references["W_NEAR"], ("W_SEED", "W_OTHER"))

    def test_the_edges_of_a_level_are_still_derived_from_those_lists(self):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        near = work("W_NEAR", title="Near Chebyshev", refs=["W_SEED"])
        client = FakeClient([seed, near], citers={"W_SEED": [near]})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0), "Near": unit(0)}),
                            writer, tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        snowball.expand(["W_SEED"], 1)
        self.assertIn(("W_NEAR", "W_SEED", "cites", "W_SEED"), writer.edges_seen)
        self.assertEqual(snowball.registry.nodes["W_NEAR"].referenced_works, {"W_SEED"},
                         "оставленный узел потерял свои ссылки — depth+1 их не увидит")

    def test_a_kept_candidate_still_reaches_its_node_with_its_vector(self):
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev")
        near = work("W_NEAR", title="Near Chebyshev", refs=["W_SEED"])
        client = FakeClient([seed, near], citers={"W_SEED": [near]})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0), "Near": unit(0)}),
                            writer, tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        snowball.expand(["W_SEED"], 1)
        self.assertEqual(snowball.registry.nodes["W_NEAR"].embedding, unit(0))

    def test_a_record_is_absorbed_only_once_it_has_passed_tau(self):
        """The filter reads three fields; absorbing a record builds the
        namespaced id set, the author list, the DOI and a copy of the record
        itself. Doing that for every candidate -- and then again in
        registry.add() for the ones kept -- pays the whole per-candidate
        cost twice at a level where nine candidates in ten are dropped.
        """
        snowball, _embedder = self._seeded(n_keep=10, n_drop=90)
        with mock.patch.object(Node, "absorb", autospec=True,
                               side_effect=Node.absorb) as absorbed:
            kept = snowball.expand(["W_SEED"], 1)
        self.assertEqual(len(kept), 10)
        self.assertEqual(absorbed.call_count, 10,
                         "запись поглощена не только за прошедших τ")
    """seeding.py takes a registry, a client and callables -- never a
    Snowball -- so the seed set can be established and asserted here with
    neither a crawl object nor a writer in sight.
    """

    def _registry_and_client(self):
        records = [work("W_SEED_A", title="Seed Chebyshev")]
        return WorkRegistry(), FakeClient(records)

    def test_collect_seeds_registers_the_known_and_journals_the_rest(self):
        registry, client = self._registry_and_client()
        steps, n_matched = seeding.collect_seeds(
            registry, client, "c", ["doc_a", "doc_b"], {"doc_a": "W_SEED_A"})
        self.assertEqual(sorted(registry.nodes), ["W_SEED_A"])
        self.assertEqual(n_matched, 1)
        self.assertEqual(sorted(s["action"] for s in steps), ["seed", "seed-missing"])

    def test_collect_seeds_writes_nothing_itself(self):
        """The journal rows come back for the caller to write -- crawl.py
        writes them before the centroid, because a run with no seeds raises
        there and the rows have to survive it.
        """
        registry, client = self._registry_and_client()
        steps, _ = seeding.collect_seeds(registry, client, "c", ["doc_b"], {})
        self.assertEqual([s["action"] for s in steps], ["seed-missing"])

    def test_rank_seeds_returns_the_state_instead_of_assigning_it(self):
        registry, client = self._registry_and_client()
        _steps, n_matched = seeding.collect_seeds(
            registry, client, "c", ["doc_a", "doc_b"], {"doc_a": "W_SEED_A"})

        def embed_nodes(nodes):
            vectors = [unit(0) for _ in nodes]
            for node, vector in zip(nodes, vectors):
                node.embedding = vector
            return vectors

        keys, centre, per_depth_row = seeding.rank_seeds(registry, embed_nodes, 2, n_matched)
        self.assertEqual(keys, ["W_SEED_A"])
        self.assertEqual(centre, unit(0))
        self.assertEqual(per_depth_row, {"seeds": 1, "seed_missing": 1})
        self.assertAlmostEqual(registry.nodes["W_SEED_A"].score, 1.0)

    def test_no_seed_at_all_has_no_centre_to_rank_against(self):
        with self.assertRaises(ValueError):
            seeding.rank_seeds(WorkRegistry(), lambda nodes: [], 1, 0)


class WriterConformanceTests(unittest.TestCase):
    """The two writers are one seam: crawl.py swaps them by --dry-run, and
    the dry run is what decides whether a real, quota-spending crawl is
    worth launching. So both are driven through the SAME call sequence and
    their answers compared, rather than each being trusted separately --
    DryRunWriter.works() used to add the accumulated total on every call,
    which nothing noticed until the second batch.
    """

    @staticmethod
    def _nodes(*keys):
        made = []
        for key in keys:
            node = Node(key=key, kind="external-skeleton", depth=1)
            node.absorb(work(key, title=f"Title {key}"))
            made.append(node)
        return made

    @staticmethod
    def _steps(*keys):
        return [journal.fetch("c", 1, key, 1, 1) for key in keys]

    def _drive(self, writer):
        """One fixed sequence; every per-call answer plus the running counts
        after each write."""
        trace = []
        for nodes in (self._nodes("W1", "W2"), self._nodes("W3"), []):
            trace.append(("works", writer.works(nodes), dict(writer.counts)))
        for edges in ([("W1", "W2", "cites", "W1")], [("W3", "W1", "cites", "W3")], []):
            trace.append(("edges", writer.edges(edges), dict(writer.counts)))
        for steps in (self._steps("W1", "W2"), self._steps("W3"), []):
            trace.append(("journal", writer.journal(steps), dict(writer.counts)))
        return trace

    def test_both_implementations_satisfy_the_writer_protocol(self):
        self.assertIsInstance(DryRunWriter(), Writer)
        self.assertIsInstance(PostgresWriter({}), Writer)

    def test_the_same_call_sequence_produces_the_same_counts(self):
        """With nothing in the database to refuse a row, the live writer
        reports what the dry run predicts -- the counts diverge only where
        the upsert actually refuses (an edge already known, a promote key
        with no work row), which is the live half of this pair.
        """
        def accept_everything(env, table_columns, rows, **kwargs):
            written = sum(1 for _ in rows)
            return CopyResult(written, f"{written}\n" if kwargs.get("epilogue") else "")

        with mock.patch.object(store, "copy_csv_rows", side_effect=accept_everything):
            live = self._drive(PostgresWriter({}))
        self.assertEqual(self._drive(DryRunWriter()), live)

    def test_a_write_is_one_script_over_a_temp_staging_table(self):
        """Staging DDL, the \\copy and the upsert in one psql invocation:
        a shared, globally named staging table between three invocations was
        observable -- and droppable -- by any writer running at the time.
        """
        calls = []
        with mock.patch.object(store, "copy_csv_rows",
                                side_effect=lambda env, target, rows, **kw: (
                                    calls.append((target, list(rows), kw)),
                                    CopyResult(2, "2\n"))[1]):
            PostgresWriter({}).works(self._nodes("W1", "W2"))
        self.assertEqual(len(calls), 1)
        target, rows, kwargs = calls[0]
        self.assertTrue(target.startswith("stage_work ("), target)
        self.assertEqual(len(rows), 2)
        self.assertIn("CREATE TEMP TABLE stage_work", kwargs["preamble"])
        self.assertIn("ON COMMIT DROP", kwargs["preamble"])
        self.assertIn("INSERT INTO citation.work", kwargs["epilogue"])

    def test_the_number_reported_is_the_one_the_database_counted(self):
        """Not the number submitted: the cites upsert drops self-edges and
        skips edges the graph already has.
        """
        with mock.patch.object(store, "copy_csv_rows",
                                return_value=CopyResult(3, "1\n")):
            writer = PostgresWriter({})
            accepted = writer.edges([("W1", "W2", "cites", "W1"),
                                     ("W1", "W3", "cites", "W1"),
                                     ("W1", "W4", "cites", "W1")])
        self.assertEqual(accepted, 1)
        self.assertEqual(writer.counts["cites"], 1)

    def test_no_staging_relation_is_named_in_the_shared_schema(self):
        text = pathlib.Path(store_sql.__file__).read_text(encoding="utf-8")
        self.assertNotIn("citation.stage_", text,
                         "staging is TEMP and session-private")

    def test_counts_are_rows_accepted_by_this_call_not_the_running_total(self):
        writer = DryRunWriter()
        self.assertEqual(writer.works(self._nodes("W1", "W2")), 2)
        self.assertEqual(writer.works(self._nodes("W3")), 1)
        self.assertEqual(writer.counts["work"], 3)


if __name__ == "__main__":
    unittest.main()
