"""The BFS itself: what the snowball keeps, drops, journals and writes.

Snowball is driven with a fake OpenAlex and a planned embedder, so keep/drop,
the journal shape and the edge derivation are asserted exactly, and nothing
here needs a database. The one property no stub can carry -- what real SQL
does on conflict -- lives next door in test_citations_crawl_live.py.
"""
from __future__ import annotations

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
from citation_vocab import Relation
from citations import seeding
from citations.crawl import Snowball
from citations.frontier import EMBED_BATCH
from citations import candidates as candidates_mod
from citations import registry as registry_mod
from citations.registry import Node, WorkRegistry
from citations.dry_store import DryRunWriter


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

    def test_a_candidate_of_two_frontier_nodes_is_counted_by_both(self):
        """n_found and n_kept over ONE population, and every node credited.

        The citer below cites both seeds, so it is one candidate that two
        frontier nodes found. Crediting the alphabetically first hit alone
        left the other node's fetch row reading 1 found / 0 kept about a
        candidate it discovered and the filter accepted -- and n_found
        counted (node, reference) incidences while n_kept counted
        candidates, so the two were not comparable in the first place.
        """
        writer = DryRunWriter()
        client = FakeClient(
            [work("W_A", title="Seed one"), work("W_B", title="Seed two")],
            citers={"W_A": [work("W_C", title="Seed citer", refs=["W_A", "W_B"])]},
        )
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a", "doc_b"], {"doc_a": "W_A", "doc_b": "W_B"})
        snowball.expand(["W_A", "W_B"], 1)
        fetch = {s["frontier_key"]: (s["n_found"], s["n_kept"])
                 for s in writer.steps_seen if s["action"] == "fetch"}
        self.assertEqual(fetch, {"W_A": (1, 1), "W_B": (1, 1)})
        keeps = [s for s in writer.steps_seen if s["action"] == "keep"]
        self.assertEqual([s["candidate_key"] for s in keeps], ["W_C"],
                         "один кандидат — одна строка журнала")
        self.assertEqual(keeps[0]["frontier_key"], "W_A",
                         "в одноимённой колонке — наименьший ключ множества")

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
        """The edge is still a fact about both nodes, but asking OpenAlex
        for metadata we already hold buys nothing -- and a work already in
        the graph is no CANDIDATE, so it is not among the level's counted
        ones either.
        """
        writer = DryRunWriter()
        seed_a = work("W_A", title="Seed one", refs=["W_B"])
        seed_b = work("W_B", title="Seed two")
        client = FakeClient([seed_a, seed_b], citers={})
        snowball = Snowball(client, PlannedEmbedder({"Seed": unit(0)}), writer,
                            tau=0.5, crawl_id="c", log=lambda *_: None)
        snowball.seed(["doc_a", "doc_b"], {"doc_a": "W_A", "doc_b": "W_B"})
        client.id_batches.clear()
        candidates, _hubs, _references = snowball.gather(["W_A"])
        self.assertEqual(client.id_batches, [], "известную ссылку спросили заново")
        self.assertEqual(candidates, [])

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
        candidates, _hubs, _references = snowball.gather(["W_A"])
        self.assertEqual(client.id_batches, [["W_NEW"]])
        self.assertEqual([hits for _record, _relation, hits in candidates],
                         [frozenset({"W_A"})])

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

    def test_a_level_of_leaves_only_ends_the_crawl_and_says_so(self):
        """The complement of the rule above: a node reached by
        `relation='referenced'` is written and never opened, so a level
        whose survivors are all references leaves the next frontier empty.
        The crawl stops there rather than asking OpenAlex for the citers of
        nothing -- and it says why, because a run that ends a level early
        looks exactly like a run that finished otherwise.
        """
        said = []
        writer = DryRunWriter()
        seed = work("W_SEED", title="Seed Chebyshev", refs=["W_REF"])
        reference = work("W_REF", title="Near Chebyshev")
        client = FakeClient([seed, reference], citers={})
        embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0)})
        snowball = Snowball(client, embedder, writer, tau=0.5, crawl_id="c",
                            log=said.append)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
        per_depth = snowball.run(2)
        self.assertEqual(snowball.registry.nodes["W_REF"].relation, Relation.REFERENCED)
        # 0 is the seed level; depth 2 is the one that never ran.
        self.assertEqual(sorted(per_depth), [0, 1])
        self.assertEqual(client.cites_batches, [["W_SEED"]])
        self.assertTrue(any("фронтир пуст" in line for line in said), said)

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
        candidates, _hubs, _refs = snowball.gather(["W_SEED"])
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
        candidates, _hubs, references = snowball.gather(["W_SEED"])
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

    def test_a_candidates_identifier_set_is_derived_once(self):
        """The same cost one layer up from absorb(): every candidate's
        namespaced id set (a DOI normalised, short_id per identifier) was
        built by score()'s registry.find(), again by add()'s own find(), and
        a third time by absorb(). It is a pure function of the record, so
        the resolution score() already computed travels with the candidate
        and add() re-runs only the INDEX lookup, which is the part that
        changes between the two (a twin union happens on add()).
        """
        snowball, _embedder = self._seeded(n_keep=10, n_drop=90)
        derived = mock.Mock(side_effect=registry_mod.record_ids)
        # Counted at BOTH binding sites: the derivation is one function, and
        # which module happens to name it is not what is being measured.
        with mock.patch.object(registry_mod, "record_ids", derived), \
             mock.patch.object(candidates_mod, "record_ids", derived):
            kept = snowball.expand(["W_SEED"], 1)
        self.assertEqual(len(kept), 10)
        self.assertEqual(derived.call_count, 100,
                         "набор идентификаторов кандидата выведен не один раз")
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


if __name__ == "__main__":
    unittest.main()
