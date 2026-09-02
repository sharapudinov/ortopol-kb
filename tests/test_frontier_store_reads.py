"""The store-read seam of the scoring pass: what is bought and what is read.

frontier.vectors_for() answers with two batch sizes at once -- KEY_BATCH
keys per psql round trip, EMBED_BATCH texts per ollama request -- because
the two seams are charged differently. Split from test_citations_crawl.py
for size (kb/CLAUDE.md FILE_SIZE); that file is about what the BFS keeps,
drops and journals, this one about what a level pays for.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, build_snowball, unit, work
from citations import frontier
from citations.crawl import Snowball
from citations.frontier import EMBED_BATCH, KEY_BATCH
from citations.registry import scoring_fields
from citations.store import DryRunWriter


class StoredVectorsAreReusedTests(unittest.TestCase):
    """A vector already in citation.work is read, not bought again.

    --calibrate embeds every depth-1 candidate and writes no node, so the
    crawl that follows meets exactly the same candidate set; a re-crawl
    without --resume meets every node it ever wrote. Both used to pay
    ollama a second time for vectors already stored.
    """

    def _holders(self, n):
        return [scoring_fields(work(f"W{i}", title=f"Candidate {i}")) for i in range(n)]

    def test_only_the_unknown_candidates_reach_the_embedder(self):
        holders = self._holders(10)
        stored = {f"W{i}": unit(i) for i in (1, 3, 5, 7)}
        asked = []

        def known(keys):
            asked.append(list(keys))
            return stored

        embedder = PlannedEmbedder({})
        list(frontier.vectors_for(embedder, known, holders))
        self.assertEqual(len(embedder.texts), 6)
        self.assertEqual(sorted(t.split()[-1] for t in embedder.texts),
                         ["0", "2", "4", "6", "8", "9"])
        # One read per CHUNK, naming that chunk's keys -- not one per key,
        # and not one dict over the whole level.
        self.assertEqual(asked, [[f"W{i}" for i in range(10)]])

    def test_the_store_is_read_a_block_at_a_time(self):
        """The read side is chunked too: a dict over the whole level holds a
        1024-float vector for every already-known candidate, which at ~4262
        candidates is the peak this pass exists to avoid.
        """
        holders = self._holders(KEY_BATCH * 3 + 5)
        asked = []

        def known(keys):
            asked.append(list(keys))
            return {}

        list(frontier.vectors_for(PlannedEmbedder({}), known, holders))
        self.assertEqual(len(asked), 4)
        self.assertTrue(all(len(chunk) <= KEY_BATCH for chunk in asked), asked)
        self.assertEqual([key for chunk in asked for key in chunk],
                         [h.key for h in holders])

    def test_the_read_takes_key_batch_and_the_embedder_embed_batch(self):
        """The two sizes are independent, and neither collapses onto the
        other: reading at the embed size turned one level into an order of
        magnitude more psql spawns than the IN-list the read batches by.
        """
        holders = self._holders(KEY_BATCH + EMBED_BATCH)
        asked, embedded = [], []
        planned = PlannedEmbedder({})

        def known(keys):
            asked.append(list(keys))
            return {}

        def embed(texts):
            embedded.append(len(texts))
            return planned(texts)

        list(frontier.vectors_for(embed, known, holders))
        self.assertEqual([len(chunk) for chunk in asked], [KEY_BATCH, EMBED_BATCH])
        self.assertEqual(
            embedded,
            [EMBED_BATCH] * (KEY_BATCH // EMBED_BATCH)
            + ([KEY_BATCH % EMBED_BATCH] if KEY_BATCH % EMBED_BATCH else [])
            + [EMBED_BATCH])

    def test_a_read_block_holds_no_more_than_its_own_keys(self):
        """What the store hands back is alive for one block, not for the
        level: that dict is what bounds the peak (KEY_BATCH stored vectors)
        once the embedded ones are released per sub-batch.
        """
        holders = self._holders(KEY_BATCH * 2)
        widest = 0

        def known(keys):
            nonlocal widest
            widest = max(widest, len(keys))
            return {key: unit(0) for key in keys}

        list(frontier.vectors_for(PlannedEmbedder({}), known, holders))
        self.assertEqual(widest, KEY_BATCH)

    def test_the_stored_vector_is_yielded_verbatim_and_in_order(self):
        holders = self._holders(10)
        stored = {f"W{i}": unit(i) for i in (1, 3, 5, 7)}
        pairs = list(frontier.vectors_for(PlannedEmbedder({}),
                                          lambda keys: stored, holders))
        self.assertEqual([h.key for h, _v in pairs], [h.key for h in holders])
        for i in (1, 3, 5, 7):
            self.assertEqual(pairs[i][1], unit(i))

    def test_nothing_is_embedded_when_everything_is_known(self):
        holders = self._holders(4)
        stored = {h.key: unit(0) for h in holders}
        embedder = PlannedEmbedder({})
        list(frontier.vectors_for(embedder, lambda keys: stored, holders))
        self.assertEqual(embedder.calls, 0)

    def test_seeds_already_embedded_are_not_embedded_again(self):
        """The 56 seeds are the same works on every run, and their vectors
        are in citation.work from the first crawl onward.
        """
        writer = DryRunWriter()
        embedder = PlannedEmbedder({"Seed": unit(0)})
        client = FakeClient([work("W_SEED_A", title="Seed Chebyshev")])
        snowball = Snowball(client, embedder, writer, tau=0.0, crawl_id="c",
                            log=lambda *_: None,
                            known_vectors=lambda keys: {"W_SEED_A": unit(0)})
        snowball.seed(["doc_a"], {"doc_a": "W_SEED_A"})
        self.assertEqual(embedder.texts, [])
        self.assertEqual(snowball.registry.nodes["W_SEED_A"].embedding, unit(0))

    def test_a_snowball_given_no_reader_embeds_everything(self):
        """The default seam knows nothing, which is what a unit test with no
        database sees -- and what makes the reuse an addition, not a change
        of meaning.
        """
        writer = DryRunWriter()
        embedder = PlannedEmbedder({"Seed": unit(0)})
        _client, snowball, seeds = build_snowball(writer, tau=0.0, embedder=embedder)
        snowball.seed(["doc_a"], seeds)
        self.assertEqual(len(embedder.texts), 1)

if __name__ == "__main__":
    unittest.main()
