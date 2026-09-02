"""What a dry run holds at once, counted rather than reasoned about.

A vector is 1024 floats; a depth-2 level is thousands of candidates
(~4262 distinct references measured at tau=0.50) of which the filter keeps
a fraction. crawl.scores_of() has always claimed peak memory is a function
of the KEPT set -- this counts the vectors actually alive at the moment the
embedder is called and holds the claim to it.

The writer is the other half of the same question, and lives here for the
same reason: --dry-run is the cheap rehearsal a real crawl is authorised
against, so what it retains per level is a number to hold it to.

Separate file from test_citations_crawl.py for size (kb/CLAUDE.md
FILE_SIZE), and because the technique is its own thing: the vectors are
weak-referenced, so "alive" is measured, not inferred from call counts.
"""
from __future__ import annotations

import gc
import unittest
import weakref

import _pathfix  # noqa: F401
from citations import frontier
from citations.crawl import Snowball
from citations.frontier import EMBED_BATCH, KEY_BATCH
from citations.registry import scoring_fields
from citations.store import DryRunWriter


class _Vector(list):
    """A vector that can be weak-referenced, and is otherwise a list."""

    __slots__ = ("__weakref__",)


class _CountingEmbedder:
    """Returns weak-referenced vectors and counts how many are still alive
    each time it is asked for more -- a list of weakref, not a WeakSet: a
    list is unhashable, and the vectors ARE lists. A text carrying
    `keep_marker` scores 1.0 against the centroid below, everything else
    0.0.
    """

    def __init__(self, keep_marker: str):
        self.keep_marker = keep_marker
        self.issued: list[weakref.ref] = []
        self.alive_at_call: list[int] = []
        self.batches: list[int] = []

    def alive(self) -> int:
        gc.collect()
        return sum(1 for ref in self.issued if ref() is not None)

    def __call__(self, texts):
        self.alive_at_call.append(self.alive())
        self.batches.append(len(texts))
        out = []
        for text in texts:
            vector = _Vector([1.0 if self.keep_marker in text else 0.0] + [0.0] * 1023)
            self.issued.append(weakref.ref(vector))
            out.append(vector)
        return out


class PeakIsTheKeptSetTests(unittest.TestCase):
    N_KEEP = 10
    N_DROP = 90

    def _holders(self):
        keep = [scoring_fields({"id": f"W_KEEP{i}", "title": "Near Chebyshev"})
                for i in range(self.N_KEEP)]
        drop = [scoring_fields({"id": f"W_DROP{i}", "title": "Far unrelated"})
                for i in range(self.N_DROP)]
        # Interleaved, so the kept ones are not all in the first chunk and
        # the count at the last embed call is a genuine running total.
        out = []
        for i in range(max(len(keep), len(drop))):
            if i < len(keep):
                out.append(keep[i])
            out.extend(drop[i * 9:(i + 1) * 9])
        return out

    def _scored(self):
        embedder = _CountingEmbedder("Near")
        snowball = Snowball(object(), embedder, DryRunWriter(), tau=0.5,
                            crawl_id="c", log=lambda *_: None)
        snowball.centroid = [1.0] + [0.0] * 1023
        scored = snowball.scores_of(self._holders())
        return scored, embedder

    def test_the_filter_keeps_exactly_the_near_candidates(self):
        scored, _embedder = self._scored()
        self.assertEqual(len(scored), self.N_KEEP + self.N_DROP)
        with_vector = [k for k, (_s, v) in scored.items() if v is not None]
        self.assertEqual(len(with_vector), self.N_KEEP)

    def test_no_more_than_one_chunk_plus_the_kept_set_is_alive_at_once(self):
        _scored, embedder = self._scored()
        ceiling = EMBED_BATCH + self.N_KEEP
        self.assertLessEqual(
            max(embedder.alive_at_call), ceiling,
            f"живых векторов {embedder.alive_at_call}, потолок {ceiling}")
        # And the level really was embedded in chunks, so the ceiling is
        # not met by never asking for more than one batch of work.
        self.assertEqual(sum(embedder.batches), self.N_KEEP + self.N_DROP)
        self.assertGreater(len(embedder.batches), 1)

    def test_the_whole_set_would_not_fit_under_that_ceiling(self):
        """The guard is only worth anything if the old shape failed it."""
        self.assertGreater(self.N_KEEP + self.N_DROP, EMBED_BATCH + self.N_KEEP)

    def test_the_stored_vectors_alive_are_one_read_block_plus_the_kept_set(self):
        """The other half of the peak, on the read side.

        The store is read KEY_BATCH keys at a time -- the psql round trip is
        what that size is for -- so a block's stored vectors are alive while
        the block is scored and released when the next block is read. What
        crosses a block boundary is only what the consumer kept.
        """
        n_keep = 5
        holders = [scoring_fields({"id": f"W_KEEP{i}", "title": "Near Chebyshev"})
                   for i in range(n_keep)]
        holders += [scoring_fields({"id": f"W{i}", "title": "Far unrelated"})
                    for i in range(KEY_BATCH * 2)]
        issued: list[weakref.ref] = []

        def known(keys):
            out = {}
            for key in keys:
                vector = _Vector([0.0] * 1024)
                issued.append(weakref.ref(vector))
                out[key] = vector
            return out

        kept, alive_seen = [], []
        for index, (holder, vector) in enumerate(
                frontier.vectors_for(_CountingEmbedder("Near"), known, holders)):
            if holder.key.startswith("W_KEEP"):
                kept.append(vector)
            if index % 25 == 0:
                gc.collect()
                alive_seen.append(sum(1 for ref in issued if ref() is not None))
        ceiling = KEY_BATCH + len(kept)
        self.assertEqual(len(kept), n_keep)
        self.assertLessEqual(max(alive_seen), ceiling,
                             f"живых хранимых векторов {alive_seen}, потолок {ceiling}")
        self.assertGreater(len(holders), ceiling,
                           "потолок не ниже уровня — проверка ничего не держит")


class DryRunSampleIsBoundedTests(unittest.TestCase):
    """PostgresWriter streams a level through copy_csv_rows and frees it,
    so its peak follows ONE level. The dry run kept every row of every
    level instead, and a depth-2 journal is ~100k rows (pg_copy.py) -- on
    the machine about to spend a real quota window. It keeps a sample now;
    the quantity was always in counts.
    """

    ROWS = 10_000

    def _feed(self, writer) -> None:
        writer.works([f"W_{i}" for i in range(self.ROWS)])
        writer.edges([(f"W_{i}", "W_SEED", "cites", "W_SEED") for i in range(self.ROWS)])
        writer.journal([{"action": "keep", "depth": 2} for _ in range(self.ROWS)])
        writer.promote([{"key": f"W_{i}"} for i in range(self.ROWS)])

    def test_ten_thousand_rows_of_each_kind_leave_a_bounded_sample(self):
        writer = DryRunWriter()
        self._feed(writer)
        for kind in ("works_seen", "edges_seen", "steps_seen", "promoted_seen"):
            with self.subTest(kind=kind):
                self.assertLessEqual(len(getattr(writer, kind)),
                                     DryRunWriter.SAMPLE_LIMIT)

    def test_the_counts_still_speak_for_every_row(self):
        """The sample is the specimen; the estimate a crawl is authorised
        against is the count, and it is of everything submitted.
        """
        writer = DryRunWriter()
        self._feed(writer)
        self.assertEqual(writer.counts, {"work": self.ROWS, "cites": self.ROWS,
                                         "step": self.ROWS, "twin": self.ROWS})

    def test_what_is_kept_is_the_head_of_the_batch(self):
        writer = DryRunWriter()
        self._feed(writer)
        self.assertEqual(writer.works_seen[0], "W_0")
        self.assertEqual(len(writer.works_seen), DryRunWriter.SAMPLE_LIMIT)

    def test_each_call_still_reports_what_it_accepted(self):
        writer = DryRunWriter()
        self.assertEqual(writer.works([f"W_{i}" for i in range(self.ROWS)]), self.ROWS)


if __name__ == "__main__":
    unittest.main()
