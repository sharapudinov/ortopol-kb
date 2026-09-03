"""The embedder's disk memo: what it saves, and what it refuses to save.

The pipeline the docs prescribe is `--calibrate` and then a crawl at the
measured tau, over the same depth-1 candidates. A calibration writes no
citation.work row (its writer is a DryRunWriter by construction), so the
store read finds nothing of what it just embedded, and every bge-m3
inference of that level used to be bought a second time. These tests are
about that pair: the second pass over the same candidates asks ollama
nothing.

The mode is the cache OBJECT's, never a flag: under --dry-run the memo is
handed a ReadOnlyCache and must leave the tree exactly as it found it --
not even a directory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import _pathfix  # noqa: F401
from citations import frontier
from citations.http_cache import DiskCache, ReadOnlyCache, cache_for
from citations.vector_cache import VectorMemo, entry_name, memoizing_embedder

MODEL = "bge-m3"


@dataclass
class Holder:
    key: str
    title: str | None
    abstract: str | None


class CountingEmbedder:
    """The ollama seam, counting what it is actually asked to compute."""

    def __init__(self):
        self.calls = 0
        self.texts: list[str] = []

    def __call__(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        return [[float(len(text)), 0.5] for text in texts]


def _holders(n: int) -> list[Holder]:
    return [Holder(f"W{i}", f"Title {i}", f"Abstract {i}") for i in range(n)]


def _level(embed, memo, holders):
    """One pass of the filter over `holders` with nothing in the store --
    the shape both --calibrate and the crawl after it have."""
    return list(frontier.vectors_for(memoizing_embedder(embed, memo),
                                     lambda keys: {}, holders))


class EntryNameTests(unittest.TestCase):
    def test_the_same_text_under_two_models_is_two_entries(self):
        self.assertNotEqual(entry_name("bge-m3", "x"), entry_name("other", "x"))

    def test_a_changed_text_is_a_different_entry(self):
        self.assertNotEqual(entry_name(MODEL, "Title A"), entry_name(MODEL, "Title B"))

    def test_the_name_is_one_safe_path_component(self):
        name = entry_name("nomic-embed-text:v1.5", "x")
        self.assertNotIn("/", name)
        self.assertNotIn(":", name)


class SecondPassCostsNothingTests(unittest.TestCase):
    def test_the_crawl_after_a_calibration_asks_ollama_nothing(self):
        holders = _holders(5)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "embeddings"
            calibrate = CountingEmbedder()
            first = _level(calibrate, VectorMemo(DiskCache(directory), MODEL), holders)
            crawl = CountingEmbedder()
            second = _level(crawl, VectorMemo(DiskCache(directory), MODEL), holders)
        self.assertEqual(len(calibrate.texts), 5)
        self.assertEqual(crawl.calls, 0, "второй проход снова платил за те же векторы")
        self.assertEqual([v for _h, v in first], [v for _h, v in second])

    def test_a_candidate_whose_title_changed_is_embedded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "embeddings"
            _level(CountingEmbedder(), VectorMemo(DiskCache(directory), MODEL),
                   [Holder("W1", "Old title", None)])
            again = CountingEmbedder()
            _level(again, VectorMemo(DiskCache(directory), MODEL),
                   [Holder("W1", "New title", None)])
        self.assertEqual(len(again.texts), 1)

    def test_a_vector_from_another_model_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "embeddings"
            _level(CountingEmbedder(), VectorMemo(DiskCache(directory), MODEL), _holders(2))
            other = CountingEmbedder()
            _level(other, VectorMemo(DiskCache(directory), "other-model"), _holders(2))
        self.assertEqual(len(other.texts), 2)


class MemoContractTests(unittest.TestCase):
    def test_vectors_come_back_in_input_order_whoever_answered(self):
        with tempfile.TemporaryDirectory() as tmp:
            memo = VectorMemo(DiskCache(Path(tmp)), MODEL)
            memo.put("second", [2.0, 2.0])
            embed = CountingEmbedder()
            out = memoizing_embedder(embed, memo)(["first", "second", "third"])
        self.assertEqual(out[1], [2.0, 2.0])
        self.assertEqual(embed.texts, ["first", "third"])
        self.assertEqual(out[0], [5.0, 0.5])

    def test_a_repeated_text_in_one_batch_is_asked_for_once(self):
        """At a depth-2 level the same candidate arrives through several
        frontier nodes."""
        with tempfile.TemporaryDirectory() as tmp:
            embed = CountingEmbedder()
            out = memoizing_embedder(embed, VectorMemo(DiskCache(Path(tmp)), MODEL))(
                ["same", "other", "same"])
        self.assertEqual(embed.texts, ["same", "other"])
        self.assertEqual(out[0], out[2])

    def test_an_unreadable_entry_is_a_miss_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / entry_name(MODEL, "text")).write_text("{oops", encoding="utf-8")
            memo = VectorMemo(DiskCache(directory), MODEL)
            self.assertIsNone(memo.get("text"))
            (directory / entry_name(MODEL, "text")).write_text(
                json.dumps({"vector": [1, 2]}), encoding="utf-8")
            self.assertIsNone(memo.get("text"))

    def test_no_cache_at_all_is_no_memo(self):
        memo = VectorMemo(None, MODEL)
        memo.put("text", [1.0])
        self.assertIsNone(memo.get("text"))


class DryRunLeavesTheTreeAloneTests(unittest.TestCase):
    """DRY_RUN_WRITES_NOTHING's rule for this channel: the memo is handed a
    cache object, and a read-only one writes nothing and creates nothing.
    """

    def test_a_read_only_memo_writes_no_entry_and_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "embeddings"
            embed = CountingEmbedder()
            _level(embed, VectorMemo(ReadOnlyCache(directory), MODEL), _holders(3))
            self.assertFalse(directory.exists())
            again = CountingEmbedder()
            _level(again, VectorMemo(ReadOnlyCache(directory), MODEL), _holders(3))
        self.assertEqual(len(again.texts), 3)

    def test_a_read_only_memo_still_serves_what_is_already_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "embeddings"
            _level(CountingEmbedder(), VectorMemo(DiskCache(directory), MODEL), _holders(2))
            embed = CountingEmbedder()
            _level(embed, VectorMemo(ReadOnlyCache(directory), MODEL), _holders(2))
        self.assertEqual(embed.calls, 0)

    def test_the_run_picks_the_object_the_way_the_other_caches_are_picked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsInstance(cache_for(Path(tmp), read_only=True), ReadOnlyCache)
            self.assertIsInstance(cache_for(Path(tmp), read_only=False), DiskCache)


if __name__ == "__main__":
    unittest.main()
