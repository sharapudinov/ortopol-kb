"""The response cache as a seam: one contract, two independent objects.

Split out of test_citations_http.py (kb/CLAUDE.md FILE_SIZE) because this is
not about a client's failure branches at all: it is about the third channel
DRY_RUN_WRITES_NOTHING names, and about the property that makes the promise
structural -- ReadOnlyCache does not INHERIT "writes nothing" from the
writer, it simply has no write in it, the way store.DryRunWriter has none.

The conformance block runs the SAME assertions against both implementations:
a contract stated once and checked twice is what stops the two from drifting
into two behaviours, which is exactly what a subclass hid.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
from citations import hub_cache, http_cache, openalex_client


class CacheProtocolTests(unittest.TestCase):
    """Both implementations answer to the Protocol, and neither is the
    other's subclass: a write-capable member added to DiskCache tomorrow is
    absent from ReadOnlyCache rather than inherited live.
    """

    def test_both_implementations_satisfy_the_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cache in (http_cache.DiskCache(Path(tmp)),
                          http_cache.ReadOnlyCache(Path(tmp))):
                self.assertIsInstance(cache, http_cache.Cache)

    def test_neither_implementation_inherits_the_other(self):
        self.assertNotIsInstance(http_cache.ReadOnlyCache(Path("/nonexistent")),
                                 http_cache.DiskCache)
        self.assertEqual(http_cache.DiskCache.__bases__, (object,))
        self.assertEqual(http_cache.ReadOnlyCache.__bases__, (object,))

    def test_the_factory_answers_the_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsInstance(http_cache.cache_for(Path(tmp)), http_cache.DiskCache)
            self.assertIsInstance(http_cache.cache_for(Path(tmp), read_only=True),
                                  http_cache.ReadOnlyCache)
            self.assertIsNone(http_cache.cache_for(None))


class CacheConformanceTests(unittest.TestCase):
    """What every implementation must do the same way. Written once, run
    against both: the READ half is the whole reason the seam exists (three
    clients had re-spelled it), so both objects have to serve it alike.
    """

    def _both(self, tmp: str):
        return (http_cache.DiskCache(Path(tmp)), http_cache.ReadOnlyCache(Path(tmp)))

    def test_a_hit_is_served_and_counted_by_the_cache_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.json").write_text('{"x": 1}', encoding="utf-8")
            for cache in self._both(tmp):
                self.assertEqual(cache.read("a.json"), '{"x": 1}')
                self.assertIsNone(cache.read("missing.json"))
                self.assertEqual(cache.hits, 1, type(cache).__name__)

    def test_a_body_at_or_below_the_floor_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "page.html").write_text("x" * 100, encoding="utf-8")
            for cache in self._both(tmp):
                self.assertIsNone(cache.read("page.html", floor=100))
                self.assertEqual(cache.read("page.html", floor=99), "x" * 100)
                self.assertEqual(cache.hits, 1, type(cache).__name__)

    def test_a_limited_read_stops_at_the_head(self):
        """What a multi-megabyte page costs to classify: its first field,
        not its body. hub_cache's prefilter reads through the seam, so the
        seam is what has to offer the head.
        """
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "big.json").write_text("head" + "x" * 10000, encoding="utf-8")
            for cache in self._both(tmp):
                self.assertEqual(cache.read("big.json", limit=4), "head")

    def test_names_lists_the_cached_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("b.json", "a.json", "a.meta.json"):
                (Path(tmp) / name).write_text("{}", encoding="utf-8")
            for cache in self._both(tmp):
                self.assertEqual(sorted(cache.names()),
                                 ["a.json", "a.meta.json", "b.json"])

    def test_a_cache_directory_that_does_not_exist_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "never-written"
            cache = http_cache.ReadOnlyCache(absent)
            self.assertEqual(cache.names(), [])
            self.assertIsNone(cache.read("a.json"))
            self.assertFalse(absent.exists())


class WriteHalfTests(unittest.TestCase):
    def test_the_disk_cache_creates_its_directory_and_keeps_what_it_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "openalex"
            cache = http_cache.DiskCache(directory)
            self.assertTrue(directory.is_dir())
            cache.write("a.json", "kept")
            self.assertEqual((directory / "a.json").read_text(encoding="utf-8"), "kept")

    def test_the_read_only_cache_writes_nothing_and_makes_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "openalex"
            cache = http_cache.ReadOnlyCache(directory)
            cache.write("a.json", "dropped")
            self.assertFalse(directory.exists())
            self.assertIsNone(cache.read("a.json"))


class HubCacheGoesThroughTheSeamTests(unittest.TestCase):
    """The sidecar of a cached batch page is an artifact of the data tree,
    and it now has ONE writer. Read through a read-only cache, the reading
    pass leaves the tree exactly as it found it -- what --dry-run promises
    and what a bare Path could not deliver.
    """

    def _page(self, directory: Path, name: str, ids: list[str], count: int) -> None:
        body = {"meta": {"count": count, "x_query": {
            "oql": "works where it cites (" + " or ".join(ids) + ")",
            "url": f"/works?filter=referenced_works:{'|'.join(ids)}"}}}
        (directory / name).write_text(json.dumps(body), encoding="utf-8")

    def test_a_read_only_pass_creates_no_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._page(directory, "a.json", ["W1"], 18904)
            counts = hub_cache.batch_counts(http_cache.ReadOnlyCache(directory))
            self.assertEqual(counts, [18904])
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ["a.json"])

    def test_a_writing_pass_leaves_the_sidecar_the_client_would_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._page(directory, "a.json", ["W1"], 18904)
            hub_cache.batch_counts(http_cache.DiskCache(directory))
            sidecar = directory / openalex_client.sidecar_name("a.json")
            self.assertTrue(sidecar.is_file())

    def test_the_package_writes_the_tree_only_through_a_seam(self):
        """Two modules may touch the filesystem: the response cache
        (http_cache.py) and the measurements writer that renders a report
        (spike_runs.py). A third one is a channel --dry-run does not cover,
        which is precisely how the sidecar came to have two writers.
        """
        allowed = {"http_cache.py", "spike_runs.py"}
        package = Path(http_cache.__file__).resolve().parent
        for module in sorted(package.glob("*.py")):
            if module.name in allowed:
                continue
            source = module.read_text(encoding="utf-8")
            for spelling in (".write_text(", ".mkdir(", ".open("):
                self.assertNotIn(
                    spelling, source,
                    f"{module.name}: {spelling} -- запись в дерево данных "
                    "принадлежит шву, иначе --dry-run держится аккуратностью")


if __name__ == "__main__":
    unittest.main()
