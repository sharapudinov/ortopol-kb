"""What a node still carries once its row has been written.

Split off test_citations_crawl.py by subject (and by kb/CLAUDE.md
FILE_SIZE): that file is the shape of a level -- what is gathered, filtered,
journalled and turned into edges -- and this one is the LIFETIME of the two
payloads a node holds only until the writer has taken them.

The registry lives as long as the crawl does, because the next level reads
each node's ids, relation and reference list off it. The vector and the raw
source records are read once, by writer.works(), and never again -- so held
past that write they are dead weight multiplied by every node the crawl ever
keeps: 1024 floats each, plus a record list that grows on every re-sighting.
That defeats the bound the frontier/scores restructuring was built for,
which is a property of what stays alive, not of what is written.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, unit, work
from citations.crawl import Snowball
from citations.dry_store import DryRunWriter


class _WatchingWriter(DryRunWriter):
    """A dry run that copies each node's payload AT the moment it is written.

    The nodes themselves are the same objects the registry holds, so a
    writer that only keeps references cannot tell "the vector never arrived"
    from "the vector arrived and was released afterwards".
    """

    def __init__(self):
        super().__init__()
        self.vectors: dict[str, list[float] | None] = {}
        self.n_records: dict[str, int] = {}

    def works(self, nodes) -> int:
        nodes = list(nodes)
        for node in nodes:
            self.vectors[node.key] = node.embedding
            self.n_records[node.key] = len(node.records)
        return super().works(nodes)


def _crawl(writer, **kwargs):
    """Seed W_SEED, cited by W_A, itself cited by W_B: two levels."""
    seed = work("W_SEED", title="Seed Chebyshev")
    first = work("W_A", title="First Chebyshev", refs=["W_SEED"])
    second = work("W_B", title="Second Chebyshev", refs=["W_A"])
    client = FakeClient([seed, first, second],
                        citers={"W_SEED": [first], "W_A": [second]})
    snowball = Snowball(client, PlannedEmbedder({"Chebyshev": unit(0)}), writer,
                        tau=0.5, crawl_id="c", log=lambda *_: None, **kwargs)
    snowball.seed(["doc_a"], {"doc_a": "W_SEED"})
    return snowball


class TheWriteTakesThePayloadTests(unittest.TestCase):
    """It reaches the writer -- that is what it is for."""

    def test_a_kept_candidate_hands_its_vector_to_the_write(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.expand(["W_SEED"], 1)
        self.assertEqual(writer.vectors["W_A"], unit(0))

    def test_a_kept_candidate_hands_its_record_to_the_write(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.expand(["W_SEED"], 1)
        self.assertEqual(writer.n_records["W_A"], 1)

    def test_a_seed_hands_over_its_vector_too(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.write_seeds()
        self.assertEqual(writer.vectors["W_SEED"], unit(0))


class NothingIsHeldAfterTheWriteTests(unittest.TestCase):
    def test_a_written_node_keeps_neither_vector_nor_records(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.expand(["W_SEED"], 1)
        node = snowball.registry.nodes["W_A"]
        self.assertIsNone(node.embedding, "вектор жив после записи узла")
        self.assertEqual(node.records, [], "сырые записи живы после записи узла")

    def test_a_written_seed_keeps_neither_either(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.write_seeds()
        node = snowball.registry.nodes["W_SEED"]
        self.assertIsNone(node.embedding)
        self.assertEqual(node.records, [])

    def test_what_the_next_level_reads_survives(self):
        """Released is the write-only half. The ids, the relation and the
        reference list are what expandable() and edges.among_known ask the
        registry for at the level after this one.
        """
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.expand(["W_SEED"], 1)
        node = snowball.registry.nodes["W_A"]
        self.assertEqual(node.openalex_ids(), ["W_A"])
        self.assertEqual(node.referenced_works, {"W_SEED"})
        self.assertEqual(node.relation, "cites")
        self.assertEqual(node.title, "First Chebyshev")

    def test_the_level_after_the_release_still_runs(self):
        writer = _WatchingWriter()
        snowball = _crawl(writer)
        snowball.run(2)
        self.assertEqual(sorted(snowball.registry.nodes), ["W_A", "W_B", "W_SEED"])
        self.assertEqual(writer.vectors["W_B"], unit(0),
                         "узел второго уровня записан без вектора")
        for key in ("W_SEED", "W_A", "W_B"):
            self.assertIsNone(snowball.registry.nodes[key].embedding, key)
            self.assertEqual(snowball.registry.nodes[key].records, [], key)


if __name__ == "__main__":
    unittest.main()
