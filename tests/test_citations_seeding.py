"""Establishing the seed set: which of our documents enter the graph, as
what, and what is recorded about the ones that do not.

Split out of test_citations_crawl.py, which is about the BFS: seeding.py
takes a registry, a client and callables -- never a Snowball -- so the seed
set is established and asserted here with neither a crawl object nor a
writer in sight. Under the crawl module's own class the four cases below
were methods of ScoringMemoryTests, inheriting a batch-recording embedder
they never use and invisible to anyone grepping for the suite by name.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, unit, work
from citation_vocab import WorkKind
from citations import seeding
from citations.registry import WorkRegistry


class SeedingTests(unittest.TestCase):
    """collect_seeds() and rank_seeds(): what is registered, what is
    journalled, and what is returned rather than assigned."""

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


class SeedErrorTests(unittest.TestCase):
    """A document run 85 matched to an OpenAlex id the source then did not
    return.

    Three outcomes, three different rows, and the difference is the whole
    point of journalling at all: no match at all is `seed-missing`, a match
    the source answers for is `seed`, and a match the source is silent
    about is `error` -- we asked for a record that ought to exist and got
    nothing back. Collapsed into the first, that third case reads as "not
    in OpenAlex", which is a statement about the corpus rather than about
    the fetch, and the completeness predicate reads these rows.
    """

    def test_a_requested_id_the_source_never_returned_is_an_error_row(self):
        registry = WorkRegistry()
        client = FakeClient([work("W_SEED_A", title="Seed Chebyshev")])
        steps, n_matched = seeding.collect_seeds(
            registry, client, "c", ["doc_a", "doc_b"],
            {"doc_a": "W_SEED_A", "doc_b": "W_GONE"})
        self.assertEqual(n_matched, 2)
        self.assertEqual(sorted(registry.nodes), ["W_SEED_A"])
        errors = [s for s in steps if s["action"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["frontier_key"], "doc_b")
        self.assertEqual(errors[0]["candidate_key"], "W_GONE")

    def test_the_three_outcomes_are_three_different_actions(self):
        registry = WorkRegistry()
        client = FakeClient([work("W_SEED_A", title="Seed Chebyshev")])
        steps, _ = seeding.collect_seeds(
            registry, client, "c", ["doc_a", "doc_b", "doc_c"],
            {"doc_a": "W_SEED_A", "doc_c": "W_GONE"})
        self.assertEqual({s["action"]: s["frontier_key"] for s in steps},
                         {"seed": "doc_a", "seed-missing": "doc_b", "error": "doc_c"})


class EnrichTests(unittest.TestCase):
    """What OpenAlex does not carry, filled in from the two other sources.

    The abstract wins from OpenAlex when it has one and falls back to
    zbMATH's review (measured: 30 of 56 seeds have one in OpenAlex, 48
    after the fallback); the Math-Net titles are cached on the seed in BOTH
    languages so the twin rule works offline afterwards. Neither precedence
    was exercised by any test, and both are silent when wrong: a swapped
    fallback relabels an OpenAlex abstract as a zbMATH review, and a lost
    title weakens the identity anchor invisibly.
    """

    ABSTRACT = {"Chebyshev": [0], "polynomials": [1]}

    def _seeded(self, record, abstracts=None, names=None):
        registry = WorkRegistry()
        client = FakeClient([record])
        seeding.collect_seeds(registry, client, "c", ["doc_a"], {"doc_a": "W_SEED_A"},
                              abstracts=abstracts, names=names)
        return registry.nodes["W_SEED_A"]

    def test_the_openalex_abstract_wins_when_there_is_one(self):
        node = self._seeded(work("W_SEED_A", abstract=self.ABSTRACT),
                            abstracts={"doc_a": ("обзор zbMATH", "1234.56789")})
        self.assertIn("Chebyshev", node.abstract)
        self.assertNotEqual(node.abstract_source, "zbmath")

    def test_the_zbmath_review_fills_a_seed_openalex_left_blank(self):
        node = self._seeded(work("W_SEED_A"),
                            abstracts={"doc_a": ("обзор zbMATH", "1234.56789")})
        self.assertEqual(node.abstract, "обзор zbMATH")
        self.assertEqual(node.abstract_source, "zbmath")
        self.assertEqual(node.zbmath_id, "1234.56789")

    def test_a_seed_no_source_has_an_abstract_for_keeps_none(self):
        node = self._seeded(work("W_SEED_A"), abstracts={"doc_b": ("чужой", "1.1")})
        self.assertFalse(node.abstract)
        self.assertIsNone(node.zbmath_id)

    def test_both_math_net_titles_are_merged_beside_the_openalex_one(self):
        node = self._seeded(
            work("W_SEED_A", title="Seed Chebyshev", year=1989),
            names={"doc_a": (["Рус название", "Eng title"], [1989, 1991])})
        self.assertIn("Seed Chebyshev", node.titles)
        self.assertIn("Рус название", node.titles)
        self.assertIn("Eng title", node.titles)
        self.assertEqual(sorted(node.years), [1989, 1991])

    def test_a_title_already_on_the_node_is_not_duplicated(self):
        node = self._seeded(work("W_SEED_A", title="Seed Chebyshev", year=1989),
                            names={"doc_a": (["Seed Chebyshev"], [1989])})
        self.assertEqual(node.titles.count("Seed Chebyshev"), 1)
        self.assertEqual(node.years.count(1989), 1)

    def test_neither_source_is_required(self):
        """collect_seeds() is called with no abstracts and no names by the
        calibration path; both default to nothing rather than to a lookup.
        """
        node = self._seeded(work("W_SEED_A", title="Seed Chebyshev"))
        self.assertEqual(node.titles, ["Seed Chebyshev"])
        self.assertEqual(node.kind, WorkKind.OUR_DOCUMENT)


if __name__ == "__main__":
    unittest.main()
