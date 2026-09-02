"""Turning a level's records into citation.cites rows.

Split from test_citations_crawl.py (kb/CLAUDE.md FILE_SIZE) along the seam
citations/edges.py already draws: an edge is a fact about two nodes, not
about the level one of them was found at, so it is derived after the level
is registered and asserted on its own.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import work
from citations import edges as edges_mod
from citations.registry import WorkRegistry


class SelfCitationTests(unittest.TestCase):
    """A reference that resolves to the citing node itself is no edge.

    It happens through the twin union, not through a source error: the
    English translation carries the Russian original among its
    referenced_works, both records share a DOI, and after the union both
    ends of that "edge" are one node (citation.cites CHECKs citing <> cited).
    """

    def _registry_with_a_twin(self):
        registry = WorkRegistry()
        original = work("W_SEED", title="Seed Chebyshev", doi="10.1/x")
        node, _new = registry.add(original, kind="our-document", depth=0,
                                  document_id="doc_a")
        node.referenced_works |= {"W_TRANS"}
        translation = work("W_TRANS", title="Seed Chebyshev, translated", doi="10.1/x")
        _same, is_new = registry.add(translation, kind="external-skeleton", depth=1)
        self.assertFalse(is_new, "перевод не слился с оригиналом — фикстура не о том")
        return registry

    def test_the_self_loop_is_not_written(self):
        registry = self._registry_with_a_twin()
        self.assertEqual(registry.resolve_openalex("W_TRANS"), "W_SEED")
        self.assertEqual(edges_mod.among_known(registry, ["W_SEED"], [], {}), [])

    def test_a_reference_to_anybody_else_still_is(self):
        registry = self._registry_with_a_twin()
        registry.add(work("W_OTHER", title="Somebody else"),
                     kind="external-skeleton", depth=1)
        registry.nodes["W_SEED"].referenced_works |= {"W_OTHER"}
        self.assertEqual(edges_mod.among_known(registry, ["W_SEED"], [], {}),
                         [("W_SEED", "W_OTHER", "referenced", "W_SEED")])

class DerivationCostTests(unittest.TestCase):
    """The caller has already pruned `references` to the candidates that
    became nodes, so a dropped one is not an endpoint and cannot contribute
    an edge. Re-resolving it anyway means rebuilding its whole namespaced id
    set (a normalised DOI and a short_id per identifier) to be told None --
    at a depth-2 level, thousands of times, for the fourth time in that same
    level.
    """

    class _CountingRegistry(WorkRegistry):
        def __init__(self):
            super().__init__()
            self.finds = 0

        def find(self, record):
            self.finds += 1
            return super().find(record)

    def _level(self):
        registry = self._CountingRegistry()
        registry.add(work("W_SEED", title="Seed"), kind="our-document", depth=0,
                     document_id="doc_a")
        kept = work("W_KEEP", title="Near", refs=["W_SEED"])
        registry.add(kept, kind="external-skeleton", depth=1)
        dropped = [work(f"W_DROP{i}", title="Far") for i in range(20)]
        hits = frozenset({"W_SEED"})
        candidates = ([(kept, "cites", hits)]
                      + [(record, "cites", hits) for record in dropped])
        # What crawl.py hands over: only the candidates that became nodes.
        references = {"W_KEEP": ["W_SEED"]}
        return registry, candidates, references

    def test_the_dropped_candidates_are_not_resolved_again(self):
        registry, candidates, references = self._level()
        before = registry.finds
        edges = edges_mod.among_known(registry, ["W_SEED"], candidates, references)
        self.assertEqual(edges, [("W_KEEP", "W_SEED", "cites", "W_SEED")])
        self.assertLessEqual(registry.finds - before, len(references),
                             "registry.find вызван по отброшенным кандидатам")

    def test_a_candidate_with_no_pruned_references_contributes_nothing(self):
        registry, candidates, _references = self._level()
        self.assertEqual(edges_mod.among_known(registry, [], candidates, {}), [])


if __name__ == "__main__":
    unittest.main()
