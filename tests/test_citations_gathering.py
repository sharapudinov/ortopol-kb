"""citations/gathering.gather() on its own: what ONE level asks for.

Driven directly with a fake client and a registry, not through
Snowball.expand(): the hub exclusion, the attribution of a citer to the
frontier node it actually cites, and the found-counts are properties of
this function, and a test that reaches them through the crawl's wiring goes
on passing when only that wiring is right.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, work

from citations import gathering
from citations.registry import WorkRegistry

HUB_CAP = 1000


def _registry(*records) -> WorkRegistry:
    registry = WorkRegistry()
    for record, cited_by in records:
        node, _is_new = registry.add(record, kind="external-skeleton", depth=1)
        node.cited_by_count = cited_by
    return registry


class MixedFrontierTests(unittest.TestCase):
    """A hub and an ordinary node in the SAME level. A node past the cap is
    not asked upward -- its citer set is about the field rather than about
    the work -- but it still expands downward, because its references came
    free with the record.
    """

    def setUp(self):
        self.hub = work("W_HUB", title="Hub", refs=["W_R1"])
        self.plain = work("W_PLAIN", title="Plain", refs=["W_R2"])
        # A citer of the ordinary node, and one that only ever cites the hub.
        self.citer = work("W_CITER", title="Citer", refs=["W_PLAIN"])
        self.hub_only = work("W_HUBFAN", title="Hub fan", refs=["W_HUB"])
        self.registry = _registry((self.hub, HUB_CAP + 1), (self.plain, 3))

    def _gather(self, client):
        return gathering.gather(client, self.registry, ["W_HUB", "W_PLAIN"], HUB_CAP)

    def test_only_the_node_under_the_cap_is_asked_upward(self):
        client = FakeClient([work("W_R2", title="Reference")],
                            citers={"W_PLAIN": [self.citer], "W_HUB": [self.hub_only]})
        _candidates, _found, hubs, _references = self._gather(client)
        self.assertEqual(client.cites_batches, [["W_PLAIN"]],
                         "хаб спрошен вверх вопреки колпаку")
        self.assertEqual(hubs, [("W_HUB", HUB_CAP + 1)])

    def test_a_citer_of_the_hub_alone_never_arrives(self):
        """It is not merely dropped after the fact: the hub's ids are not in
        the ask, so nothing about it is paid for.
        """
        client = FakeClient([work("W_R2", title="Reference")],
                            citers={"W_PLAIN": [self.citer], "W_HUB": [self.hub_only]})
        candidates, _found, _hubs, _references = self._gather(client)
        arrived = {record["id"].rsplit("/", 1)[-1] for record, _rel, _from in candidates}
        self.assertIn("W_CITER", arrived)
        self.assertNotIn("W_HUBFAN", arrived)

    def test_a_citer_of_both_is_attributed_to_the_node_that_was_asked(self):
        """The response does not say which of the asked ids a citer cites;
        that is recovered from its own referenced_works, and the hub is not
        among the owners, so it cannot claim the citer.
        """
        both = work("W_BOTH", title="Cites both", refs=["W_HUB", "W_PLAIN"])
        client = FakeClient([work("W_R2", title="Reference")],
                            citers={"W_PLAIN": [both]})
        candidates, found, _hubs, _references = self._gather(client)
        cites = [c for c in candidates if c[1] == "cites"]
        self.assertEqual([(c[1], c[2]) for c in cites], [("cites", "W_PLAIN")])
        self.assertEqual(found["W_HUB"], 1, "хабу засчитан чужой цитирующий")

    def test_found_counts_both_directions_for_the_node_that_earned_them(self):
        """Upward for the node under the cap, downward for both: a hub's
        references cost no request and still count as found.
        """
        client = FakeClient([work("W_R2", title="Reference")],
                            citers={"W_PLAIN": [self.citer]})
        _candidates, found, _hubs, _references = self._gather(client)
        self.assertEqual(found, {"W_HUB": 1, "W_PLAIN": 2})

    def test_every_frontier_key_gets_a_count_even_when_nothing_is_found(self):
        registry = _registry((work("W_QUIET", title="Quiet"), 0))
        _candidates, found, hubs, _references = gathering.gather(
            FakeClient([]), registry, ["W_QUIET"], HUB_CAP)
        self.assertEqual(found, {"W_QUIET": 0})
        self.assertEqual(hubs, [])

    def test_a_reference_the_source_does_not_return_is_still_counted(self):
        """found is what the frontier POINTS AT, not what came back: the
        journal must not read as if the node had fewer references than it
        does because OpenAlex dropped one.
        """
        client = FakeClient([], citers={})
        _candidates, found, _hubs, references = self._gather(client)
        self.assertEqual(found, {"W_HUB": 1, "W_PLAIN": 1})
        self.assertEqual(references, {})

    def test_a_reference_is_asked_for_once_and_credited_to_its_discoverer(self):
        client = FakeClient([work("W_R1", title="Hub reference"),
                             work("W_R2", title="Reference")])
        candidates, _found, _hubs, _references = self._gather(client)
        self.assertEqual(sorted(client.id_batches[0]), ["W_R1", "W_R2"])
        referenced = {record["id"].rsplit("/", 1)[-1]: discovered_from
                      for record, relation, discovered_from in candidates
                      if relation == "referenced"}
        self.assertEqual(referenced, {"W_R1": "W_HUB", "W_R2": "W_PLAIN"})


class CandidateShapeTests(unittest.TestCase):
    """A candidate carries no referenced_works: it is most of a work's
    bytes, a depth-2 level is thousands of candidates, and the list travels
    beside them so the caller can free what the filter dropped.
    """

    def test_the_reference_list_travels_separately_keyed_by_openalex_id(self):
        registry = _registry((work("W_SEED", title="Seed"), 0))
        citer = work("W_CITER", title="Citer", refs=["W_SEED", "W_OTHER"])
        client = FakeClient([], citers={"W_SEED": [citer]})
        candidates, _found, _hubs, references = gathering.gather(
            client, registry, ["W_SEED"], HUB_CAP)
        record, relation, discovered_from = candidates[0]
        self.assertEqual((relation, discovered_from), ("cites", "W_SEED"))
        self.assertNotIn("referenced_works", record)
        self.assertEqual(record["referenced_works_count"], 2,
                         "счётчик ссылок нужен отчёту и остаётся")
        self.assertEqual(references, {"W_CITER": ("W_SEED", "W_OTHER")})

    def test_without_references_leaves_everything_else_alone(self):
        record = gathering.without_references(
            {"id": "W1", "title": "Т", "referenced_works": ["W2"],
             "referenced_works_count": 1})
        self.assertEqual(record, {"id": "W1", "title": "Т",
                                  "referenced_works_count": 1})


if __name__ == "__main__":
    unittest.main()
