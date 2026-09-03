#!/usr/bin/env python3
"""What one level of the snowball ASKS FOR, and what comes back from it.

Split from crawl.py the way seeding.py is: that file is about traversal --
which nodes open the next level, what the filter keeps, what the journal
says -- and this one about the two OpenAlex questions a level consists of
(who cites the frontier, what the frontier cites) and the shape of their
answers. Neither holds a Snowball; the crawl passes its client and registry
in and assigns what comes back.

The one rule that lives here rather than in the caller: a candidate record
carries no referenced_works. It is most of a work's bytes, a depth-2 level
is thousands of candidates, and only the few that become nodes still need
the list afterwards -- so it travels separately, keyed by OpenAlex id, and
the caller frees what the filter dropped.
"""
from __future__ import annotations

from citation_vocab import Relation
from .openalex_client import short_id


def without_references(record: dict) -> dict:
    """The record minus its reference list -- what a candidate carries.

    Same field registry.Node.absorb() leaves out of node.records, and for
    the same reason: it is by far the bulkiest thing OpenAlex returns, and
    what the crawl needs from it (which known works this one points at) is
    extracted once and kept as ids.
    """
    return {key: value for key, value in record.items() if key != "referenced_works"}


def principal_hit(hits) -> str:
    """The one frontier node named where only ONE name fits.

    A candidate is reached from a SET of frontier nodes, but three places
    hold a single name: crawl_step.frontier_key on the keep/drop row, the
    node's own evidence, and the provenance of an edge. The smallest key is
    that name -- deterministic, so two runs over unchanged data write the
    same rows, and never a silent stand-in for the rest: the level's
    per-node counters (the fetch rows) credit every hit, and it is those,
    not this, that answer "what did this frontier node bring in".
    """
    return min(hits) if hits else ""


def gather(client, registry, frontier_keys: list[str], hub_cap: int):
    """(candidates, hub skips, reference lists) for a level.

    A candidate is (record, relation, hits): the relation is one of
    citation_vocab.Relation's, and `hits` is the FROZENSET of frontier nodes
    it was reached from -- discovery is many-to-many in both directions (a
    citer can cite several members of the frontier; a work can be
    referenced by several of them), and a single name would have to pick
    one arbitrarily. Records already known to the registry are returned too
    -- their edges still count -- and the caller recognises them as
    not-new rather than re-scoring them.

    The record a candidate carries has NO referenced_works: that one
    field is most of a work's bytes, a depth-2 level is thousands of
    candidates, and the list is held here only until the level's edges
    are derived. It travels beside the candidates instead, keyed by
    OpenAlex id, so expand() can free the lists of everything the filter
    dropped as soon as the journal row is written -- a dropped candidate
    is no node, and nothing points at it.

    No per-node counts come back from here. They are the crawl's, counted
    over the candidates it actually judged (citations/crawl.py): a count of
    what the frontier POINTS AT is not comparable with a count of what
    passed the filter, and the journal publishes the two side by side.

    A node past the hub cap is not asked upward: its citer set is huge
    and, being huge, is about the field rather than about this work. It
    still expands DOWNWARD, because its references came free with the
    record and cost no request.
    """
    frontier = [registry.nodes[k] for k in frontier_keys]
    hubs = [n for n in frontier if n.cited_by_count > hub_cap]
    hub_keys = {n.key for n in hubs}
    owner = {i: node.key for node in frontier if node.key not in hub_keys
             for i in node.openalex_ids()}
    candidates: list[tuple[dict, str, frozenset[str]]] = []
    references: dict[str, tuple[str, ...]] = {}

    for record in client.citers_of(sorted(owner)):
        cited = tuple(short_id(r) for r in (record.get("referenced_works") or []))
        hits = frozenset(owner[r] for r in cited if r in owner)
        if not hits:
            continue
        candidates.append((without_references(record), Relation.CITES, hits))
        references[short_id(record.get("id"))] = cited

    # Every frontier node that points at a reference, not the first one to:
    # the same work is commonly referenced by several members of the level,
    # and each of them discovered it. Insertion order is the map's own and
    # nothing reads it: the batch below is sorted where it is formed, and
    # the values are sets. A sort per node bought that order twice, over
    # every reference list of every level.
    wanted: dict[str, set[str]] = {}
    for node in frontier:
        for reference in node.referenced_works:
            if registry.resolve_openalex(reference) is None:
                wanted.setdefault(reference, set()).add(node.key)
    for record in client.works_by_ids(sorted(wanted)):
        identity = short_id(record.get("id"))
        candidates.append((without_references(record), Relation.REFERENCED,
                           frozenset(wanted.get(identity, ()))))
        references[identity] = tuple(
            short_id(r) for r in (record.get("referenced_works") or []))
    return candidates, [(n.key, n.cited_by_count) for n in hubs], references
