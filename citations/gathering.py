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

from .openalex_client import short_id


def without_references(record: dict) -> dict:
    """The record minus its reference list -- what a candidate carries.

    Same field registry.Node.absorb() leaves out of node.records, and for
    the same reason: it is by far the bulkiest thing OpenAlex returns, and
    what the crawl needs from it (which known works this one points at) is
    extracted once and kept as ids.
    """
    return {key: value for key, value in record.items() if key != "referenced_works"}


def gather(client, registry, frontier_keys: list[str], hub_cap: int):
    """(candidates, found counts, hub skips, reference lists) for a level.

    A candidate is (record, relation, discovered_from) with relation in
    {'cites', 'referenced'}. Records already known to the registry are
    returned too -- their edges still count -- and the caller recognises
    them as not-new rather than re-scoring them.

    The record a candidate carries has NO referenced_works: that one
    field is most of a work's bytes, a depth-2 level is thousands of
    candidates, and the list is held here only until the level's edges
    are derived. It travels beside the candidates instead, keyed by
    OpenAlex id, so expand() can free the lists of everything the filter
    dropped as soon as the journal row is written -- a dropped candidate
    is no node, and nothing points at it.

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
    candidates: list[tuple[dict, str, str]] = []
    references: dict[str, tuple[str, ...]] = {}
    found: dict[str, int] = {key: 0 for key in frontier_keys}

    for record in client.citers_of(sorted(owner)):
        cited = tuple(short_id(r) for r in (record.get("referenced_works") or []))
        hit = sorted({owner[r] for r in cited if r in owner})
        if not hit:
            continue
        for key in hit:
            found[key] += 1
        candidates.append((without_references(record), "cites", hit[0]))
        references[short_id(record.get("id"))] = cited

    wanted: dict[str, str] = {}
    for node in frontier:
        for reference in sorted(node.referenced_works):
            found[node.key] += 1
            if registry.resolve_openalex(reference) is None:
                wanted.setdefault(reference, node.key)
    for record in client.works_by_ids(sorted(wanted)):
        identity = short_id(record.get("id"))
        candidates.append((without_references(record), "referenced",
                           wanted.get(identity, "")))
        references[identity] = tuple(
            short_id(r) for r in (record.get("referenced_works") or []))
    return candidates, found, [(n.key, n.cited_by_count) for n in hubs], references
