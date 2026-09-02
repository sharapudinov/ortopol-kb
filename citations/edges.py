#!/usr/bin/env python3
"""Deriving citation.cites rows from records the crawl already holds.

Separate from the BFS because an edge is a fact about two nodes, not about
the level at which one of them was discovered: it is written whenever BOTH
endpoints are known, seed-to-seed included, and a node dropped by the filter
is not an endpoint at all.

Derivation runs AFTER a level's nodes are registered. Doing it during
gathering would miss every edge into a work fetched at that same level --
the reference would not yet resolve to a node.
"""
from __future__ import annotations

from .openalex_client import short_id


def among_known(registry, frontier_keys, candidates,
                references) -> list[tuple[str, str, str, str]]:
    """[(citing key, cited key, relation, discovered_from)], deduplicated.

    `references` is {OpenAlex id: reference ids} for the candidates of this
    level -- the lists their records no longer carry (citations/gathering.py)
    and which the caller has already pruned to the ones that became nodes.
    A candidate absent from it points at nothing we know, which is the same
    answer an empty list gives.

    So the second pass walks THAT dict, not the candidate list: a dropped
    candidate has no entry there, and asking the registry to resolve it
    rebuilds its whole namespaced id set (a normalised DOI and a short_id
    per identifier) only to be told None. `candidates` is read once, for the
    provenance of each kept id -- how it reached this level and from which
    node -- which is the one thing the pruned dict does not carry.
    """
    edges: set[tuple[str, str, str, str]] = set()

    def emit(citing_key: str, references, relation: str, source_key: str) -> None:
        for reference in references or []:
            target = registry.resolve_openalex(reference)
            if target and target != citing_key:
                edges.add((citing_key, target, relation, source_key or citing_key))

    for key in frontier_keys:
        emit(key, sorted(registry.nodes[key].referenced_works), "referenced", key)
    provenance: dict[str, tuple[str, str]] = {}
    for record, relation, source_key in candidates:
        provenance.setdefault(short_id(record.get("id")), (relation, source_key))
    for openalex_id, reference_ids in references.items():
        key = registry.resolve_openalex(openalex_id)
        if key is None:
            continue
        relation, source_key = provenance.get(openalex_id, ("", key))
        emit(key, reference_ids, relation, source_key)
    return sorted(edges)
