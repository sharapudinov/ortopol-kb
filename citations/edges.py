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


def among_known(registry, frontier_keys, candidates) -> list[tuple[str, str, str, str]]:
    """[(citing key, cited key, relation, discovered_from)], deduplicated."""
    edges: set[tuple[str, str, str, str]] = set()

    def emit(citing_key: str, references, relation: str, source_key: str) -> None:
        for reference in references or []:
            target = registry.resolve_openalex(reference)
            if target and target != citing_key:
                edges.add((citing_key, target, relation, source_key or citing_key))

    for key in frontier_keys:
        emit(key, sorted(registry.nodes[key].referenced_works), "referenced", key)
    for record, relation, source_key in candidates:
        key = registry.find(record)
        if key is not None:
            emit(key, record.get("referenced_works"), relation, source_key)
    return sorted(edges)
