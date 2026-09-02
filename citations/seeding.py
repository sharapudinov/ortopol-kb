#!/usr/bin/env python3
"""Establishing the seed set: which corpus documents enter the graph, as what.

Split out of crawl.py because it answers a different question. The BFS asks
"what is next to this node"; seeding asks "which of OUR 69 documents does the
source know at all, and under which record" -- and its hard part is not
traversal but identity across three sources: the run-85 match for the
OpenAlex record, zbMATH for the abstract OpenAlex lacks, Math-Net for the
names in both languages.

A document the source does not have gets a journal row and NO work row.
"Absent from OpenAlex" is a recorded decision that the completeness
predicate reads; a missing row would be indistinguishable from a bug.
"""
from __future__ import annotations

from . import journal
from .frontier import centroid, cosine
from .openalex_client import short_id


def _fetch_seeds(snow, documents, matches) -> tuple[dict[str, str], list[dict]]:
    """(openalex id -> document id, journal rows) after one batched fetch."""
    steps = [journal.seed_missing(snow.crawl_id, d)
             for d in documents if d not in matches]
    wanted = {short_id(matches[d]): d for d in documents if d in matches}
    seen = set()
    for record in snow.client.works_by_ids(sorted(wanted)):
        identity = short_id(record.get("id"))
        node, _ = snow.registry.add(record, kind="our-document", depth=0,
                                    document_id=wanted.get(identity))
        seen.add(identity)
        steps.append(journal.seed(snow.crawl_id, wanted.get(identity), node.key))
    for openalex_id, document in sorted(wanted.items()):
        if openalex_id not in seen:
            steps.append(journal.seed_error(snow.crawl_id, document, openalex_id))
    return wanted, steps


def _enrich(snow, abstracts, names) -> None:
    """Fill in what OpenAlex does not carry, from the two other sources.

    The abstract wins from OpenAlex when it exists and falls back to zbMATH's
    review (measured: 30 of 56 seeds have one in OpenAlex, 48 after the
    fallback). The names are both Math-Net citations, cached on the seed so
    the twin rule (twins.py) has the Russian AND the English form later
    without going back to a site that rate-limits after a few dozen requests.
    """
    for node in snow.registry.nodes.values():
        filled = (abstracts or {}).get(node.document_id)
        if filled and not node.abstract:
            node.abstract, node.abstract_source = filled[0], "zbmath"
            node.zbmath_id = filled[1]
        titles, years = (names or {}).get(node.document_id, ([], []))
        for title in titles:
            if title not in node.titles:
                node.titles.append(title)
        for year in years:
            if year not in node.years:
                node.years.append(year)


def seed(snow, documents, matches, abstracts=None, names=None) -> list[str]:
    """documents: every corpus document id; matches: document -> OpenAlex id
    from run 85; abstracts: document -> (text, zbMATH id); names: document ->
    (titles, years) off Math-Net. Returns the seed keys."""
    wanted, steps = _fetch_seeds(snow, documents, matches)
    _enrich(snow, abstracts, names)

    # Journalled BEFORE the centroid is computed, not after: a run where
    # OpenAlex returned nothing has no centroid and cannot continue, and the
    # error rows explaining why must survive that exit.
    snow.writer.journal(steps)

    snow.seed_keys = sorted(snow.registry.nodes)
    seeds = [snow.registry.nodes[k] for k in snow.seed_keys]
    snow.centroid = centroid(snow.embed_nodes(seeds))
    for node in seeds:
        node.score = cosine(node.embedding, snow.centroid)
    snow.per_depth[0] = {"seeds": len(snow.seed_keys),
                         "seed_missing": len(documents) - len(wanted)}
    return snow.seed_keys
