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

Shape, like edges.py next door: explicit parameters in, values out. Nothing
here holds a Snowball or touches its attributes -- crawl.py assigns every
piece of state from what these two functions return, so the seed set's
invariants are readable in one file and both functions are testable with a
registry and a fake client alone.

Two functions rather than one because the ORDER matters and the caller owns
it: the journal rows collect_seeds() returns have to be written before
rank_seeds() runs, since a run where OpenAlex returned nothing has no
centroid, raises, and must still leave behind the rows explaining why.
"""
from __future__ import annotations

from . import journal
from .frontier import centroid, cosine
from .openalex_client import short_id


def _fetch_seeds(registry, client, crawl_id, documents, matches) -> tuple[dict[str, str], list[dict]]:
    """(openalex id -> document id, journal rows) after one batched fetch."""
    steps = [journal.seed_missing(crawl_id, d) for d in documents if d not in matches]
    wanted = {short_id(matches[d]): d for d in documents if d in matches}
    seen = set()
    for record in client.works_by_ids(sorted(wanted)):
        identity = short_id(record.get("id"))
        node, _ = registry.add(record, kind="our-document", depth=0,
                               document_id=wanted.get(identity))
        seen.add(identity)
        steps.append(journal.seed(crawl_id, wanted.get(identity), node.key))
    for openalex_id, document in sorted(wanted.items()):
        if openalex_id not in seen:
            steps.append(journal.seed_error(crawl_id, document, openalex_id))
    return wanted, steps


def _enrich(registry, abstracts, names) -> None:
    """Fill in what OpenAlex does not carry, from the two other sources.

    The abstract wins from OpenAlex when it exists and falls back to zbMATH's
    review (measured: 30 of 56 seeds have one in OpenAlex, 48 after the
    fallback). The names are both Math-Net citations, cached on the seed so
    the twin rule (twins.py) has the Russian AND the English form later
    without going back to a site that rate-limits after a few dozen requests.
    """
    for node in registry.nodes.values():
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


def collect_seeds(registry, client, crawl_id, documents, matches,
                  abstracts=None, names=None) -> tuple[list[dict], int]:
    """Registers every seed the source has, and says what happened to the rest.

    documents: every corpus document id; matches: document -> OpenAlex id
    from run 85; abstracts: document -> (text, zbMATH id); names: document ->
    (titles, years) off Math-Net.

    Returns (journal rows, how many documents the source had a record for).
    The rows are the caller's to write, and must be written before
    rank_seeds() -- see the module docstring.
    """
    wanted, steps = _fetch_seeds(registry, client, crawl_id, documents, matches)
    _enrich(registry, abstracts, names)
    return steps, len(wanted)


def rank_seeds(registry, embed_nodes, n_documents: int,
               n_matched: int) -> tuple[list[str], list[float], dict]:
    """(seed keys, centroid, the depth-0 journal counters).

    `embed_nodes` is the caller's list[Node] -> list[vector], which also
    stores each vector on its node; the embedding model is bound by the
    caller, never chosen here. Raises through centroid() when there are no
    seeds at all: a crawl with no centre cannot score anything.
    """
    seed_keys = sorted(registry.nodes)
    seeds = [registry.nodes[key] for key in seed_keys]
    centre = centroid(embed_nodes(seeds))
    for node in seeds:
        node.score = cosine(node.embedding, centre)
    return seed_keys, centre, {"seeds": len(seed_keys),
                               "seed_missing": n_documents - n_matched}
