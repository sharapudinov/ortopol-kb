#!/usr/bin/env python3
"""What a level's candidates are worth, and which of them are new.

Split from crawl.py the way seeding.py and gathering.py are: that file is
about traversal -- which nodes open the next level, what the filter keeps,
what the journal says -- and this one about the step between gathering and
the filter, where raw records become scored candidates. Neither holds a
Snowball; the crawl passes its registry and its vector seam in.

Nothing here decides anything: tau is applied by the caller, one journal row
per candidate. What is decided here is only what a score is measured OVER --
a candidate is scored as registry.ScoringFields (key, title, and the
abstract still inverted), never as a Node, because it becomes a node only
after it has passed tau.
"""
from __future__ import annotations

from .gathering import principal_hit
from .openalex_records import short_id
from .registry import record_ids, scoring_fields
from .scoring import NO_TEXT_SCORE, cosine_unit, keeps


def scores_of(vectors_for, centroid, tau, holders) -> dict[str, tuple[float, list[float] | None]]:
    """{key: (score, vector)} for `holders` -- registry.ScoringFields
    triples, or anything else carrying key/title/abstract -- embedded a
    batch at a time, and the vector of anything below tau dropped as soon
    as its score is known.

    Batched by `vectors_for`, because the vector is the expensive thing
    here: 1024 floats per candidate, and a depth-2 level is thousands of
    candidates (~4262 distinct references measured at tau=0.50) of which
    the filter keeps a fraction. vectors_for() yields a chunk at a time and
    this loop scores each pair as it arrives, so the vectors alive at once
    are one chunk plus what passed tau -- a function of the KEPT set, not of
    the candidate set.
    """
    scored: dict[str, tuple[float, list[float] | None]] = {}
    for holder, vector in vectors_for(holders):
        score = cosine_unit(vector, centroid)
        scored[holder.key] = (score, vector if keeps(score, tau) else None)
    return scored


def score(registry, measure, candidates) -> list[dict]:
    """Cosine to the seed centroid for every NEW candidate.

    `measure` is a callable holders -> {key: (score, vector)}; the crawl
    binds scores_of() above with its own centroid and tau.

    A candidate with no title carries no semantic content (the predicate
    pg_embed.py applies to a page) and is scored NO_TEXT_SCORE rather than
    embedded: an empty string would land somewhere arbitrary on the sphere
    instead of being visibly unusable. The journal says the same thing in
    words off that same number (citations/journal.drop) -- "below-threshold"
    would report a relevance verdict on a candidate nothing was measured on.

    `hits` travels through unchanged -- the whole set of frontier nodes the
    candidate was reached from -- and `discovered_from` is the one name the
    single-valued places need (gathering.principal_hit).

    The resolution asking "is this one new" travels out with the candidate:
    `record_ids` is a pure function of the record and every kept candidate
    is resolved again by registry.add(), so deriving it here and handing it
    back is one derivation per candidate instead of three. The ANSWER is not
    carried -- only the ids -- because a twin union earlier in the same level
    can move a candidate onto an existing node between this filter and the
    write. Same cost as the one ScoringFields exists to avoid, one layer up.
    """
    fresh, seen = [], set()
    for record, relation, hits in candidates:
        identity = short_id(record.get("id"))
        if identity in seen:
            continue
        ids = record_ids(record)
        if registry.find_ids(ids) is not None:
            continue
        seen.add(identity)
        fresh.append((record, relation, hits, identity, ids))

    holders = [scoring_fields(item[0]) for item in fresh]
    scored = measure([holder for holder in holders if holder.title])

    out = []
    for record, relation, hits, key, ids in fresh:
        value, vector = scored.get(key, (NO_TEXT_SCORE, None))
        out.append({
            "record": record, "relation": relation, "hits": hits,
            "discovered_from": principal_hit(hits),
            "candidate_key": key, "record_ids": ids,
            "score": value, "vector": vector,
            "title": record.get("title") or record.get("display_name"),
            "year": record.get("publication_year"),
        })
    return out
