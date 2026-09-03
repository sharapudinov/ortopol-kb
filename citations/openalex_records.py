#!/usr/bin/env python3
"""What an OpenAlex record MEANS, apart from how it is asked for.

Split off openalex_client.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that module owns the conversation -- url, quota, retries,
cache -- and this one owns the readings of what comes back. Every function
here is pure and takes a decoded body, so the shapes measured in run 85
(survey.md §8) are pinned by tests that need neither network nor cache.
"""
from __future__ import annotations

from citation_vocab import Relation

# The OTHER direction a batch can be asked in, and not a relation at all:
# `openalex_id:` fetches works BY id, which is how the down direction buys
# the metadata for references it already knows. It has no crawl_step.relation
# spelling because no journal row is ever about it.
OPENALEX_ID = "openalex_id"


def restore_abstract(inverted: dict | None) -> str | None:
    """`abstract_inverted_index` (word -> [positions]) back into text.

    Gaps in the position sequence are left alone rather than padded: the
    index is what the source published, and inventing filler words would
    put text we made up into work.abstract.
    """
    if not inverted:
        return None
    placed: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions or []:
            placed.append((int(position), word))
    if not placed:
        return None
    placed.sort()
    return " ".join(word for _, word in placed).strip() or None


SIDECAR_SUFFIX = ".meta.json"


def sidecar_name(page: str) -> str:
    """The name of one cached page's index, beside the page itself."""
    return (page[: -len(".json")] if page.endswith(".json") else page) + SIDECAR_SUFFIX


# Which way a batch was asked, by the FIELD its filter names. OpenAlex
# echoes the request's filter back normalised -- the crawl asks `cites:` and
# x_query.url comes back saying `referenced_works:`, it asks `openalex_id:`
# and the answer says `ids.openalex:` -- so both spellings of each direction
# are listed. Measured over the whole cache: 253 pages of the first, 6 of
# the second, nothing else.
#
# The direction is a QUERY fact and is read as one. The same page also
# carries meta.x_query.oql, OpenAlex's rendered English sentence ("works
# where it cites (...)"), and a reader that sniffed that sentence for the
# word "cites" was reading third-party presentation text: a rewording on
# their side classifies every page as not-a-batch, and the hub measurement
# then reports "nothing to measure" against a cache full of what it wanted.
DIRECTIONS = {
    "referenced_works": Relation.CITES,
    "cites": Relation.CITES,
    "ids.openalex": OPENALEX_ID,
    "openalex_id": OPENALEX_ID,
}


def _filter_value(url: str) -> str:
    return url.split("filter=", 1)[1].split("&", 1)[0] if "filter=" in url else ""


def direction_of(filter_value: str) -> str | None:
    """Relation.CITES | OPENALEX_ID | None for a `filter=` value.

    None is an honest answer: a page fetched by some other filter belongs to
    neither direction, and so does one whose url carried no filter at all.
    """
    return DIRECTIONS.get(str(filter_value).split(":", 1)[0])


def page_index(body: dict) -> dict:
    """{filter, direction, oql, count}: what a cached page says about the
    BATCH it is a page OF -- a few hundred bytes standing in for a body of up
    to 200 works with their referenced_works lists.

    The batch's identity is the `filter=` value, NOT the request url: the url
    carries the cursor in its tail, so eight batches wear 253 distinct urls
    (one per page) and a reader keyed on the url counts the same meta.count
    once per page -- measured: 3 392 521 promised citers instead of 51 652.
    The `filter=` value is the same on every page of a batch.

    `oql` is kept for a human reading the sidecar, and for nothing else:
    every decision a reader makes comes off `direction`.
    """
    meta = body.get("meta") or {}
    query = meta.get("x_query") or {}
    url = query.get("url") or ""
    oql = query.get("oql") or ""
    filter_value = _filter_value(url)
    identity = filter_value or oql or url
    return {"filter": identity, "direction": direction_of(filter_value),
            "oql": oql, "count": meta.get("count") or 0}


def note_direction(note: dict) -> str | None:
    """A page index's direction, for indexes written before it was a field.

    Sidecars are durable and are not rewritten: the ones already lying beside
    the cached pages carry `filter` but no `direction`, and the direction is
    exactly what `filter` says.
    """
    if "direction" in note:
        return note["direction"]
    return direction_of(note.get("filter") or "")


def short_id(value: str | None) -> str:
    """'https://openalex.org/W123' -> 'W123'; already-short ids pass through."""
    if not value:
        return ""
    return str(value).rstrip("/").rsplit("/", 1)[-1]
