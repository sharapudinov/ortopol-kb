#!/usr/bin/env python3
"""What an OpenAlex record MEANS, apart from how it is asked for.

Split off openalex_client.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that module owns the conversation -- url, quota, retries,
cache -- and this one owns the readings of what comes back. Every function
here is pure and takes a decoded body, so the shapes measured in run 85
(survey.md §8) are pinned by tests that need neither network nor cache.
"""
from __future__ import annotations


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


def page_index(body: dict) -> dict:
    """{filter, oql, count}: what a cached page says about the BATCH it is a
    page OF -- a few hundred bytes standing in for a body of up to 200 works
    with their referenced_works lists.

    The batch's identity is the `filter=` value, NOT the request url: the url
    carries the cursor in its tail, so eight batches wear 253 distinct urls
    (one per page) and a reader keyed on the url counts the same meta.count
    once per page -- measured: 3 392 521 promised citers instead of 51 652.
    The `filter=` value is the same on every page of a batch.
    """
    meta = body.get("meta") or {}
    query = meta.get("x_query") or {}
    url = query.get("url") or ""
    oql = query.get("oql") or ""
    identity = url.split("filter=", 1)[1].split("&", 1)[0] if "filter=" in url else (oql or url)
    return {"filter": identity, "oql": oql, "count": meta.get("count") or 0}


def short_id(value: str | None) -> str:
    """'https://openalex.org/W123' -> 'W123'; already-short ids pass through."""
    if not value:
        return ""
    return str(value).rstrip("/").rsplit("/", 1)[-1]
