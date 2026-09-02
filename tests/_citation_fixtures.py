"""Fixtures shared by the two citation-snowball test modules.

Kept beside _pathfix.py rather than duplicated: the offline unit tests and
the crawl/idempotency tests need the same fake OpenAlex responses, and two
copies of a fixture drift the moment one of them is corrected.

The vectors are axis-aligned 1024-vectors on purpose. The embedding column
is vector(1024), so the live half of the suite needs the real width, and
axis alignment makes every cosine in the tests exactly 1.0 or 0.0 -- an
assertion about the filter, not about floating point.
"""
from __future__ import annotations

import _pathfix  # noqa: F401
from citations.openalex_client import batched, short_id

DIMS = 1024


def work(identifier, *, title=None, doi=None, mag=None, year=2000, refs=(), abstract=None):
    """An OpenAlex work record in the shape WORK_SELECT asks for."""
    ids = {"openalex": f"https://openalex.org/{identifier}"}
    if doi:
        ids["doi"] = f"https://doi.org/{doi}"
    if mag:
        ids["mag"] = mag
    return {
        "id": f"https://openalex.org/{identifier}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title if title is not None else f"Title of {identifier}",
        "display_name": title if title is not None else f"Title of {identifier}",
        "publication_year": year,
        "ids": ids,
        "abstract_inverted_index": abstract,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "referenced_works_count": len(refs),
        "cited_by_count": 0,
        "authorships": [{"author": {"display_name": "I. I. Sharapudinov"}}],
        "type": "article",
        "language": "en",
    }


class FakeClient:
    """Records keyed by id, plus a citers table; counts calls and batch sizes."""

    def __init__(self, records, citers=None):
        self.records = {short_id(r["id"]): r for r in records}
        self.citers = citers or {}
        self.id_batches: list[list[str]] = []
        self.cites_batches: list[list[str]] = []
        self.n_requests = 0
        self.n_cache_hits = 0

    def works_by_ids(self, ids):
        out = []
        for chunk in batched(short_id(i) for i in ids):
            self.id_batches.append(chunk)
            self.n_requests += 1
            out += [self.records[i] for i in chunk if i in self.records]
        return out

    def citers_of(self, ids):
        out, seen = [], set()
        for chunk in batched(short_id(i) for i in ids):
            self.cites_batches.append(chunk)
            self.n_requests += 1
            for identifier in chunk:
                for citer in self.citers.get(identifier, []):
                    if short_id(citer["id"]) not in seen:
                        seen.add(short_id(citer["id"]))
                        out.append(citer)
        return out


def unit(index: int, weight: float = 1.0) -> list[float]:
    """A 1024-vector on axis `index`, so cosines are exactly plannable."""
    vector = [0.0] * DIMS
    vector[index % DIMS] = weight
    return vector


class PlannedEmbedder:
    """Maps text -> vector by a substring table; unknown text -> a far axis."""

    def __init__(self, table):
        self.table = table
        self.calls = 0
        # Every text ever handed to the embedder, flat. `calls` counts
        # batches; the question "was this vector paid for" is about texts.
        self.texts: list[str] = []

    def __call__(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        out = []
        for text in texts:
            for marker, vector in self.table.items():
                if marker in text:
                    out.append(vector)
                    break
            else:
                out.append(unit(1023))
        return out


