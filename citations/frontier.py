#!/usr/bin/env python3
"""The relevance filter: what the snowball keeps and what it drops.

Score = cosine between a candidate's title+abstract embedding and the
centroid of the seeds' embeddings. Both sides go through the SAME model the
corpus uses (corpus.embedding_model, read from the database and passed in
here -- never a constant in this file): mixing models produces a distance
rather than an error, which is the failure mode CLAUDE.md names explicitly.

The threshold tau is NOT chosen here and has no default anywhere in the
package. It is measured: pg_load_citations.py --calibrate scores every
depth-1 candidate, writes the distribution to
measurements.citation_frontier_threshold, and the verdict on where the line
falls is the orchestrator's. The helpers at the bottom exist to render that
distribution honestly (quantiles and a text histogram), not to pick a
number.
"""
from __future__ import annotations

import math
import urllib.request

from pg_embedding_text import MAX_CHARS, works_text
from pg_search import EMBED_BATCH, OLLAMA_URL, embed_batch

from .inputs import KEY_BATCH


def candidate_text(title: str | None, abstract: str | None) -> str:
    """What a node means, for the filter: its title and what it is about.

    The rule is pg_embedding_text's, not this module's: the other producer
    of citation.work.embedding (pg_embed.py) builds the same string in SQL,
    and two spellings of it would differ by a plausible cosine rather than
    by an error.
    """
    return works_text(title, abstract)


def embed_texts(
    texts: list[str],
    model: str,
    dims: int,
    *,
    url: str = OLLAMA_URL,
    opener=urllib.request.urlopen,
    batch: int = EMBED_BATCH,
) -> list[list[float]]:
    """Embeddings in input order, through the repository's embedding seam.

    The seam is pg_search's, not one of our own: resolve_model() reads which
    model produced the stored vectors and embed_batch() is the request, the
    batching and both checks (count and width). A second implementation of
    that contract here would be a second place for the model, the URL and
    the dimension check to drift -- and a query vector from a different
    model yields a perfectly well-formed distance that means nothing.
    """
    return embed_batch(model, dims, texts, ollama_url=url, batch=batch, opener=opener)


def vectors_for(embed, known_vectors, holders):
    """(holder, vector) pairs for `holders` (anything carrying key/title/
    abstract), in order, a chunk at a time: read from the store where
    already known, embedded where not.

    Two seams, both bound by the caller -- `embed` is the model-bound
    embedder above, `known_vectors` is list[str] -> {key: vector} over what
    the database already holds (citations/inputs.known_embeddings).

    The read is not free either: --calibrate embeds every depth-1
    candidate and writes no node, the crawl that follows meets the same
    candidates, and a re-crawl without --resume meets every node it ever
    wrote. At a depth-2 level (~4262 distinct candidates measured) that is
    thousands of bge-m3 inferences and hundreds of round trips paid for
    vectors already in Postgres.

    TWO batch sizes, because the two seams are charged differently. The
    store read is a psql round trip -- a temp script, a fork, a fresh
    connection -- so it takes KEY_BATCH keys at once, the same IN-list size
    the read batches by (citations/inputs.py); at the measured level that
    is ~22 round trips instead of ~267. The embedder is charged per text
    and answers one request per call, so the misses INSIDE a read block go
    to ollama in EMBED_BATCH sub-batches.

    A GENERATOR, and chunked end to end. A vector is 1024 floats (~32 KB as
    a Python list), so the returned list alone was ~130 MB at that level,
    plus as much again in the dict the whole-set read built: peak memory was
    a function of the CANDIDATE set however promptly the consumer dropped
    what it did not want. Yielding per sub-batch is what lets scores_of()
    have a peak that follows the KEPT set, which is what its own docstring
    always claimed. What survives a sub-batch is one read block's stored
    vectors (KEY_BATCH of them at most) plus what the consumer chose to
    keep, which is why `fresh` is rebound per sub-batch rather than filled
    once per block.
    """
    for start in range(0, len(holders), KEY_BATCH):
        block = holders[start:start + KEY_BATCH]
        known = known_vectors([h.key for h in block])
        for offset in range(0, len(block), EMBED_BATCH):
            chunk = block[offset:offset + EMBED_BATCH]
            fresh: dict[str, list[float]] = {}
            missing = [h for h in chunk if h.key not in known]
            if missing:
                vectors = embed([candidate_text(h.title, h.abstract) for h in missing])
                fresh = {h.key: vector for h, vector in zip(missing, vectors)}
            for holder in chunk:
                yield holder, (known[holder.key] if holder.key in known
                               else fresh[holder.key])


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def centroid(vectors: list[list[float]]) -> list[float]:
    """Mean of the L2-normalized seed vectors, itself normalized.

    Normalizing before averaging is what makes the centroid a direction
    rather than a length-weighted average: without it a seed with a long
    abstract would pull the centre harder than a seed with a short one, for
    no reason anyone would defend.
    """
    if not vectors:
        raise ValueError("центроид пустого множества семян не определён")
    unit = [l2_normalize(v) for v in vectors]
    size = len(unit[0])
    if any(len(v) != size for v in unit):
        raise ValueError("векторы семян разной длины")
    mean = [sum(v[i] for v in unit) / len(unit) for i in range(size)]
    return l2_normalize(mean)


def cosine(a: list[float], b: list[float]) -> float:
    """Косинус между двумя произвольными векторами: нормируются оба."""
    if len(a) != len(b):
        raise ValueError(f"разная размерность: {len(a)} и {len(b)}")
    na, nb = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(na, nb))


def cosine_unit(a: list[float], unit_b: list[float]) -> float:
    """То же число, когда вторая сторона УЖЕ единичная.

    Ровно случай фильтра: центроид возвращается из centroid() нормированным
    и не меняется весь обход, а cosine() нормировал бы его заново на каждого
    кандидата — проход по 1024 числам за суммой квадратов, второй за
    делением и свежий список на выброс, тысячи раз за уровень.
    """
    if len(a) != len(unit_b):
        raise ValueError(f"разная размерность: {len(a)} и {len(unit_b)}")
    norm = math.sqrt(sum(v * v for v in a))
    if norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, unit_b)) / norm


def split_by_threshold(scored: dict[str, float], tau: float) -> tuple[list[str], list[str]]:
    """(kept, dropped) keys. `>= tau` keeps: the boundary belongs to the
    side the calibration recommended, and a candidate scoring exactly the
    recommended number is by construction one we said we wanted."""
    kept = sorted(k for k, s in scored.items() if s >= tau)
    dropped = sorted(k for k, s in scored.items() if s < tau)
    return kept, dropped


def quantiles(values: list[float], points=(0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[float, float]:
    """Nearest-rank quantiles -- every reported number is an observed score,
    not an interpolation between two of them."""
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
        out[point] = ordered[index]
    return out


def histogram(values: list[float], bins: int = 20) -> list[tuple[float, float, int]]:
    """[(low, high, count)] over [min, max]; the last bin is closed."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [(low, high, len(values))]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return [(low + i * width, low + (i + 1) * width, counts[i]) for i in range(bins)]
