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

import json
import math
import urllib.request

OLLAMA_URL = "http://127.0.0.1:5471/api/embed"
EMBED_BATCH = 16
# Same bound pg_embed.py uses: bge-m3 holds 8192 tokens, the cut is by
# characters with room to spare so a long abstract is truncated, not dropped.
MAX_CHARS = 6000


def candidate_text(title: str | None, abstract: str | None) -> str:
    """What a node means, for the filter: its title and what it is about."""
    parts = [p.strip() for p in (title, abstract) if p and p.strip()]
    return " ".join(parts)[:MAX_CHARS]


def embed_texts(
    texts: list[str],
    model: str,
    dims: int,
    *,
    url: str = OLLAMA_URL,
    opener=urllib.request.urlopen,
    batch: int = EMBED_BATCH,
) -> list[list[float]]:
    """Embeddings in input order, via the same local ollama pg_embed.py uses."""
    out: list[list[float]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        payload = json.dumps({"model": model, "input": chunk}).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with opener(request, timeout=300) as response:
            body = json.load(response)
        vectors = body["embeddings"]
        if len(vectors) != len(chunk):
            raise RuntimeError(f"ollama вернула {len(vectors)} векторов на {len(chunk)} текстов")
        for vector in vectors:
            if len(vector) != dims:
                raise RuntimeError(f"ожидалось {dims} измерений, пришло {len(vector)}")
        out += vectors
    return out


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
    if len(a) != len(b):
        raise ValueError(f"разная размерность: {len(a)} и {len(b)}")
    na, nb = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(na, nb))


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
