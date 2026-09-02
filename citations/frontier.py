#!/usr/bin/env python3
"""The relevance filter's SEAM: what a candidate means as a vector, and
where vectors come from.

Score = cosine between a candidate's title+abstract embedding and the
centroid of the seeds' embeddings. Both sides go through the SAME model the
corpus uses (corpus.embedding_model, read from the database and passed in
here -- never a constant in this file): mixing models produces a distance
rather than an error, which is the failure mode CLAUDE.md names explicitly.

The arithmetic itself is citations/scoring.py and has no dependency beyond
`math`; everything here binds something outside the process -- the ollama
embedder through pg_search, the stored vectors through a reader the caller
passes in. The two lived in one module and the consumer that wanted only
the second half said so by importing it under an alias.
"""
from __future__ import annotations

import urllib.request

from pg_embedding_text import works_text
from pg_search import EMBED_BATCH, OLLAMA_URL, embed_batch

from .inputs import KEY_BATCH
from .vector_cache import memoizing_embedder


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


def bound_embedder(model: str, dims: int, memo):
    """The crawl's embedder: embed_texts() bound to the corpus model, with
    the data tree's vector memo in front of it (citations/vector_cache.py).

    Both halves belong to this seam rather than to the caller. The model is
    bound once, from corpus.embedding_model, because a vector from another
    model is a plausible distance and not an error; the memo is what keeps
    the documented `--calibrate` then crawl pair from buying the same level
    of vectors twice. Which memo -- writing or read-only -- is the caller's,
    and only the caller's: it is an http_cache object chosen by the run's
    mode (pg_load_citations.main), never a flag threaded through here.
    """
    return memoizing_embedder(lambda texts: embed_texts(texts, model, dims), memo)


def vectors_for(embed, known_vectors, holders):
    """(holder, vector) pairs for `holders` (anything carrying key/title/
    abstract), in order, a chunk at a time: read from the store where
    already known, embedded where not.

    Two seams, both bound by the caller -- `embed` is the model-bound
    embedder above, `known_vectors` is list[str] -> {key: vector} over what
    the database already holds (citations/inputs.known_embeddings).

    The read is not free either: a re-crawl without --resume meets every
    node it ever wrote. At a depth-2 level (~4262 distinct candidates
    measured) that is thousands of bge-m3 inferences and hundreds of round
    trips paid for vectors already in Postgres.

    What the store read canNOT save is the `--calibrate` then crawl pair
    the docs prescribe: a calibration writes no work row, so the level it
    just embedded leaves nothing here to find. That saving belongs to the
    embedder's own memo in the data tree (citations/vector_cache.py, bound
    by bound_embedder above) -- this function's `embed` is already behind
    it, and a miss here is a cache lookup before it is an inference.

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
