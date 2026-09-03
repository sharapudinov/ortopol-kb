#!/usr/bin/env python3
"""A disk memo in front of the embedder: the fourth channel the data tree
keeps, beside the three response caches.

WHY it exists. The store read (citations/inputs.known_embeddings) answers
"is this candidate's vector already in citation.work", and the documented
pipeline -- `--calibrate` to measure tau, then a crawl at that tau -- gets
no help from it at all: a calibration run writes no work row (its writer is
DryRunWriter, by construction, because it is a measurement and not a crawl),
so the level it just embedded leaves nothing behind. The OpenAlex cache
saves the pages; nothing saved the vectors, and the same thousands of bge-m3
inferences were paid twice. A re-crawl of a level whose nodes were written
IS served by the store read; the pair the docs single out was not.

WHAT the key is: the model and the exact text the vector was computed from
(frontier.candidate_text, i.e. pg_embedding_text's rule). Not the candidate
key -- a work whose title or abstract was corrected must get a new vector,
and a key-addressed memo would serve the old one forever, which is the same
class of silent wrongness EMBEDDING_ONE_CONTRACT names for the model. The
model is part of the entry name rather than of its content so that a corpus
re-embedded under another model simply misses, instead of reading a
plausible number.

HOW it writes nothing under --dry-run: it does not decide that. It is
handed an http_cache.Cache object (ReadOnlyCache in that mode), like every
other channel into the tree -- DRY_RUN_WRITES_NOTHING's rule that a module
working with a cache takes the OBJECT, never a path and never a flag.
`cache=None` is "no memo at all", the same thing the HTTP clients read it
as; it is what a unit test with no tree sees.

WHERE it sits: around the embedder, not inside frontier.vectors_for(). The
embedder is the seam that costs money, its contract is exactly
list[str] -> list[list[float]], and the crawl already binds it once from
the model (frontier.bound_embedder). So the memo is a decorator of
that callable and the traversal does not learn a new argument.
"""
from __future__ import annotations

import hashlib
import json
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def entry_name(model: str, text: str) -> str:
    """The file name one (model, text) pair is cached under.

    The digest is of the text alone and the model is a readable prefix, so
    the directory can be read by a human ("whose vectors are these") and
    swept per model, while collisions between two candidates remain a
    sha256 question. The prefix is sanitised because a model name is an
    ollama tag (`bge-m3`, but also `nomic-embed-text:v1.5`) and this is a
    path component.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{_UNSAFE.sub('_', model)}-{digest}.json"


class VectorMemo:
    """Read/write access to the vectors already computed for one model.

    A miss -- absent entry, unreadable entry, anything that is not a list
    of numbers -- is a miss, never an exception and never a partial vector:
    the answer to a miss is to embed the text, which is always available.
    """

    def __init__(self, cache, model: str):
        self.cache = cache
        self.model = model
        self.hits = 0

    def get(self, text: str) -> list[float] | None:
        if self.cache is None:
            return None
        body = self.cache.read(entry_name(self.model, text))
        if body is None:
            return None
        try:
            vector = json.loads(body)
        except ValueError:
            return None
        if not isinstance(vector, list) or not all(
                isinstance(value, (int, float)) for value in vector):
            return None
        self.hits += 1
        return [float(value) for value in vector]

    def put(self, text: str, vector: list[float]) -> None:
        if self.cache is not None:
            self.cache.write(entry_name(self.model, text), json.dumps(vector))


def memoizing_embedder(embed, memo: VectorMemo):
    """`embed` with the memo in front of it, same contract as `embed`.

    Vectors come back in input order whichever side answered, and only the
    texts nobody has answered for reach `embed` -- one request for the
    misses of a batch rather than one per batch. Duplicate texts inside one
    call are asked for once: at a depth-2 level the same candidate is
    reached through several frontier nodes.
    """
    def embed_with_memo(texts: list[str]) -> list[list[float]]:
        known: dict[str, list[float]] = {}
        missing: list[str] = []
        pending: set[str] = set()
        for text in texts:
            if text in known or text in pending:
                continue
            cached = memo.get(text)
            if cached is None:
                missing.append(text)
                pending.add(text)
            else:
                known[text] = cached
        if missing:
            for text, vector in zip(missing, embed(missing)):
                known[text] = vector
                memo.put(text, vector)
        return [known[text] for text in texts]

    return embed_with_memo
