"""Shared knowledge of ollama's /api/tags contract.

build_package.py's manifest_probe RECORDS the digest of the model an ollama
instance is currently serving; smoke_checks.py later hard-ASSERTS a smoke
stack's ollama serves that exact same digest, not merely a model with the
same name/dims (see check_embedding_model_digest's docstring). Those two
sides used to each re-derive the /api/tags URL from an /api/embed URL and
re-implement ollama's tag-naming/matching rules independently -- exactly
the pair of call sites that must never drift apart, since one writes the
contract and the other enforces it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def tags_url(ollama_url: str) -> str:
    """The /api/tags sibling of an /api/embed (or other /api/*) endpoint."""
    return ollama_url.rsplit("/api/", 1)[0] + "/api/tags"


def normalize_model_tag(model: str) -> str:
    """name[:tag] -- appends the default tag 'latest' ONLY when `model` has
    none of its own.

    corpus.embedding_model.model is not schema-constrained to be tag-less
    (nothing in pg_schema.sql or pg_embed.py requires it): the moment a
    deployment pins a specific tag/quantisation (e.g. "bge-m3:q8_0"),
    unconditionally appending ":latest" would turn that into the
    self-contradictory "bge-m3:q8_0:latest", which can never match anything
    ollama actually serves.
    """
    return model if ":" in model else f"{model}:latest"


def served_model_digest(ollama_url: str, model: str) -> tuple[str, int]:
    """digest, size_bytes of `model` as currently served by the ollama
    instance behind ollama_url.

    A registry tag ("bge-m3:latest") is mutable and gives no guarantee two
    deploys ran the same model build; the digest does.

    Raises RuntimeError -- never anything else -- on any transport failure,
    a malformed/non-JSON response, a response shaped unlike ollama's own
    /api/tags contract, or the model not being found in it. Every caller
    (build_package.py, smoke_checks.check_embedding_model_digest) treats
    this as the single failure mode they need to handle; letting a
    transient bad response surface as a bare json.JSONDecodeError/KeyError/
    TypeError instead would crash past smoke_checks.py's whole (ok, detail)
    contract (see that module's docstring) for a condition no more fatal
    than "could not query the URL".
    """
    url = tags_url(ollama_url)
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not query {url} for {model} digest: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} returned a malformed response (not a JSON object): {payload!r}")
    wanted_name = normalize_model_tag(model)
    for entry in payload.get("models", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == wanted_name or entry.get("model") == wanted_name:
            try:
                return entry["digest"], int(entry["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{url} entry for {wanted_name} is missing digest/size or "
                    f"has the wrong shape: {entry!r}"
                ) from exc
    raise RuntimeError(f"{wanted_name} not found in {url} response")
