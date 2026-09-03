"""Exact-phrase full-text search over the corpus, via Postgres tsvector.

Deliberately uses the 'russian' text-search configuration, not 'simple':
'simple' (Postgres's no-op config, equivalent to SQLite FTS5's default
unicode61 tokenizer) does no stemming at all, so a query for "повторных"
would miss a document containing only "повторные". 'russian' runs the
Snowball Russian stemmer and stopword list, so morphological variants
collapse to the same lexeme — verified directly against this instance:
    to_tsvector('russian','повторные средние')
        @@ to_tsquery('russian','повторных & средних')  -- true
    (same query under 'simple' config)                   -- false

phraseto_tsquery() (not plainto_tsquery/to_tsquery) is used so a match
requires the lexemes to be *adjacent*, using the position data tsvector
keeps by default — i.e. an actual phrase match, not just "all these words
appear somewhere on the page".

Usage:
    python3 pg_search.py "повторные средние" [--corpus-dir DIR] [--pgenv FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pg_common import (
    FIELD_SEP, PostgresUnavailable, RECORD_SEP, load_pgenv, row_or_none, run_sql,
)

# paths.py is repo-specific (walks up looking for a theory/iis/ data tree,
# meaningless outside a checkout) and is deliberately NOT bundled into the self-
# contained deploy artifact (see deploy/artifact_bundle.py's
# CORPUS_LIB_FILES). Every function above main() is imported by that
# artifact's smoke_checks.py/manifest_probe.py, which never touch
# paths.py at all -- only this module's own CLI entrypoint does, so the
# import is deferred there (with a local guard) instead of failing the
# whole module at import time on a standalone deploy.

OLLAMA_URL = "http://127.0.0.1:5471/api/embed"
# How many texts go into one ollama request when a caller has many (see
# embed_batch): the crawl's levels are thousands of candidates, and one
# request per candidate is one round trip per candidate.
EMBED_BATCH = 16
TS_CONFIG = "russian"


_SEARCH_SQL = """
SELECT d.id, p.page_number, left(p.body, 200)
FROM corpus.pages p
JOIN corpus.documents d ON d.id = p.document_id
WHERE p.tsv @@ phraseto_tsquery(:'cfg', :'q')
ORDER BY d.id, p.page_number
LIMIT :lim;
"""

# Vector search answers the question full-text cannot: a page that shares no word
# with the query but does share its meaning. Cosine distance, HNSW index.
_VECTOR_SQL = """
SELECT d.id, p.page_number, left(p.body, 200)
FROM corpus.pages p
JOIN corpus.documents d ON d.id = p.document_id
WHERE p.embedding IS NOT NULL
ORDER BY p.embedding <=> :'vec'::vector
LIMIT :lim;
"""

# Hybrid: reciprocal rank fusion. Neither score is comparable to the other —
# ts_rank and cosine distance live on different scales — so fuse by RANK, which
# is scale-free. A page that both lists rate highly wins; a page only one of them
# finds still surfaces. RRF_K damps the top of each list so a single confident
# ranker cannot monopolise the result.
RRF_K = 60
_HYBRID_SQL = """
WITH ft AS (
    SELECT p.id, row_number() OVER (ORDER BY ts_rank(p.tsv, q.query) DESC) AS rnk
    FROM corpus.pages p, phraseto_tsquery(:'cfg', :'q') AS q(query)
    WHERE p.tsv @@ q.query
    LIMIT 100
), vec AS (
    SELECT p.id, row_number() OVER (ORDER BY p.embedding <=> :'vec'::vector) AS rnk
    FROM corpus.pages p
    WHERE p.embedding IS NOT NULL
    LIMIT 100
)
SELECT d.id, p.page_number, left(p.body, 200)
FROM corpus.pages p
JOIN corpus.documents d ON d.id = p.document_id
LEFT JOIN ft  ON ft.id  = p.id
LEFT JOIN vec ON vec.id = p.id
WHERE ft.id IS NOT NULL OR vec.id IS NOT NULL
ORDER BY coalesce(1.0 / ({k} + ft.rnk), 0) + coalesce(1.0 / ({k} + vec.rnk), 0) DESC
LIMIT :lim;
""".replace("{k}", str(RRF_K))


def resolve_model(env: dict[str, str]) -> tuple[str, int] | None:
    """The (model, dims) pair from corpus.embedding_model -- the read
    embed_query() does on every one-off call.

    Split out so a caller that embeds many queries against the SAME
    instance in a loop (drift_probe.measure_drift) can look this up ONCE
    instead of paying one extra psql round trip per repeat for a value the
    schema guarantees cannot change mid-run (corpus.embedding_model carries
    exactly one row, CHECK (id = 1)).

    None when the table is empty, mirroring embed_query()'s own contract.
    """
    row = row_or_none(
        env, "SELECT model, dims FROM corpus.embedding_model WHERE id = 1;",
        variables=None, expected_columns=2, what="resolve_model: model/dims lookup",
    )
    if row is None:
        return None
    model, dims = row
    return model, int(dims)


def embed_with(model: str, dims: int, query: str, ollama_url: str = OLLAMA_URL) -> str | None:
    """Embeds `query` against ollama_url using an already-resolved (model,
    dims) pair -- see resolve_model(). embed_query() is the convenience
    wrapper that resolves the pair itself for a one-off caller; a caller
    embedding many queries against the same instance should call
    resolve_model() once and this n times instead.

    Returns None when the embedding service is unreachable, so callers can
    fall back to full-text rather than fail: full-text needs no model.
    """
    payload = json.dumps({"model": model, "input": query}).encode()
    req = urllib.request.Request(ollama_url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            vec = json.load(resp)["embeddings"][0]
    except (urllib.error.URLError, OSError, KeyError, IndexError):
        return None
    if len(vec) != dims:
        raise ValueError(
            f"{model} вернула {len(vec)} измерений, в базе записано {dims}: "
            "модель сменилась, векторы надо пересчитать (pg_embed.py)")
    return json.dumps(vec)


def embed_batch(model: str, dims: int, texts: list[str],
                ollama_url: str = OLLAMA_URL, batch: int = EMBED_BATCH,
                opener=urllib.request.urlopen) -> list[list[float]]:
    """Many texts through the same seam embed_with() puts one query through,
    a batch of `batch` per request, vectors returned in input order.

    Two contracts differ from embed_with() on purpose, and both belong to
    the caller that embeds a whole crawl level rather than one query:

    - an unreachable service RAISES instead of returning None. A search can
      fall back to full-text, which needs no model; a crawl that silently
      scored nothing would write a level of dropped candidates.
    - the count is checked as well as the width. ollama answering fewer
      vectors than texts would otherwise pair each text with the NEXT text's
      vector, and every score after the gap would be plausible and wrong.
    """
    out: list[list[float]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        payload = json.dumps({"model": model, "input": chunk}).encode()
        request = urllib.request.Request(
            ollama_url, data=payload, headers={"Content-Type": "application/json"})
        with opener(request, timeout=300) as response:
            vectors = json.load(response)["embeddings"]
        if len(vectors) != len(chunk):
            raise RuntimeError(f"ollama вернула {len(vectors)} векторов на {len(chunk)} текстов")
        for vector in vectors:
            if len(vector) != dims:
                raise RuntimeError(f"ожидалось {dims} измерений, пришло {len(vector)}")
        out += vectors
    return out


def embed_query(query: str, env: dict[str, str], ollama_url: str = OLLAMA_URL) -> str | None:
    """Embed the query with THE SAME model that produced the stored vectors.

    Not a preference — a correctness requirement. An embedding is only meaningful
    inside the vector space of the model that produced it; a query vector from a
    different model still yields a perfectly well-formed cosine distance that means
    nothing at all. The failure is silent, which is why the model is recorded in
    corpus.embedding_model and checked here rather than assumed.

    ollama_url is a parameter, not just the module default, so deploy/
    (which talks to a throwaway kb-smoke ollama on a different port, or builds a
    package from a host other than the developer's local one) can point this at
    the right instance instead of duplicating the embed+nearest-neighbour logic.

    Returns None when the embedding service is unreachable (either resolve_model's
    lookup found no row, or embed_with's HTTP call failed), so callers can fall
    back to full-text rather than fail: full-text needs no model. A caller that
    embeds more than once against the same instance should call resolve_model()/
    embed_with() directly instead (see drift_probe.measure_drift).
    """
    resolved = resolve_model(env)
    if resolved is None:
        return None
    model, dims = resolved
    return embed_with(model, dims, query, ollama_url)


def search(query: str, env: dict[str, str], limit: int = 20,
           mode: str = "fulltext") -> list[tuple[str, int, str]]:
    """mode: fulltext | vector | hybrid.

    vector and hybrid need the embedding service; when it is unreachable both
    degrade to fulltext with a note on stderr rather than failing outright.
    """
    variables = {"cfg": TS_CONFIG, "q": query, "lim": str(int(limit))}
    sql = _SEARCH_SQL
    if mode in ("vector", "hybrid"):
        vec = embed_query(query, env)
        if vec is None:
            print(f"эмбеддинги недоступны, режим {mode} -> fulltext", file=sys.stderr)
        else:
            variables["vec"] = vec
            sql = _VECTOR_SQL if mode == "vector" else _HYBRID_SQL
    result = run_sql(
        env,
        sql,
        variables=variables,
        # Snippets are raw PDF text and routinely contain literal newlines,
        # so rows can't be split on "\n" (splitlines()) — psql's -R gives an
        # explicit record separator distinct from any in-cell newline.
        extra_args=["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP],
    )
    rows = []
    for record in result.stdout.split(RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        doc_id, page_number, snippet = record.split(FIELD_SEP, 2)
        rows.append((doc_id, int(page_number), snippet))
    return rows


def main(argv: list[str] | None = None) -> int:
    # paths.py sits right next to this file in a checkout and is deliberately
    # NOT bundled into the deploy artifact (see the module-level comment
    # above) -- main() is itself a checkout-only code path (never invoked
    # from the bundled corpus_lib/pg_search.py copy, nothing there calls
    # __main__), so a local import here, guarded rather than reaching
    # sideways into deploy/ for a helper, keeps the dependency
    # direction one-way: deploy/ imports from the repository root, never the
    # reverse. ImportError: paths.py absent entirely (bundled artifact, in
    # practice unreachable here). RuntimeError: paths.data_root() found no
    # ancestor directory with a theory/iis/ tree -- a plain checkout with
    # no surrounding data tree.
    try:
        from paths import default_corpus_dir
        corpus_dir_default = default_corpus_dir()
    except (ImportError, RuntimeError):
        corpus_dir_default = None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["fulltext", "vector", "hybrid"],
                        default="fulltext")
    parser.add_argument("--corpus-dir", type=Path, default=corpus_dir_default)
    parser.add_argument("--pgenv", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pgenv is not None:
        pgenv_path = args.pgenv
    elif args.corpus_dir is not None:
        pgenv_path = args.corpus_dir / ".pgenv"
    else:
        print("no --pgenv given and no repository context to locate one -- "
              "pass --pgenv <file>", file=sys.stderr)
        return 1

    try:
        env = load_pgenv(pgenv_path)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    hits = search(args.query, env, mode=args.mode)
    if not hits:
        print("no matches")
    for doc_id, page_number, snippet in hits:
        print(f"{doc_id} p.{page_number}: {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
