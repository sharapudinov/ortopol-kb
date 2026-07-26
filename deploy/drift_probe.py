#!/usr/bin/env python3
"""Repeat-measurement mode backing smoke_checks.VECTOR_PROBE_RANK_TOLERANCE
and VECTOR_PROBE_DISTANCE_TOLERANCE.

Both constants used to be justified only by a comment quoting one-off
manual observations, with no command anyone could re-run to reproduce
them. measure_drift() re-embeds the SAME probe query `n` times against a
target ollama and re-ranks the SAME fixed reference (document_id,
page_number) every time -- exactly the quantity those two tolerances
bound -- and reports how far each repeat strayed from the reference.

Two ways to run it:
  - smoke_test.py --measure-drift N, while a deployed kb-smoke stack is up:
    uses the artifact's own manifest["vector_probe"] as the fixed
    reference (built on the live instance), so this measures the actual
    cross-process comparison check_vector guards -- live-built reference
    vs. a freshly deployed kb-smoke's ollama.
  - directly, `python3 drift_probe.py --pgenv FILE --ollama-url URL -n N`:
    establishes its own reference via pg_rank_probe.nearest_page against
    whichever instance --pgenv/--ollama-url point at (the live instance,
    or a kb-smoke stack reached the same way smoke_test.py's own psql_env
    does), then repeats against that same instance -- measures same-
    process/same-hardware repeat variance in isolation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import pg_rank_probe  # noqa: E402
import pg_search  # noqa: E402
from manifest_contract import VECTOR_PROBE_QUERY as DEFAULT_QUERY  # noqa: E402
from manifest_contract import Key  # noqa: E402
from pg_common import PostgresUnavailable, load_pgenv  # noqa: E402

# DEFAULT_QUERY is manifest_contract.VECTOR_PROBE_QUERY under a name that
# reads right as this standalone CLI's own --query default (`python3
# drift_probe.py --pgenv ...` needs no manifest.json at all, only a running
# instance) -- previously this was a second, hand-copied literal kept in
# sync with manifest_probe.py's copy only by a comment.


def measure_drift(env: dict, ollama_url: str, probe: dict, n: int) -> dict:
    """probe: a manifest["vector_probe"]-shaped dict (query, document_id,
    page_number, rank, distance) -- the SAME fixed reference smoke_checks.
    check_vector compares a single fresh page_rank() against. Runs n
    independent embed+page_rank() round trips against ollama_url and
    reports how far each one strayed from that fixed reference.

    The model/dims pair is resolved via pg_search.resolve_model() ONCE,
    before the loop, not once per repeat: corpus.embedding_model carries
    exactly one CHECK-constrained row, so the value cannot change mid-run,
    and re-resolving it n times previously cost n redundant psql spawns
    interleaved with the actual measurement (8-10 with the CLI defaults).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    resolved = pg_search.resolve_model(env)
    if resolved is None:
        raise RuntimeError("corpus.embedding_model is empty -- run pg_embed.py first")
    model, dims = resolved
    rank_shifts: list[int] = []
    distance_deltas: list[float] = []
    for _ in range(n):
        vec = pg_search.embed_with(model, dims, probe[Key.QUERY], ollama_url)
        if vec is None:
            raise RuntimeError(f"embedding service at {ollama_url} unreachable")
        hit = pg_rank_probe.page_rank(env, vec, probe[Key.DOCUMENT_ID], probe[Key.PAGE_NUMBER])
        if hit is None:
            raise RuntimeError(
                f"{probe[Key.DOCUMENT_ID]} p.{probe[Key.PAGE_NUMBER]}: no longer embedded"
            )
        rank_shifts.append(hit["rank"] - probe[Key.RANK])
        distance_deltas.append(abs(hit["distance"] - probe[Key.DISTANCE]))
    return {
        "n": n,
        "rank_shifts": rank_shifts,
        "distance_deltas": distance_deltas,
        # Magnitude, matching distance_deltas -- not max(rank_shifts):
        # rank_shifts keeps its sign (a page can rank BETTER than the
        # reference, not just worse), and today's reference always happens
        # to be rank 1 (nearest_page's own contract), which floors every
        # possible shift at 0 and hides this. Against a rank > 1 reference,
        # an all-negative set of shifts would make max(rank_shifts)
        # negative -- printed as "less drift than none", the opposite of
        # what a worst-case bound must report.
        "max_rank_shift": max(abs(s) for s in rank_shifts),
        # One-sided companion, kept separately: the worst-case WORSENING
        # specifically (rank grew, i.e. moved further from the top), signed
        # so "no repeat got worse than the reference" is visible as a
        # non-positive number rather than folded into the magnitude above.
        "max_rank_increase": max(rank_shifts),
        "max_distance_delta": max(distance_deltas),
    }


def establish_reference(env: dict, ollama_url: str, query: str) -> dict:
    """One embed_query()+nearest_page() call, shaped exactly like
    manifest["vector_probe"] (minus the "query" key's provenance -- this IS
    the query), so measure_drift() can treat a freshly-established
    reference the same as a manifest-recorded one.
    """
    vec = pg_search.embed_query(query, env, ollama_url=ollama_url)
    if vec is None:
        raise RuntimeError(f"embedding service at {ollama_url} unreachable")
    nearest = pg_rank_probe.nearest_page(env, vec)
    if nearest is None:
        raise RuntimeError("corpus.pages has no embedded rows -- cannot establish a reference")
    return {Key.QUERY: query, **nearest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgenv", type=Path, required=True,
                         help="credentials for the instance to measure against "
                              "(the live corpus/.pgenv, or a kb-smoke-style env)")
    parser.add_argument("--ollama-url", default=pg_search.OLLAMA_URL,
                         help="the /api/embed endpoint of that same instance's ollama")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("-n", "--repeats", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        env = load_pgenv(args.pgenv)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    reference = establish_reference(env, args.ollama_url, args.query)
    print(f"reference: {reference['document_id']} p.{reference['page_number']} "
          f"rank={reference['rank']} distance={reference['distance']:.6f}")
    report = measure_drift(env, args.ollama_url, reference, args.repeats)
    print(f"{report['n']} repeats: rank_shifts={report['rank_shifts']}, "
          f"distance_deltas={[f'{d:.6f}' for d in report['distance_deltas']]}")
    print(f"max rank shift={report['max_rank_shift']}, "
          f"max |Δdistance|={report['max_distance_delta']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
