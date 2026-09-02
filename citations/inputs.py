#!/usr/bin/env python3
"""Where the crawl starts from: the four reads that establish a run.

Split from store.py, which is the WRITE seam (Writer / PostgresWriter /
DryRunWriter) and nothing else. These are the opposite direction and answer
to nobody's --dry-run promise: reading the corpus, the run-85 matches and
what was fetched recently costs the database nothing and writes nowhere, so
a mode that must not write is free to ask them.

Which embedding model produced the stored vectors is NOT among them: that
read is pg_search.resolve_model(), the same one the search and the graph
queries use. Two readings of corpus.embedding_model with two failure
contracts is exactly how a crawl ends up scoring against a model the corpus
was never embedded with.
"""
from __future__ import annotations

from pg_common import run_sql, sql_literal


def corpus_document_ids(env, source_dir: str = "theory/iis") -> list[str]:
    out = run_sql(
        env,
        "SELECT id FROM corpus.documents WHERE source_dir = :'dir' "
        "AND extraction_state <> 'metadata' ORDER BY id;",
        variables={"dir": source_dir},
        extra_args=["-t", "-A"],
    ).stdout.strip()
    return [line for line in out.split("\n") if line]


def seed_matches(env, run_id: int, source: str) -> dict[str, str]:
    out = run_sql(
        env,
        "SELECT document_id, matched_id FROM measurements.citation_source_coverage "
        "WHERE run_id = :run AND source = :'source' AND matched_id IS NOT NULL "
        "ORDER BY document_id;",
        variables={"run": str(int(run_id)), "source": source},
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    matches = {}
    for line in out.split("\n"):
        if not line:
            continue
        document_id, _, matched_id = line.partition("\x1f")
        matches[document_id] = matched_id
    return matches


def fresh_keys(env, days: int) -> set[str]:
    """Keys fetched within `days` -- what --resume declines to re-fetch.

    The interval is the one value here that cannot travel as a psql script
    variable: `interval :'days'` is not valid syntax, and the concatenated
    form would still have to be quoted. sql_literal() is that quoting, in
    the one place the whole repository shares -- not an f-string, which is
    what the module docstring above rules out.
    """
    interval = sql_literal(f"{int(days)} days")
    out = run_sql(
        env,
        f"SELECT key FROM citation.work WHERE fetched_at > now() - interval {interval};",
        extra_args=["-t", "-A"],
    ).stdout.strip()
    return {line for line in out.split("\n") if line}
