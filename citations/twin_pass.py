#!/usr/bin/env python3
"""The `--merge-twins` pass: promote translations of our own works.

Runs against what is ALREADY in citation.work, not against the network, so
it is cheap, idempotent and rerunnable after every crawl. Three steps:

1. read each seed's Math-Net id (from corpus.documents.source_url) and its
   cached title list (citation.work.external_ids -> titles, filled at seeding
   from the Math-Net page: Russian and English at once);
2. build both indexes (twins.py) and test every external-skeleton node:
   DOI suffix first, normalized title + year second;
3. promote a match to kind='our-document' with the seed's document_id and
   evidence.twin_of, and journal it.

Edges are untouched by design. The translation really is cited by whoever
cites it; collapsing those edges into the original would destroy attested
facts to tidy up identity. What changes is only what the node CLAIMS to be,
which is what `pg_graph.py candidates` filters on.

A seed with no cached titles contributes nothing rather than falling back to
the OpenAlex display name alone: one-language matching is what run 85 already
measured as the source of cross-language misses.
"""
from __future__ import annotations

import json

from pg_common import copy_csv_into, run_sql, scalar

from . import journal, twins
from .store import STEP_COLUMNS, csv_rows


def corpus_years(env) -> dict[str, list[int]]:
    """document_id -> the years its titles may carry.

    Both the original's year and the translation's: Math-Net prints both in
    the citation line, and the crawl cached them beside the titles.
    """
    out = run_sql(
        env,
        "SELECT document_id, coalesce(external_ids->>'years', '[]') "
        "FROM citation.work WHERE kind = 'our-document' AND document_id IS NOT NULL;",
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    years: dict[str, list[int]] = {}
    for line in out.split("\n"):
        if not line:
            continue
        document_id, _, raw = line.partition("\x1f")
        try:
            years[document_id] = [int(y) for y in json.loads(raw)]
        except (ValueError, TypeError):
            years[document_id] = []
    return years


def seed_titles(env) -> list[dict]:
    """[{key, document_id, titles, mathnet_id}] for every seed.

    The Math-Net id comes from corpus.documents.source_url, not from a name
    convention over document ids: the URL is data the base already holds, and
    the five documents Math-Net does not carry simply have none.
    """
    out = run_sql(
        env,
        "SELECT w.key, w.document_id, coalesce(w.external_ids->>'titles', '[]'), "
        "CASE WHEN d.source_url LIKE '%mathnet.ru%' "
        "     THEN lower(regexp_replace(d.source_url, '^.*/', '')) ELSE '' END "
        "FROM citation.work w JOIN corpus.documents d ON d.id = w.document_id "
        "WHERE w.kind = 'our-document' AND w.document_id IS NOT NULL ORDER BY w.key;",
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    seeds = []
    for line in out.split("\n"):
        if not line:
            continue
        parts = (line.split("\x1f") + ["", "", ""])[:4]
        key, document_id, raw, identifier = parts
        try:
            titles = [t for t in json.loads(raw) if t]
        except (ValueError, TypeError):
            titles = []
        if titles or identifier:
            seeds.append({"key": key, "document_id": document_id,
                          "titles": titles, "mathnet_id": identifier})
    return seeds


def skeleton_nodes(env):
    """(key, title, year, doi) for every node still claiming to be a stranger's."""
    out = run_sql(
        env,
        "SELECT key, coalesce(title, ''), coalesce(year::text, ''), coalesce(doi, '') "
        "FROM citation.work WHERE kind = 'external-skeleton' ORDER BY key;",
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    rows = []
    for line in out.split("\n"):
        if not line:
            continue
        key, title, year, doi = (line.split("\x1f") + ["", "", ""])[:4]
        rows.append((key, title, int(year) if year.isdigit() else None, doi))
    return rows


def promote(env, key: str, document_id: str, seed_key: str, rule: str) -> None:
    """Idempotent: rerunning on an already-promoted node changes nothing."""
    run_sql(
        env,
        "UPDATE citation.work SET kind = 'our-document', document_id = :'doc', "
        "evidence = coalesce(evidence, '{}'::jsonb) || jsonb_build_object("
        "  'twin_of', :'seed', 'twin_rule', :'rule') "
        "WHERE key = :'key';",
        variables={"doc": document_id, "seed": seed_key, "key": key, "rule": rule},
    )


def merge_twins(env, crawl_id: str, *, dry_run: bool = False) -> list[dict]:
    """[{key, title, document_id, seed_key}] for every node promoted."""
    seeds = seed_titles(env)
    if not seeds:
        raise RuntimeError(
            "ни у одного семени нет ни названий, ни mathnet-id; посев "
            "выполнялся до правила двойников — перезапустите посев"
        )
    title_index = twins.build_index(seeds)
    mathnet_index = twins.build_mathnet_index(seeds)
    years = corpus_years(env)
    merged = []
    for key, title, year, doi in skeleton_nodes(env):
        hit = twins.find_twin(title, year, doi, title_index, mathnet_index, years)
        if hit is None:
            continue
        document_id, seed_key, rule = hit
        merged.append({"key": key, "title": title, "rule": rule,
                       "document_id": document_id, "seed_key": seed_key})
        if not dry_run:
            promote(env, key, document_id, seed_key, rule)
    if merged and not dry_run:
        copy_csv_into(
            env, f"citation.crawl_step ({', '.join(STEP_COLUMNS)})",
            csv_rows([[journal.twin(crawl_id, m["key"], m["document_id"],
                                    m["seed_key"]).get(c) for c in STEP_COLUMNS]
                      for m in merged]),
        )
    return merged


def kind_counts(env) -> str:
    return scalar(
        env,
        "SELECT string_agg(kind || '=' || n, ' ' ORDER BY kind) FROM "
        "(SELECT kind, count(*) n FROM citation.work GROUP BY kind) t;",
    )
