#!/usr/bin/env python3
"""Where the crawl starts from: the reads that establish a run.

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

import json

from paths import IIS_SOURCE_DIR
from pg_common import run_sql, scalar, sql_literal
from pg_common import FIELD_SEP, ROW_ARGS, split_records

# The measurement that established which OpenAlex/zbMATH record each of our
# documents is (measurements.citation_source_coverage). A run number, i.e. a
# fact about the data, so it belongs beside the read that uses it rather
# than in whichever entry point happens to call that read.
COVERAGE_RUN = 85

# The seed document set, written ONCE. Both readings below are of THIS
# predicate: "what counts as a seed document" changing for the crawl but not
# for its Math-Net title anchor produces seeds the twin rule cannot anchor,
# and nothing reports it.
_SEED_DOCUMENTS_SQL = (
    "SELECT id, coalesce(source_url, '') FROM corpus.documents "
    "WHERE source_dir = :'dir' AND extraction_state <> 'metadata' ORDER BY id;"
)


def corpus_seed_documents(env, source_dir: str = IIS_SOURCE_DIR) -> list[tuple[str, str]]:
    """[(document id, source_url or '')] for the corpus the crawl seeds from.

    source_dir is paths.IIS_SOURCE_DIR by default and never the literal:
    corpus.documents.source_dir carries that exact string, and a second
    spelling of it returns zero rows rather than failing (paths.py's own
    docstring).
    """
    out = run_sql(
        env, _SEED_DOCUMENTS_SQL,
        variables={"dir": source_dir},
        extra_args=ROW_ARGS,
    ).stdout
    rows = []
    for record in split_records(out):
        document_id, _, url = record.partition(FIELD_SEP)
        rows.append((document_id, url))
    return rows


def corpus_document_ids(env, source_dir: str = IIS_SOURCE_DIR) -> list[str]:
    return [document_id for document_id, _url in corpus_seed_documents(env, source_dir)]


def seed_matches(env, run_id: int, source: str) -> dict[str, str]:
    out = run_sql(
        env,
        "SELECT document_id, matched_id FROM measurements.citation_source_coverage "
        "WHERE run_id = :run AND source = :'source' AND matched_id IS NOT NULL "
        "ORDER BY document_id;",
        variables={"run": str(int(run_id)), "source": source},
        extra_args=ROW_ARGS,
    ).stdout
    matches = {}
    for record in split_records(out):
        document_id, _, matched_id = record.partition(FIELD_SEP)
        matches[document_id] = matched_id
    return matches


def stored_zbmath_abstracts(env) -> dict[str, str]:
    """document_id -> the zbMATH abstract citation.work already holds.

    The fallback's own memory, and the reason it is safe to skip the
    request: the abstract is static between runs, and its provenance is
    recorded (evidence.abstract_source, store.PostgresWriter.evidence_of),
    so a stored one can stand in for the fetch WITHOUT the run pretending
    an OpenAlex abstract came from zbMATH. Only rows that say zbmath count
    -- an OpenAlex abstract arrives again with the record and needs no
    fallback at all.

    An abstract is prose with newlines in it, so it travels as JSON rather
    than as separated fields: psql's row separator is a newline and no
    field separator saves a value that contains one (kb/CLAUDE.md, "что
    легко сломать"). json_agg escapes it and the whole answer is one line.
    """
    out = scalar(
        env,
        "SELECT coalesce(json_agg(json_build_array(document_id, abstract)), '[]') "
        "FROM citation.work WHERE document_id IS NOT NULL AND abstract IS NOT NULL "
        "AND evidence->>'abstract_source' = 'zbmath';",
    )
    return {document_id: abstract
            for document_id, abstract in json.loads(out or "[]") if abstract}


def stored_mathnet_titles(env) -> dict[str, tuple[list[str], list[int]]]:
    """document_id -> (titles, years) citation.work already holds.

    The Math-Net anchor's own memory, and the reason it is safe to skip the
    page: the crawl caches both titles of a seed on its work row when it
    first fetched them (store.PostgresWriter, external_ids), and
    twin_pass.seed_titles() reads them back from there for the twin rule.
    A page re-fetched to learn what the database already knows costs a
    request and a 0.6 s pause per seed on the startup path of every crawl.

    Only rows with at least one title count: an empty list is "we asked and
    the page carried no citation", which is a fact the failure counter
    reports, not a value to serve back.

    Titles are third-party prose and travel as JSON for the same reason the
    zbMATH abstracts do -- psql's row separator is a newline and no field
    separator survives a value containing one.
    """
    out = scalar(
        env,
        "SELECT coalesce(json_agg(json_build_array("
        "  document_id, external_ids->'titles', "
        "  coalesce(external_ids->'years', '[]'::jsonb))), '[]') "
        "FROM citation.work WHERE document_id IS NOT NULL "
        "AND jsonb_array_length(coalesce(external_ids->'titles', '[]'::jsonb)) > 0;",
    )
    return {document_id: (titles, years or [])
            for document_id, titles, years in json.loads(out or "[]") if titles}


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
        extra_args=ROW_ARGS,
    ).stdout
    return set(split_records(out))


# How many keys travel in one `key IN (...)` list. The same reason the
# OpenAlex client batches its id filter: the whole list goes into one
# statement, and an unbounded one is a psql command line (and a planner
# input) that grows with the level. A depth-2 level is thousands of
# candidates, so the read is a handful of round trips rather than one
# enormous statement or one statement per key.
KEY_BATCH = 200


def known_embeddings(env, keys) -> dict[str, list[float]]:
    """{key: stored vector} for those of `keys` citation.work already holds.

    What makes a re-crawl (and the calibrate-then-crawl pair, which asks
    OpenAlex the same depth-1 pages twice by design) stop paying ollama for
    vectors already in the database. A miss is simply absent from the
    answer: the caller embeds exactly the misses.

    The vectors are the corpus model's -- store.py writes what the crawl
    embedded, and the crawl binds its embedder from pg_search.resolve_model()
    over corpus.embedding_model, the same read pg_embed.py uses. There is no
    per-row model column to check against, so a re-embedding of the corpus
    under a new model must clear these vectors the way it clears
    corpus.pages.embedding; a stale one here would score against the wrong
    model and produce a plausible number rather than an error.

    pgvector prints a vector as a JSON array of numbers, which is why the
    parse is json.loads and not a hand-written split.
    """
    keys = list(dict.fromkeys(keys))
    out: dict[str, list[float]] = {}
    for start in range(0, len(keys), KEY_BATCH):
        chunk = keys[start:start + KEY_BATCH]
        listed = ", ".join(sql_literal(key) for key in chunk)
        rows = run_sql(
            env,
            "SELECT key, embedding FROM citation.work "
            f"WHERE embedding IS NOT NULL AND key IN ({listed});",
            extra_args=ROW_ARGS,
        ).stdout
        for record in split_records(rows):
            key, _, vector = record.partition(FIELD_SEP)
            out[key] = json.loads(vector)
    return out
