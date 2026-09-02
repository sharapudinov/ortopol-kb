"""Completeness checks for the citation graph (schema `citation`, see
pg_schema_citation.sql).

Split out of corpus_completeness.py the same way external_checks.py is: the
corpus proper is checked against pdfinfo/INDEX.md, and this checks a
different structure (a graph, not a file tree) against a different set of
invariants. corpus_completeness.py calls citation_problems(env) exactly like
external_problems(theory_dir, env).

What it refuses, and why:

- an ISU corpus document with no citation.work(kind='our-document') row AND
  no citation.crawl_step(action='seed-missing') row -- one of the two is the
  crawl's OWN record of what happened to that document (found and placed, or
  looked for and not found); neither existing means nobody ever decided,
  which is a hole in the journal crawl_step exists to prevent (see
  pg_schema_citation.sql's own comment on that table);
- an external-skeleton or indexed work with no evidence -- evidence is the
  only thing standing between a crawl decision and an unverifiable claim
  (kind, title, exclusion_reason all trace back to it);
- a self-loop in citation.cites -- protected by a CHECK constraint already,
  checked again here as a predicate rather than trusted to still hold on
  whatever database this is asked about;
- a stale projection -- citation_graph must be a faithful copy of
  citation.work/cites (pg_graph_common.py owns the comparison logic, reused
  here rather than reimplemented, per its compare_counts's own docstring);
- a work with a non-empty title and no embedding -- pg_embed.py's own
  contract ("every result record carries a semantic key", EXTENDING.md § 2)
  applied to the 'works' TARGETS entry;
- an 'indexed' work (promoted after being read in full, see
  pg_schema_citation.sql's kind comment) with no corresponding
  corpus.documents row under theory/external -- 'indexed' means we read the
  work, and reading it is exactly what the external-literature pipeline
  records (EXTENDING.md Procedure B).

A missing schema is reported as a single problem, not silently skipped: the
citation graph is part of the knowledge base, not an optional
extra a completeness run can be indifferent to.
"""
from __future__ import annotations

import json
from typing import NamedTuple

import pg_graph_common
from citation_vocab import CrawlAction, WorkKind
from paths import EXTERNAL_SOURCE_DIR, IIS_SOURCE_DIR
from pg_common import scalar
from pg_graph_common import citation_schema_exists, kind_counts_expression


_UNPLACED_SQL = f"""
SELECT d.id
FROM corpus.documents d
WHERE d.source_dir = '{IIS_SOURCE_DIR}' AND d.extraction_state <> 'metadata'
  AND NOT EXISTS (
      SELECT 1 FROM citation.work w WHERE w.kind = '{WorkKind.OUR_DOCUMENT}'
        AND w.document_id = d.id)
  AND NOT EXISTS (
      SELECT 1 FROM citation.crawl_step c
      WHERE c.action = '{CrawlAction.SEED_MISSING}' AND c.frontier_key = d.id)
ORDER BY d.id;
"""


_NO_EVIDENCE_SQL = f"""
SELECT key, kind FROM citation.work
WHERE kind IN ({", ".join(f"'{kind}'" for kind in WorkKind.NEED_EVIDENCE)})
  AND evidence IS NULL
ORDER BY key;
"""


_SELF_LOOP_SQL = "SELECT DISTINCT citing FROM citation.cites WHERE citing = cited ORDER BY citing;"


# Titled works with no vector -- the crawl writes one the moment it scores a
# candidate, so these are rows it never scored: written before the column
# existed, or edited by hand since. `python3 pg_embed.py works` fills them
# with the SAME text rule and the SAME model (pg_embedding_text,
# corpus.embedding_model), which is what makes the top-up safe to run beside
# the crawl at all.
_NO_SEMANTIC_KEY_SQL = """
SELECT key FROM citation.work
WHERE embedding IS NULL AND btrim(coalesce(title, '')) <> ''
ORDER BY key;
"""


_INDEXED_WITHOUT_EXTERNAL_SQL = f"""
SELECT w.key FROM citation.work w
WHERE w.kind = '{WorkKind.INDEXED}'
  AND (w.document_id IS NULL OR NOT EXISTS (
        SELECT 1 FROM corpus.documents d
        WHERE d.id = w.document_id AND d.source_dir = '{EXTERNAL_SOURCE_DIR}'))
ORDER BY w.key;
"""


# The five problem queries and the kind census, as ONE psql script.
#
# Each is independent, read-only and answers in a handful of rows, and each
# used to cost a psql fork, a temp script and a connection of its own --
# the price pg_graph_common._READING_SQL was collapsed for ("five of them
# for 13 ms of work was five process startups"). A completeness run made
# nine such trips; it now makes three readings: the schema question (a
# script naming citation.work must not be the thing that discovers the
# schema is absent), projection_diff(), and this.
#
# json_build_object rather than result blocks separated by RECORD_SEP: one
# statement, one row, and nesting that survives a title carrying a newline
# or a separator byte, which json escapes. A second result set in the same
# script, by contrast, is indistinguishable from another row of the first.
#
# The columns each check is read back by are named beside its SQL and
# projected ::text: a row comes back as an array in that order, so no
# reader has to know a field name, a bigint id reads the way psql's own
# -t -A printed it, and a column renamed in the SQL fails loudly here
# rather than silently returning nothing. json_agg's own ORDER BY, not the
# subquery's -- an aggregate does not inherit its input's ordering.
_CHECKS = (
    ("unplaced", _UNPLACED_SQL, ("id",)),
    ("no_evidence", _NO_EVIDENCE_SQL, ("key", "kind")),
    ("self_loops", _SELF_LOOP_SQL, ("citing",)),
    ("no_semantic_key", _NO_SEMANTIC_KEY_SQL, ("key",)),
    ("indexed_without_external", _INDEXED_WITHOUT_EXTERNAL_SQL, ("key",)),
)


def _rows_json(sql: str, columns: tuple[str, ...]) -> str:
    """`sql`'s rows as a json array of arrays -- [] when it matches nothing."""
    projection = ", ".join(f"r.{column}::text" for column in columns)
    return (f"(SELECT coalesce(json_agg(json_build_array({projection}) "
            f"ORDER BY r.{columns[0]}), '[]'::json) "
            f"FROM ({sql.strip().rstrip(';')}) r)")


_READING_SQL = "SELECT json_build_object(\n" + ",\n".join(
    [f"  'by_kind', {kind_counts_expression()}"]
    + [f"  '{name}', {_rows_json(sql, columns)}" for name, sql, columns in _CHECKS]
) + ");"


class CitationReading(NamedTuple):
    """Everything the citation TABLES are asked during a completeness run.

    One answer, because the questions are independent of each other but not
    of the run: they are all asked, always, and asking them separately buys
    nothing but process startups.
    """

    unplaced: list[str]
    no_evidence: list[tuple[str, str]]
    self_loops: list[str]
    no_semantic_key: list[str]
    indexed_without_external: list[str]
    by_kind: dict[str, int]


def citation_reading(env: dict) -> CitationReading:
    """The one reading, parsed. Assumes schema citation exists."""
    seen = json.loads(scalar(env, _READING_SQL))
    return CitationReading(
        unplaced=[row[0] for row in seen["unplaced"]],
        no_evidence=[(row[0], row[1]) for row in seen["no_evidence"]],
        self_loops=[row[0] for row in seen["self_loops"]],
        no_semantic_key=[row[0] for row in seen["no_semantic_key"]],
        indexed_without_external=[row[0] for row in seen["indexed_without_external"]],
        by_kind=seen["by_kind"],
    )


class CitationState(NamedTuple):
    """What a completeness run needs to know about the citation graph:
    everything wrong with it, and the one line of опись that describes it.

    Both together, because they are answers to the same reading. The
    summary used to re-ask whether the schema exists, re-run the kind
    census and count citation.cites a second time -- and the projection
    reading the problems had just made already carries the work and cites
    totals (pg_graph_common's own docstring prices a round trip: a psql
    fork, a temp script and a connection).
    """

    problems: list[str]
    summary: str


def citation_state(env: dict) -> CitationState:
    """Three readings, and no more: the schema question, the projection,
    and everything the tables are asked (citation_reading).
    """
    if not citation_schema_exists(env):
        return CitationState(
            ["CITATION SCHEMA MISSING: the citation graph is part of the knowledge "
             "base, not optional (python3 pg_graph.py init)"],
            "citation: schema absent")
    seen = pg_graph_common.projection_diff(env)
    read = citation_reading(env)
    return CitationState(_problems(read, seen), _summary(read, seen))


def citation_problems(env: dict) -> list[str]:
    """The problems alone, for a caller that wants no опись line."""
    return citation_state(env).problems


def _summary(read: CitationReading, seen) -> str:
    """One line for corpus_completeness.py's опись report -- purely
    informational, never a source of pass/fail (that is the problems).

    Both halves come from readings already made: the census travels in the
    same answer as the problems, and the totals beside it come from the
    projection reading rather than from a count of their own.
    """
    kinds = ", ".join(f"{k}={n}" for k, n in sorted(read.by_kind.items()))
    if seen is None:
        return f"citation: {sum(read.by_kind.values())} work ({kinds}), проекции нет"
    return f"citation: {seen.work_n} work ({kinds}), {seen.cites_n} cites"


def _projection_stale(seen) -> list[str]:
    """The same reading `pg_graph.py project --check` makes, rendered as
    completeness problems -- pg_graph_common.projection_diff() owns the
    reading, this owns only the wording. The reading itself is made once by
    citation_state() and passed in: it also carries the work/cites totals
    the summary line needs.
    """
    if seen is None:
        return ["PROJECTION STALE: citation_graph is not projected "
                "(python3 pg_graph.py project)"]
    return [f"PROJECTION STALE: {fault}"
            for fault in pg_graph_common.projection_faults(seen)]


def _problems(read: CitationReading, seen) -> list[str]:
    problems: list[str] = []
    problems += [
        f"UNPLACED DOCUMENT: {doc_id} -- neither "
        f"citation.work(kind='{WorkKind.OUR_DOCUMENT}') nor "
        f"citation.crawl_step(action='{CrawlAction.SEED_MISSING}') accounts for it"
        for doc_id in read.unplaced
    ]
    problems += [
        f"NO EVIDENCE: citation.work {key!r} (kind={kind}) has no evidence"
        for key, kind in read.no_evidence
    ]
    problems += [
        f"SELF LOOP: citation.cites has an edge {work_id} -> {work_id} "
        "(violates CHECK citing <> cited -- manual integrity check needed)"
        for work_id in read.self_loops
    ]
    problems += _projection_stale(seen)
    problems += [
        f"NO SEMANTIC KEY: citation.work {key!r} has a non-empty title and no embedding"
        for key in read.no_semantic_key
    ]
    problems += [
        f"INDEXED WITHOUT EXTERNAL DOCUMENT: citation.work {key!r} "
        f"(kind='{WorkKind.INDEXED}') has "
        f"no corpus.documents row under source_dir='{EXTERNAL_SOURCE_DIR}'"
        for key in read.indexed_without_external
    ]
    return problems
