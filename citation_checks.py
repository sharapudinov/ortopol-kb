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

import pg_graph_common
from paths import EXTERNAL_SOURCE_DIR, IIS_SOURCE_DIR
from pg_common import run_sql, scalar

FIELD_SEP = "\x1f"

_SCHEMA_EXISTS_SQL = "SELECT to_regclass('citation.work') IS NOT NULL;"


def citation_schema_exists(env: dict) -> bool:
    return scalar(env, _SCHEMA_EXISTS_SQL) == "t"


_UNPLACED_SQL = f"""
SELECT d.id
FROM corpus.documents d
WHERE d.source_dir = '{IIS_SOURCE_DIR}' AND d.extraction_state <> 'metadata'
  AND NOT EXISTS (
      SELECT 1 FROM citation.work w WHERE w.kind = 'our-document' AND w.document_id = d.id)
  AND NOT EXISTS (
      SELECT 1 FROM citation.crawl_step c
      WHERE c.action = 'seed-missing' AND c.frontier_key = d.id)
ORDER BY d.id;
"""


def unplaced_documents(env: dict) -> list[str]:
    """ISU corpus documents the crawl never recorded a decision about."""
    out = run_sql(env, _UNPLACED_SQL, extra_args=["-t", "-A"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


_NO_EVIDENCE_SQL = """
SELECT key, kind FROM citation.work
WHERE kind IN ('external-skeleton', 'indexed') AND evidence IS NULL
ORDER BY key;
"""


def works_without_evidence(env: dict) -> list[tuple[str, str]]:
    out = run_sql(env, _NO_EVIDENCE_SQL, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    rows = []
    for line in out.splitlines():
        if line.strip():
            key, kind = line.split(FIELD_SEP)
            rows.append((key, kind))
    return rows


_SELF_LOOP_SQL = "SELECT DISTINCT citing FROM citation.cites WHERE citing = cited ORDER BY citing;"


def self_loop_work_ids(env: dict) -> list[str]:
    out = run_sql(env, _SELF_LOOP_SQL, extra_args=["-t", "-A"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _projection_stale(env: dict) -> list[str]:
    if not pg_graph_common.graph_exists(env):
        return ["PROJECTION STALE: citation_graph is not projected "
                "(python3 pg_graph.py project)"]
    work_n = int(scalar(env, "SELECT count(*) FROM citation.work;"))
    cites_n = int(scalar(env, "SELECT count(*) FROM citation.cites;"))
    vertex_n, edge_n = pg_graph_common.graph_counts(env)
    diff_v, diff_e = pg_graph_common.compare_counts(work_n, cites_n, vertex_n, edge_n)
    if diff_v == 0 and diff_e == 0:
        return []
    return [f"PROJECTION STALE: work={work_n} vertices={vertex_n} (diff {diff_v}); "
            f"cites={cites_n} edges={edge_n} (diff {diff_e})"]


_NO_SEMANTIC_KEY_SQL = """
SELECT key FROM citation.work
WHERE embedding IS NULL AND btrim(coalesce(title, '')) <> ''
ORDER BY key;
"""


def works_without_semantic_key(env: dict) -> list[str]:
    out = run_sql(env, _NO_SEMANTIC_KEY_SQL, extra_args=["-t", "-A"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


_INDEXED_WITHOUT_EXTERNAL_SQL = f"""
SELECT w.key FROM citation.work w
WHERE w.kind = 'indexed'
  AND (w.document_id IS NULL OR NOT EXISTS (
        SELECT 1 FROM corpus.documents d
        WHERE d.id = w.document_id AND d.source_dir = '{EXTERNAL_SOURCE_DIR}'))
ORDER BY w.key;
"""


def indexed_without_external_document(env: dict) -> list[str]:
    out = run_sql(env, _INDEXED_WITHOUT_EXTERNAL_SQL, extra_args=["-t", "-A"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


_COUNTS_SQL = "SELECT kind, count(*) FROM citation.work GROUP BY kind ORDER BY kind;"


def counts_by_kind(env: dict) -> dict[str, int]:
    out = run_sql(env, _COUNTS_SQL, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.strip():
            kind, n = line.split(FIELD_SEP)
            counts[kind] = int(n)
    return counts


def citation_summary(env: dict) -> str:
    """One line for corpus_completeness.py's opись report -- purely
    informational, never a source of pass/fail (that is citation_problems).
    """
    if not citation_schema_exists(env):
        return "citation: schema absent"
    by_kind = counts_by_kind(env)
    work_total = sum(by_kind.values())
    cites_total = int(scalar(env, "SELECT count(*) FROM citation.cites;"))
    kinds = ", ".join(f"{k}={n}" for k, n in sorted(by_kind.items()))
    return f"citation: {work_total} work ({kinds}), {cites_total} cites"


def citation_problems(env: dict) -> list[str]:
    if not citation_schema_exists(env):
        return ["CITATION SCHEMA MISSING: the citation graph is part of the knowledge "
                "base, not optional (python3 pg_graph.py init)"]

    problems: list[str] = []
    problems += [
        f"UNPLACED DOCUMENT: {doc_id} -- neither citation.work(kind='our-document') "
        "nor citation.crawl_step(action='seed-missing') accounts for it"
        for doc_id in unplaced_documents(env)
    ]
    problems += [
        f"NO EVIDENCE: citation.work {key!r} (kind={kind}) has no evidence"
        for key, kind in works_without_evidence(env)
    ]
    problems += [
        f"SELF LOOP: citation.cites has an edge {work_id} -> {work_id} "
        "(violates CHECK citing <> cited -- manual integrity check needed)"
        for work_id in self_loop_work_ids(env)
    ]
    problems += _projection_stale(env)
    problems += [
        f"NO SEMANTIC KEY: citation.work {key!r} has a non-empty title and no embedding"
        for key in works_without_semantic_key(env)
    ]
    problems += [
        f"INDEXED WITHOUT EXTERNAL DOCUMENT: citation.work {key!r} (kind='indexed') has "
        f"no corpus.documents row under source_dir='{EXTERNAL_SOURCE_DIR}'"
        for key in indexed_without_external_document(env)
    ]
    return problems
