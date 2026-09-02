#!/usr/bin/env python3
"""Every Postgres write the snowball makes, and the reads it starts from.

Same two mechanisms as the rest of the repository (pg_common.py): script
variables for parameters, `\\copy` from a csv module-built temp file for
bulk -- no string interpolation of source-controlled text anywhere near
SQL, because titles and abstracts here come from a third party.

Upserts, never truncate-and-reload. The rule LOADERS_PRESERVE was paid for
twice in this project, and it has two specific consequences below:

- an existing `our-document` or `indexed` row is NOT demoted to
  `external-skeleton` when the crawl meets the same work again as a
  stranger's citation;
- an embedding survives unless the row arrives with a new one; a re-crawl
  that changed nothing must not send pg_embed.py back over the whole graph.
"""
from __future__ import annotations

import csv
import io
import json

from pg_common import copy_csv_into, run_sql, scalar_row

WORK_COLUMNS = (
    "key", "doi", "title", "abstract", "year", "authors", "external_ids",
    "source", "kind", "document_id", "evidence", "embedding",
)
CITES_COLUMNS = ("citing_key", "cited_key", "source", "evidence")
STEP_COLUMNS = (
    "crawl_id", "depth", "frontier_key", "candidate_key", "action",
    "n_found", "n_kept", "reason",
)

_WORK_STAGE_DDL = """
DROP TABLE IF EXISTS citation.stage_work;
CREATE UNLOGGED TABLE citation.stage_work (
    key TEXT, doi TEXT, title TEXT, abstract TEXT, year INTEGER,
    authors JSONB, external_ids JSONB, source TEXT, kind TEXT,
    document_id TEXT, evidence JSONB, embedding vector(1024));
"""

_WORK_UPSERT = """
INSERT INTO citation.work
    (key, doi, title, abstract, year, authors, external_ids, source, kind,
     document_id, evidence, embedding, fetched_at)
SELECT key, doi, title, abstract, year, authors, external_ids, source, kind,
       document_id, evidence, embedding, now()
FROM citation.stage_work
ON CONFLICT (key) DO UPDATE SET
    doi          = COALESCE(EXCLUDED.doi, citation.work.doi),
    title        = COALESCE(EXCLUDED.title, citation.work.title),
    abstract     = COALESCE(EXCLUDED.abstract, citation.work.abstract),
    year         = COALESCE(EXCLUDED.year, citation.work.year),
    authors      = COALESCE(EXCLUDED.authors, citation.work.authors),
    external_ids = COALESCE(EXCLUDED.external_ids, citation.work.external_ids),
    source       = EXCLUDED.source,
    kind         = CASE WHEN citation.work.kind IN ('our-document', 'indexed')
                         AND EXCLUDED.kind = 'external-skeleton'
                        THEN citation.work.kind ELSE EXCLUDED.kind END,
    document_id  = COALESCE(EXCLUDED.document_id, citation.work.document_id),
    evidence     = EXCLUDED.evidence,
    embedding    = COALESCE(EXCLUDED.embedding, citation.work.embedding),
    fetched_at   = now();
DROP TABLE citation.stage_work;
"""

_CITES_STAGE_DDL = """
DROP TABLE IF EXISTS citation.stage_cites;
CREATE UNLOGGED TABLE citation.stage_cites (
    citing_key TEXT, cited_key TEXT, source TEXT, evidence JSONB);
"""

# a.id <> b.id, not a.key <> b.key: after the twin union two OpenAlex
# records of one work share a node, and a "translation cites original" edge
# would otherwise violate the CHECK (citing <> cited) and abort the batch.
_CITES_UPSERT = """
INSERT INTO citation.cites (citing, cited, source, evidence)
SELECT a.id, b.id, s.source, s.evidence
FROM citation.stage_cites s
JOIN citation.work a ON a.key = s.citing_key
JOIN citation.work b ON b.key = s.cited_key
WHERE a.id <> b.id
ON CONFLICT (citing, cited, source) DO NOTHING;
DROP TABLE citation.stage_cites;
"""


def csv_rows(rows: list[list]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _json_or_null(value):
    return None if value is None else json.dumps(value, ensure_ascii=False)


def vector_literal(vector) -> str | None:
    return None if vector is None else "[" + ",".join(repr(float(v)) for v in vector) + "]"


# -- reads ---------------------------------------------------------------
def embedding_model(env) -> tuple[str, int]:
    model, dims = scalar_row(
        env, "SELECT model, dims FROM corpus.embedding_model WHERE id = 1;",
        expected_columns=2,
    )
    return model, int(dims)


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
    """Keys fetched within `days` -- what --resume declines to re-fetch."""
    out = run_sql(
        env,
        f"SELECT key FROM citation.work WHERE fetched_at > now() - interval '{int(days)} days';",
        extra_args=["-t", "-A"],
    ).stdout.strip()
    return {line for line in out.split("\n") if line}


# -- writes --------------------------------------------------------------
class PostgresWriter:
    """The live-database implementation of what crawl.py needs written."""

    def __init__(self, env, source: str = "openalex"):
        self.env = env
        self.source = source
        self.counts = {"work": 0, "cites": 0, "step": 0}

    def works(self, nodes) -> int:
        rows = []
        for node in nodes:
            rows.append([
                node.key,
                node.doi,
                node.title,
                node.abstract,
                node.year,
                _json_or_null(node.authors or None),
                _json_or_null(node.external_ids()),
                self.source,
                node.kind,
                node.document_id,
                _json_or_null(self.evidence_of(node)),
                vector_literal(getattr(node, "embedding", None)),
            ])
        if not rows:
            return 0
        run_sql(self.env, _WORK_STAGE_DDL)
        copy_csv_into(self.env, f"citation.stage_work ({', '.join(WORK_COLUMNS)})", csv_rows(rows))
        run_sql(self.env, _WORK_UPSERT)
        self.counts["work"] += len(rows)
        return len(rows)

    @staticmethod
    def evidence_of(node) -> dict:
        """The raw source records, plus how the abstract was obtained --
        a zbMATH review standing in for a missing OpenAlex abstract must be
        re-derivable as such without another network call."""
        evidence = {"records": node.records}
        if node.abstract_source:
            evidence["abstract_source"] = node.abstract_source
        if getattr(node, "zbmath_id", None):
            evidence["zbmath_id"] = node.zbmath_id
        if node.relation:
            evidence["relation"] = node.relation
        if node.discovered_from:
            evidence["discovered_from"] = node.discovered_from
        if node.score is not None:
            evidence["frontier_score"] = round(node.score, 6)
        return evidence

    def edges(self, edges) -> int:
        rows = [
            [citing, cited, self.source,
             _json_or_null({"relation": relation, "fetched_from": fetched_from})]
            for citing, cited, relation, fetched_from in edges
        ]
        if not rows:
            return 0
        run_sql(self.env, _CITES_STAGE_DDL)
        copy_csv_into(self.env, f"citation.stage_cites ({', '.join(CITES_COLUMNS)})", csv_rows(rows))
        run_sql(self.env, _CITES_UPSERT)
        self.counts["cites"] += len(rows)
        return len(rows)

    def journal(self, steps) -> int:
        rows = [[s.get(c) for c in STEP_COLUMNS] for s in steps]
        if not rows:
            return 0
        copy_csv_into(self.env, f"citation.crawl_step ({', '.join(STEP_COLUMNS)})", csv_rows(rows))
        self.counts["step"] += len(rows)
        return len(rows)


class DryRunWriter:
    """Collects what a real run would write, and writes nothing."""

    def __init__(self, source: str = "openalex"):
        self.source = source
        self.works_seen, self.edges_seen, self.steps_seen = [], [], []
        self.counts = {"work": 0, "cites": 0, "step": 0}

    def works(self, nodes) -> int:
        self.works_seen += list(nodes)
        self.counts["work"] += len(self.works_seen)
        return len(self.works_seen)

    def edges(self, edges) -> int:
        self.edges_seen += list(edges)
        self.counts["cites"] = len(self.edges_seen)
        return len(self.edges_seen)

    def journal(self, steps) -> int:
        self.steps_seen += list(steps)
        self.counts["step"] = len(self.steps_seen)
        return len(self.steps_seen)
