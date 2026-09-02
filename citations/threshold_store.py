#!/usr/bin/env python3
"""Writes to `measurements`: the tau calibration and its run row.

Kept out of store.py because it is a different schema with a different
contract. store.py writes the durable citation graph -- upserts, preserved
kinds, preserved embeddings. This writes a research result under EXTENDING
procedure D: idempotent by spike name (the id shifts on every reload, the
name does not), verdict left NULL because the verdict is the orchestrator's,
and one row per scored candidate so the distribution the threshold rests on
is queryable rather than summarised in prose.
"""
from __future__ import annotations

from pg_common import copy_csv_into, run_sql, scalar

from .store import csv_rows

THRESHOLD_DDL = """
CREATE TABLE IF NOT EXISTS measurements.citation_frontier_threshold (
    run_id        BIGINT NOT NULL REFERENCES measurements.run(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    depth         INTEGER NOT NULL,
    relation      TEXT NOT NULL CHECK (relation IN ('cites', 'referenced')),
    score         DOUBLE PRECISION NOT NULL,
    title         TEXT,
    year          INTEGER,
    -- Two columns the brief did not ask for, added because without them the
    -- report's central claim would rest on numbers living outside the base.
    -- has_abstract: OpenAlex carries an abstract for only some works, and a
    -- title-only candidate scores measurably lower -- so a high tau filters
    -- on metadata completeness as much as on relevance, and that has to be
    -- checkable by query. n_references: the price of expanding this node at
    -- the next depth, i.e. what a choice of tau actually costs in requests.
    has_abstract  BOOLEAN,
    n_references  INTEGER,
    PRIMARY KEY (run_id, candidate_key, relation)
);
-- Added after the first calibration, when the distribution turned out to
-- need explaining; ADD COLUMN IF NOT EXISTS rather than a fresh CREATE so an
-- instance that already carries the table gets them too.
ALTER TABLE measurements.citation_frontier_threshold
    ADD COLUMN IF NOT EXISTS has_abstract BOOLEAN,
    ADD COLUMN IF NOT EXISTS n_references INTEGER;
CREATE INDEX IF NOT EXISTS citation_frontier_threshold_score
    ON measurements.citation_frontier_threshold (run_id, score);
"""


# -- run row and per-candidate rows ------------------------------------
def upsert_run(env, spike: str, fields: dict) -> int:
    """Idempotent by spike name, the convention run 85 established: the id
    shifts on every reload, the spike name does not."""
    run_sql(env, "DELETE FROM measurements.run WHERE spike = :'spike';",
            variables={"spike": spike})
    names = ["spike"] + list(fields)
    variables = {"spike": spike}
    values = []
    for name, value in fields.items():
        if isinstance(value, (list, tuple)):
            if not value:
                values.append("NULL")
                continue
            variables.update({f"{name}_{i}": str(v) for i, v in enumerate(value)})
            placeholders = ",".join(f":'{name}_{i}'" for i in range(len(value)))
            values.append(f"ARRAY[{placeholders}]::text[]")
        else:
            variables[name] = str(value)
            values.append(f":'{name}'")
    sql = (f"INSERT INTO measurements.run ({', '.join(names)}) "
           f"VALUES (:'spike', {', '.join(values)});")
    run_sql(env, sql, variables=variables)
    return int(scalar(env, "SELECT id FROM measurements.run WHERE spike = :'spike';",
                      variables={"spike": spike}))


def insert_threshold_rows(env, run_id: int, rows) -> int:
    run_sql(env, THRESHOLD_DDL)
    payload = [[run_id, r["candidate_key"], r["depth"], r["relation"],
                r["score"], r.get("title"), r.get("year"),
                r.get("has_abstract"), r.get("n_references")] for r in rows]
    if not payload:
        return 0
    copy_csv_into(
        env,
        "measurements.citation_frontier_threshold "
        "(run_id, candidate_key, depth, relation, score, title, year, "
        "has_abstract, n_references)",
        csv_rows(payload),
    )
    return len(payload)
