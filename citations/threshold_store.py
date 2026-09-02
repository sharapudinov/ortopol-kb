#!/usr/bin/env python3
"""Writes to `measurements`: the tau calibration and its run row.

Kept out of store.py because it is a different schema with a different
contract. store.py writes the durable citation graph -- upserts, preserved
kinds, preserved embeddings. This writes a research result under EXTENDING
procedure D: idempotent by spike name (the id shifts on every reload, the
name does not), verdict left NULL because the verdict is the orchestrator's,
and one row per scored candidate so the distribution the threshold rests on
is queryable rather than summarised in prose.

Table shape, so that recreating from scratch gives the same columns as the
instance that already ran: (run_id, candidate_key, depth, relation, score,
title, year) plus TWO columns added after the first calibration --
`has_abstract` and `n_references`. They are not decoration. has_abstract
carries the finding that a title-only candidate scores measurably lower
(median 0.6506 against 0.6893, n=173 against 217), i.e. that a high tau
filters on metadata completeness as much as on relevance; n_references is
what expanding the node costs at the next depth. Without them the two claims
the calibration report rests on would live in a scratch script rather than in
the base. THRESHOLD_DDL therefore ends in ADD COLUMN IF NOT EXISTS, and both
CREATE and ALTER are idempotent -- a fresh instance and an instance that
predates the columns converge on the same shape.
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
def _bind(fields: dict) -> tuple[list[tuple[str, str]], dict]:
    """[(column, SQL expression)] plus the psql variables they read.

    Shared by the insert and the in-place update below so a text[] field is
    spelled the same way by both. An empty list becomes NULL rather than an
    empty array: the run row means "not applicable" by it, and the two are
    different answers to a later query.
    """
    bound: list[tuple[str, str]] = []
    variables: dict[str, str] = {}
    for name, value in fields.items():
        if isinstance(value, (list, tuple)):
            if not value:
                bound.append((name, "NULL"))
                continue
            variables.update({f"{name}_{i}": str(v) for i, v in enumerate(value)})
            placeholders = ",".join(f":'{name}_{i}'" for i in range(len(value)))
            bound.append((name, f"ARRAY[{placeholders}]::text[]"))
        else:
            variables[name] = str(value)
            bound.append((name, f":'{name}'"))
    return bound, variables


def upsert_run(env, spike: str, fields: dict) -> int:
    """Idempotent by spike name, the convention run 85 established: the id
    shifts on every reload, the spike name does not.

    Idempotent HERE means DELETE + INSERT, so it also discards every data
    row that references the old id (ON DELETE CASCADE). Use it to establish
    the run, never to amend one that already has its data:
    update_run_fields() is that.
    """
    run_sql(env, "DELETE FROM measurements.run WHERE spike = :'spike';",
            variables={"spike": spike})
    bound, variables = _bind(fields)
    variables["spike"] = spike
    names = ["spike"] + [name for name, _expr in bound]
    values = [":'spike'"] + [expr for _name, expr in bound]
    sql = (f"INSERT INTO measurements.run ({', '.join(names)}) "
           f"VALUES ({', '.join(values)});")
    run_sql(env, sql, variables=variables)
    return int(scalar(env, "SELECT id FROM measurements.run WHERE spike = :'spike';",
                      variables={"spike": spike}))


def update_run_fields(env, spike: str, fields: dict) -> None:
    """Amends the run row in place, keeping its id and its data rows.

    The field this exists for is verify_query: it must name the numbers the
    measurement produced, and those are known only after the data table is
    filled. Restating them with a second upsert_run() would delete the run
    row -- and cascade away exactly the rows just written, forcing the whole
    aggregation pass to run a second time to put them back.
    """
    bound, variables = _bind(fields)
    if not bound:
        return
    variables["spike"] = spike
    assignments = ", ".join(f"{name} = {expr}" for name, expr in bound)
    run_sql(env, f"UPDATE measurements.run SET {assignments} WHERE spike = :'spike';",
            variables=variables)


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
