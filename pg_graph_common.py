#!/usr/bin/env python3
"""The citation graph's low-level layer: AGE session contract, psql row
separators, and the projection primitives every consumer shares.

Talks to Postgres through psql, like every other pg_*.py script here (see
pg_common.py for why no driver). AGE's own contract is session-local:
`LOAD 'age'` and `SET search_path = ag_catalog, "$user", public` must be
issued by the client before any ag_catalog/cypher call, never baked into
the server config (deploy/pg/README.md "Activation"). graph_sql() is the
single place that prefixes those two statements, so no call site re-derives
the contract.

Separate from pg_graph.py, which is the CLI over these functions and
nothing else. Five modules need this plumbing and none of them wants a
command-line parser: pg_graph_queries/pg_graph_cypher issue the queries,
citation_checks.py and deploy/smoke_checks.py compare the projection
against the relational tables, and pg_load_citations.py applies the schema
and reprojects after a crawl. While these functions lived in pg_graph.py,
those consumers imported a CLI entry point to reach them, the two query
modules and their own CLI formed an import cycle (broken only by a
deferred import inside main()), and the artifact had to bundle argparse
dispatch code no recipient calls as a library.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from pg_common import run_sql, run_sql_file, scalar, scalar_row

# Applied in this order by init_schema(): data definition first (crawl_step's
# columns/indexes must exist before anything reads or backfills them), the
# AGE projection functions second (they only reference citation.work/cites,
# not the journal columns), the journal's one-time backfill last (it depends
# on the columns the first file adds). kb/CLAUDE.md FILE_SIZE split
# pg_schema_citation.sql into these three along that seam.
SCHEMA_PATHS = (
    Path(__file__).resolve().parent / "pg_schema_citation.sql",
    Path(__file__).resolve().parent / "pg_schema_citation_graph.sql",
    Path(__file__).resolve().parent / "pg_schema_citation_backfill.sql",
)
GRAPH_NAME = "citation_graph"
AGE_PREAMBLE = "LOAD 'age';\nSET search_path = ag_catalog, \"$user\", public;\n"

# How a multi-row graph query's psql output is delimited, and the psql flags
# that produce it. Here rather than in the query modules because it is the
# same plumbing graph_sql() is: a title/reason can contain a comma, a tab and
# a newline, so the separators are the ASCII unit/record ones no source text
# carries. \x1e is deliberately NOT run through str.splitlines() -- that
# treats \x1c-\x1e and \x85 as line boundaries too, which is how a
# multi-line title used to arrive as several rows.
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
ROW_ARGS = ["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP]


def split_records(stdout: str) -> list[str]:
    return [r.strip("\n") for r in stdout.split(RECORD_SEP) if r.strip("\n")]


_SCHEMA_EXISTS_SQL = "SELECT to_regclass('citation.work') IS NOT NULL;"
_KIND_COUNTS_SQL = "SELECT w.kind, count(*) FROM citation.work w{where} GROUP BY 1 ORDER BY 1;"


def citation_schema_exists(env: dict[str, str]) -> bool:
    """Is schema citation applied to this database at all?

    Here rather than in each asker because three of them ask it -- the
    completeness run (citation_checks), the packager (deploy/
    citation_profile) and the crawl CLI under --dry-run -- and to_regclass
    of one table standing for a whole schema is a convention, not an
    obvious fact: two spellings of it would answer differently the day the
    schema gains a table before citation.work exists.
    """
    return scalar(env, _SCHEMA_EXISTS_SQL) == "t"


def kind_counts(env: dict[str, str], where: str = "") -> dict[str, int]:
    """{kind: rows} over citation.work, optionally narrowed by `where`.

    The census two callers need in two shapes -- the whole table for the
    completeness summary, and the shipped-rows-only subset for the public
    artifact's manifest (`where` carries that predicate, alias `w`). One
    query and one parse: the parse is the FIELD_SEP contract above, and a
    second copy of it drifts the moment one caller's separator changes.
    """
    out = run_sql(env, _KIND_COUNTS_SQL.format(where=where),
                  extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.strip():
            kind, n = line.split(FIELD_SEP)
            counts[kind] = int(n)
    return counts


def graph_sql(env: dict[str, str], sql: str, **kwargs):
    """Runs `sql` with AGE activated for this one psql invocation."""
    return run_sql(env, AGE_PREAMBLE + sql, **kwargs)


def init_schema(env: dict[str, str]) -> None:
    for path in SCHEMA_PATHS:
        run_sql_file(env, path)


def project(env: dict[str, str]) -> tuple[int, int]:
    """Rebuilds citation_graph from citation.work/citation.cites and
    returns the (vertices, edges) counts citation.project_graph() reports.
    """
    vertices, edges = scalar_row(
        env,
        AGE_PREAMBLE + "SELECT * FROM citation.project_graph();",
        expected_columns=2,
    )
    return int(vertices), int(edges)


def graph_exists(env: dict[str, str]) -> bool:
    n = graph_sql(
        env,
        f"SELECT count(*) FROM ag_catalog.ag_graph WHERE name = '{GRAPH_NAME}';",
        extra_args=["-t", "-A"],
    ).stdout.strip()
    return n == "1"


def compare_counts(work_n: int, cites_n: int, vertex_n: int, edge_n: int) -> tuple[int, int]:
    """(diff_vertices, diff_edges) = (actual graph count - relational count).

    Cardinality only, and cardinality is not faithfulness -- see
    projection_faults() for the other half. Pulled out as a pure function
    so the arithmetic has a unit test that needs no live database.
    """
    return (vertex_n - work_n, edge_n - cites_n)


# ONE reading: four counts and, as two md5 pairs, the projection's CONTENT.
# The counts travel here rather than in four round trips of their own -- a
# round trip is a psql fork, a temp script and a connection, while the
# digests beside them cost 13 ms. The graph-side counts assume the label
# tables exist (project_graph() creates them even for an empty graph): the
# graph_exists() guard in front of this statement is what says they do. The graph does not merely
# have to be the right SIZE: the read path serves graph properties
# (pg_graph_cypher's citers query returns key/title/year/kind straight out
# of the label table), and citations/store.py updates title/kind/year on
# rows that already exist -- a row-count-preserving change that leaves the
# graph carrying yesterday's title while every count still matches.
#
# Both sides of each pair are built from the same field order and the same
# separators, and ordered under COLLATE "C" so the digest does not depend
# on the database's collation. chr(10) rather than an escaped literal: the
# statement travels through a psql script, where one backslash convention
# fewer is one hazard fewer. An empty table digests as md5('') on both
# sides, so an empty graph over empty tables is faithful, not "unknown".
_READING_SQL = """
SELECT
 (SELECT count(*) FROM citation.work),
 (SELECT count(*) FROM citation.cites),
 (SELECT count(*) FROM {graph}."Work"),
 (SELECT count(*) FROM {graph}."CITES"),
 (SELECT md5(coalesce(string_agg(
    coalesce(w.key,'')||'|'||coalesce(w.kind,'')||'|'
    ||coalesce(w.year::text,'')||'|'||coalesce(w.title,''),
    chr(10) ORDER BY w.key COLLATE "C"), ''))
  FROM citation.work w),
 (SELECT md5(coalesce(string_agg(
    v.key||'|'||v.kind||'|'||v.year||'|'||v.title,
    chr(10) ORDER BY v.key COLLATE "C"), ''))
  FROM (SELECT coalesce(properties->>'"key"','') AS key,
               coalesce(properties->>'"kind"','') AS kind,
               coalesce(properties->>'"year"','') AS year,
               coalesce(properties->>'"title"','') AS title
        FROM {graph}."Work") v),
 (SELECT md5(coalesce(string_agg(
    a.key||'|'||b.key||'|'||ci.source,
    chr(10) ORDER BY a.key COLLATE "C", b.key COLLATE "C", ci.source COLLATE "C"), ''))
  FROM citation.cites ci
  JOIN citation.work a ON a.id = ci.citing
  JOIN citation.work b ON b.id = ci.cited),
 (SELECT md5(coalesce(string_agg(
    e.citing||'|'||e.cited||'|'||e.source,
    chr(10) ORDER BY e.citing COLLATE "C", e.cited COLLATE "C", e.source COLLATE "C"), ''))
  FROM (SELECT coalesce(s.properties->>'"key"','') AS citing,
               coalesce(t.properties->>'"key"','') AS cited,
               coalesce(c.properties->>'"source"','') AS source
        FROM {graph}."CITES" c
        JOIN {graph}."Work" s ON s.id = c.start_id
        JOIN {graph}."Work" t ON t.id = c.end_id) e);
"""


class Projection(NamedTuple):
    """One reading of the projection: how big both sides are, and what both
    sides say. The fingerprints are the content half of "faithful"; the
    counts stay separate because they are what a human can act on.
    """

    work_n: int
    cites_n: int
    vertex_n: int
    edge_n: int
    work_digest: str
    graph_work_digest: str
    cites_digest: str
    graph_cites_digest: str


def projection_reading(env: dict[str, str]) -> Projection:
    """Both sides' sizes and both sides' content, in ONE round trip.

    Measured on the live 438-work / 2425-edge graph (AGE 1.7.0, PostgreSQL
    17.11): 13 ms for the four digests together, so the content reading
    costs the check nothing a caller would notice, and the counts beside
    them cost less than the psql startup they used to pay separately.
    Assumes the graph exists; projection_diff() is the guarded caller.
    """
    row = graph_sql(
        env,
        _READING_SQL.format(graph=GRAPH_NAME),
        extra_args=["-t", "-A", "-F", FIELD_SEP],
    ).stdout.strip().split(FIELD_SEP)
    work_n, cites_n, vertex_n, edge_n = (int(value) for value in row[:4])
    return Projection(work_n, cites_n, vertex_n, edge_n, *row[4:])


def content_fingerprints(env: dict[str, str]) -> tuple[str, str, str, str]:
    """(work, graph work, cites, graph cites) md5s: the content half of the
    reading, for a caller that wants only the digests."""
    seen = projection_reading(env)
    return (seen.work_digest, seen.graph_work_digest,
            seen.cites_digest, seen.graph_cites_digest)


def projection_diff(env: dict[str, str]) -> Projection | None:
    """The whole reading behind "is the projection faithful", or None when
    there is no graph.

    Does the graph exist at all, how many rows does each side hold, and
    what does each side actually carry. projection_faults() is the verdict
    over the result and every asker renders it differently -- an exit code
    (check), a list of problem strings (citation_checks), an (ok, detail)
    pair against the manifest (deploy/smoke_checks) -- but the reads and
    the graph-missing branch are this function's, not each caller's.

    Two psql invocations: the guard, because a statement naming
    citation_graph."Work" fails outright on a graph that was never
    projected, and the reading. Five of them for 13 ms of work was five
    process startups.
    """
    if not graph_exists(env):
        return None
    return projection_reading(env)


def projection_faults(seen: Projection) -> list[str]:
    """Every way `seen` is not a faithful projection, worded once here.

    Empty means faithful. A digest mismatch stands on its own: equal counts
    with unequal content is the normal shape of the failure, not an exotic
    one, so it is reported even when the arithmetic agrees.
    """
    faults = []
    diff_v, diff_e = compare_counts(
        seen.work_n, seen.cites_n, seen.vertex_n, seen.edge_n)
    if diff_v != 0 or diff_e != 0:
        faults.append(
            f"work={seen.work_n} vertices={seen.vertex_n} (diff {diff_v}); "
            f"cites={seen.cites_n} edges={seen.edge_n} (diff {diff_e})")
    for what, ours, theirs in (
        ("вершины (key|kind|year|title)", seen.work_digest, seen.graph_work_digest),
        ("рёбра (citing|cited|source)", seen.cites_digest, seen.graph_cites_digest),
    ):
        if ours != theirs:
            faults.append(f"content fingerprint differs: {what} — "
                          f"таблицы {ours}, граф {theirs}")
    return faults


def check(env: dict[str, str]) -> int:
    """Exit code of "is the projection still faithful": 0 when it is.

    Here rather than in the CLI because two entry points ask the question
    -- `pg_graph.py project --check` and pg_load_citations.py after a crawl
    -- and a second implementation of the comparison is exactly what
    projection_faults() exists to prevent.
    """
    seen = projection_diff(env)
    if seen is None:
        print(f"проекция не строилась: графа {GRAPH_NAME} нет в ag_catalog.ag_graph",
              file=sys.stderr)
        return 1
    faults = projection_faults(seen)
    if not faults:
        print(f"OK: |V|={seen.vertex_n} |E|={seen.edge_n}")
        return 0
    print("MISMATCH: " + "; ".join(faults), file=sys.stderr)
    return 1
