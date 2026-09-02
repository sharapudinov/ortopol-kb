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

from pg_common import run_sql, run_sql_file, scalar, scalar_row

SCHEMA_PATH = Path(__file__).resolve().parent / "pg_schema_citation.sql"
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
    run_sql_file(env, SCHEMA_PATH)


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


def graph_counts(env: dict[str, str]) -> tuple[int, int]:
    """Current |V|/|E| of citation_graph, read without reprojecting.

    Assumes the label tables exist (project_graph() always creates them via
    create_vlabel/create_elabel, even for an empty graph) -- callers must
    check graph_exists() first, or let this raise on a graph that was never
    projected at all.
    """
    vertices, edges = graph_sql(
        env,
        f'SELECT (SELECT count(*) FROM {GRAPH_NAME}."Work"), '
        f'(SELECT count(*) FROM {GRAPH_NAME}."CITES");',
        extra_args=["-t", "-A", "-F", FIELD_SEP],
    ).stdout.strip().split(FIELD_SEP)
    return int(vertices), int(edges)


def compare_counts(work_n: int, cites_n: int, vertex_n: int, edge_n: int) -> tuple[int, int]:
    """(diff_vertices, diff_edges) = (actual graph count - relational count).

    Both zero means the graph is a faithful projection of the relational
    tables; pulled out as a pure function so the comparison itself has a
    unit test that needs no live database.
    """
    return (vertex_n - work_n, edge_n - cites_n)


def check(env: dict[str, str]) -> int:
    """Exit code of "is the projection still faithful": 0 when it is.

    Here rather than in the CLI because two entry points ask the question
    -- `pg_graph.py project --check` and pg_load_citations.py after a crawl
    -- and a second implementation of the comparison is exactly what
    compare_counts() exists to prevent.
    """
    if not graph_exists(env):
        print(f"проекция не строилась: графа {GRAPH_NAME} нет в ag_catalog.ag_graph",
              file=sys.stderr)
        return 1
    work_n = int(scalar(env, "SELECT count(*) FROM citation.work;"))
    cites_n = int(scalar(env, "SELECT count(*) FROM citation.cites;"))
    vertex_n, edge_n = graph_counts(env)
    diff_v, diff_e = compare_counts(work_n, cites_n, vertex_n, edge_n)
    if diff_v == 0 and diff_e == 0:
        print(f"OK: |V|={vertex_n} |E|={edge_n}")
        return 0
    print(
        f"MISMATCH: work={work_n} vertices={vertex_n} (diff {diff_v}); "
        f"cites={cites_n} edges={edge_n} (diff {diff_e})",
        file=sys.stderr,
    )
    return 1
