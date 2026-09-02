#!/usr/bin/env python3
"""CLI over the citation AGE graph (schema pg_schema_citation.sql).

Talks to Postgres through psql, like every other pg_*.py script here (see
pg_common.py for why no driver). AGE's own contract is session-local:
`LOAD 'age'` and `SET search_path = ag_catalog, "$user", public` must be
issued by the client before any ag_catalog/cypher call, never baked into
the server config (deploy/pg/README.md "Activation"). graph_sql() is the
single place that prefixes those two statements, so pg_graph_queries.py's
consumers (citers/candidates/cocitation/hybrid) reuse it instead of
re-deriving the contract per call site.

pg_graph_queries is imported lazily, from inside main() rather than at
module level: it (and pg_graph_cypher, which it re-exports) imports
`pg_graph` to reach graph_sql(), so importing it up here would be a cycle.
This module stays the thin CLI/dispatch/formatting layer plus the shared
psql plumbing both query modules use; the query logic itself lives there.

Usage:
    python3 pg_graph.py init                 # applies pg_schema_citation.sql
    python3 pg_graph.py project               # rebuilds citation_graph
    python3 pg_graph.py project --check       # compares graph vs relational counts
    python3 pg_graph.py citers <document_id> [--depth N]
    python3 pg_graph.py candidates [--top K] [--query "<текст>"] [--min-links N]
    python3 pg_graph.py cocitation [--min-count M] [--export-vosviewer <dir>]
    python3 pg_graph.py hybrid "<вопрос>" [--top K] [--show-sql]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pg_common import PostgresUnavailable, load_pgenv, run_sql, run_sql_file, scalar, scalar_row

SCHEMA_PATH = Path(__file__).resolve().parent / "pg_schema_citation.sql"
GRAPH_NAME = "citation_graph"
_AGE_PREAMBLE = "LOAD 'age';\nSET search_path = ag_catalog, \"$user\", public;\n"

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


def graph_sql(env: dict[str, str], sql: str, **kwargs):
    """Runs `sql` with AGE activated for this one psql invocation.

    Every graph-touching query goes through this, not just the ones in this
    module: the read-only consumers in pg_graph_queries import it rather than repeat the
    two activation statements themselves.
    """
    return run_sql(env, _AGE_PREAMBLE + sql, **kwargs)


def init_schema(env: dict[str, str]) -> None:
    run_sql_file(env, SCHEMA_PATH)


def project(env: dict[str, str]) -> tuple[int, int]:
    """Rebuilds citation_graph from citation.work/citation.cites and
    returns the (vertices, edges) counts citation.project_graph() reports.
    """
    vertices, edges = scalar_row(
        env,
        _AGE_PREAMBLE + "SELECT * FROM citation.project_graph();",
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
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip().split("\x1f")
    return int(vertices), int(edges)


def compare_counts(work_n: int, cites_n: int, vertex_n: int, edge_n: int) -> tuple[int, int]:
    """(diff_vertices, diff_edges) = (actual graph count - relational count).

    Both zero means the graph is a faithful projection of the relational
    tables; pulled out as a pure function so the comparison itself has a
    unit test that needs no live database.
    """
    return (vertex_n - work_n, edge_n - cites_n)


def check(env: dict[str, str]) -> int:
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


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(нет строк -- проекция пуста либо запрос не нашёл совпадений)")
        return
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(row))


def cmd_citers(pgq, args, env) -> int:
    rows = pgq.citers(env, args.document_id, depth=args.depth)
    _print_table(["id", "год", "kind", "title"],
                 [[r["key"], str(r["year"] or ""), r["kind"], r["title"]] for r in rows])
    return 0


def cmd_candidates(pgq, args, env) -> int:
    rows = pgq.candidates(env, top=args.top, query=args.query, min_links=args.min_links)
    _print_table(
        ["score", "links", "key", "год", "title"],
        [[f"{r['score']:.3f}", str(r["links"]), r["key"], str(r["year"] or ""), r["title"]] for r in rows],
    )
    return 0


def cmd_cocitation(pgq, args, env) -> int:
    pairs = pgq.cocitation(env, min_count=args.min_count)
    if args.export_vosviewer:
        map_path, network_path, n_nodes, n_edges = pgq.write_vosviewer_export(pairs, Path(args.export_vosviewer))
        print(f"{map_path}: {n_nodes} узлов")
        print(f"{network_path}: {n_edges} связей")
        return 0
    _print_table(["a", "b", "count"], [[p["a_key"], p["b_key"], str(p["count"])] for p in pairs])
    return 0


def cmd_hybrid(pgq, args, env) -> int:
    if args.show_sql:
        print(pgq._HYBRID_SQL)
        return 0
    rows = pgq.hybrid(env, args.question, top=args.top)
    _print_table(
        ["seed", "год", "title", "score", "направление", "сосед", "заголовок соседа"],
        [[r["key"], str(r["year"] or ""), r["title"], f"{r['score']:.3f}",
          r["direction"], r["neighbor_key"], r["neighbor_title"]] for r in rows],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgenv", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    project_parser = sub.add_parser("project")
    project_parser.add_argument("--check", action="store_true",
                                help="compare current graph counts against "
                                     "citation.work/cites instead of reprojecting")

    citers_parser = sub.add_parser("citers")
    citers_parser.add_argument("document_id")
    citers_parser.add_argument("--depth", type=int, default=1)

    candidates_parser = sub.add_parser("candidates")
    candidates_parser.add_argument("--top", type=int, default=20)
    candidates_parser.add_argument("--query", default=None)
    candidates_parser.add_argument("--min-links", type=int, default=0, dest="min_links")

    cocitation_parser = sub.add_parser("cocitation")
    cocitation_parser.add_argument("--min-count", type=int, default=2, dest="min_count")
    cocitation_parser.add_argument("--export-vosviewer", default=None, dest="export_vosviewer")

    hybrid_parser = sub.add_parser("hybrid")
    hybrid_parser.add_argument("question")
    hybrid_parser.add_argument("--top", type=int, default=10)
    hybrid_parser.add_argument("--show-sql", action="store_true", dest="show_sql")

    args = parser.parse_args(argv)

    if args.pgenv is not None:
        pgenv_path = args.pgenv
    else:
        try:
            from paths import default_corpus_dir
            pgenv_path = default_corpus_dir() / ".pgenv"
        except (ImportError, RuntimeError) as exc:
            print(f"no --pgenv given and no repository context to locate one: {exc}",
                  file=sys.stderr)
            return 1

    try:
        env = load_pgenv(pgenv_path)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    if args.command == "init":
        init_schema(env)
        print("citation schema applied")
        return 0

    if args.command == "project":
        if args.check:
            return check(env)
        vertices, edges = project(env)
        print(f"V={vertices} E={edges}")
        return 0

    if args.command in ("citers", "candidates", "cocitation", "hybrid"):
        import pg_graph_queries as pgq  # deferred: see module docstring
        dispatch = {
            "citers": cmd_citers,
            "candidates": cmd_candidates,
            "cocitation": cmd_cocitation,
            "hybrid": cmd_hybrid,
        }
        try:
            return dispatch[args.command](pgq, args, env)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
