#!/usr/bin/env python3
"""CLI over the citation AGE graph (schema pg_schema_citation.sql).

Argument parsing, dispatch and table printing, and nothing else: the psql/
AGE plumbing lives in pg_graph_common.py and the queries in
pg_graph_queries.py (which re-exports the two Cypher ones from
pg_graph_cypher.py). Both are imported at module level -- neither imports
this module, so there is no cycle to defer an import around.

Usage:
    python3 pg_graph.py init                  # applies pg_schema_citation.sql
    python3 pg_graph.py project               # rebuilds citation_graph
    python3 pg_graph.py project --check       # compares graph vs relational counts
    python3 pg_graph.py citers <document_id> [--depth N]
    python3 pg_graph.py candidates [--top K] [--query "<текст>"] [--min-links N]
    python3 pg_graph.py cocitation [--min-count M] [--max-out-degree D] [--limit L]
                                   [--export-vosviewer <dir>]
    python3 pg_graph.py hybrid "<вопрос>" [--top K] [--show-sql]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pg_graph_common
import pg_graph_queries as pgq
from pg_common import PostgresUnavailable, load_pgenv

try:
    # paths.py walks up looking for a theory/iis/ data tree and is
    # deliberately NOT bundled into the deploy artifact, where this CLI is
    # driven with an explicit --pgenv instead; absent, only the default
    # matters (same shim, and the same reason, as pg_search.py's main()).
    from paths import default_corpus_dir
except ImportError:  # pragma: no cover -- bundled artifact only
    default_corpus_dir = None


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(нет строк -- проекция пуста либо запрос не нашёл совпадений)")
        return
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(row))


def cmd_citers(args, env) -> int:
    rows = pgq.citers(env, args.document_id, depth=args.depth)
    _print_table(["id", "год", "kind", "title"],
                 [[r["key"], str(r["year"] or ""), r["kind"], r["title"]] for r in rows])
    return 0


def cmd_candidates(args, env) -> int:
    rows = pgq.candidates(env, top=args.top, query=args.query, min_links=args.min_links)
    _print_table(
        ["score", "links", "key", "год", "title"],
        [[f"{r['score']:.3f}", str(r["links"]), r["key"], str(r["year"] or ""), r["title"]] for r in rows],
    )
    return 0


def cmd_cocitation(args, env) -> int:
    pairs = pgq.cocitation(env, min_count=args.min_count,
                           max_out_degree=args.max_out_degree, limit=args.limit)
    if args.export_vosviewer:
        map_path, network_path, n_nodes, n_edges = pgq.write_vosviewer_export(pairs, Path(args.export_vosviewer))
        print(f"{map_path}: {n_nodes} узлов")
        print(f"{network_path}: {n_edges} связей")
        return 0
    _print_table(["a", "b", "count"], [[p["a_key"], p["b_key"], str(p["count"])] for p in pairs])
    return 0


def cmd_hybrid(args, env) -> int:
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


DISPATCH = {
    "citers": cmd_citers,
    "candidates": cmd_candidates,
    "cocitation": cmd_cocitation,
    "hybrid": cmd_hybrid,
}


def build_parser() -> argparse.ArgumentParser:
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
    cocitation_parser.add_argument(
        "--max-out-degree", type=int, default=pgq.MAX_OUT_DEGREE, dest="max_out_degree",
        help="a citing work with more outgoing references than this generates no pairs "
             f"(default {pgq.MAX_OUT_DEGREE}; see pg_graph_queries._COCITATION_SQL)",
    )
    cocitation_parser.add_argument(
        "--limit", type=int, default=pgq.COCITATION_LIMIT,
        help=f"how many of the most co-cited pairs to return (default {pgq.COCITATION_LIMIT})",
    )
    cocitation_parser.add_argument("--export-vosviewer", default=None, dest="export_vosviewer")

    hybrid_parser = sub.add_parser("hybrid")
    hybrid_parser.add_argument("question")
    hybrid_parser.add_argument("--top", type=int, default=10)
    hybrid_parser.add_argument("--show-sql", action="store_true", dest="show_sql")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.pgenv is not None:
        pgenv_path = args.pgenv
    elif default_corpus_dir is None:
        print("no --pgenv given and paths.py is not available here", file=sys.stderr)
        return 1
    else:
        try:
            pgenv_path = default_corpus_dir() / ".pgenv"
        except RuntimeError as exc:
            print(f"no --pgenv given and no repository context to locate one: {exc}",
                  file=sys.stderr)
            return 1

    try:
        env = load_pgenv(pgenv_path)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    if args.command == "init":
        pg_graph_common.init_schema(env)
        print("citation schema applied")
        return 0

    if args.command == "project":
        if args.check:
            return pg_graph_common.check(env)
        vertices, edges = pg_graph_common.project(env)
        print(f"V={vertices} E={edges}")
        return 0

    if args.command in DISPATCH:
        try:
            return DISPATCH[args.command](args, env)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
