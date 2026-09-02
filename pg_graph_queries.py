"""The two relational consumers of the citation graph: candidates and
cocitation (plus the VOSviewer export the latter feeds).

Both read citation.work/citation.cites directly -- a nearest-neighbour
ranking with a 1-hop link count, and a co-citation self-join, are plain SQL
questions, and answering them through Cypher would buy nothing. The two
graph-shaped consumers (citers, hybrid) live in pg_graph_cypher.py and are
re-exported here, so pg_graph.py's main() and every other caller still see
one module with all four.

Both talk to Postgres through `pg_graph.graph_sql()` like everything else
that touches this schema -- see pg_graph.py's own docstring for why AGE's
LOAD + search_path preamble has to be applied per psql invocation.

Data functions only: CLI argument parsing, dispatch and table printing live
in pg_graph.py's main() (its docstring explains why: this module imports
pg_graph at its own top level, so pg_graph.py must not import this module at
ITS top level, only lazily from inside main(), or the two would form an
import cycle).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pg_graph
import pg_search
from pg_graph import FIELD_SEP, ROW_ARGS, split_records
from pg_graph_cypher import (  # noqa: F401  (re-exported: see the docstring)
    MAX_DEPTH,
    MIN_DEPTH,
    build_citers_sql,
    build_hybrid_sql,
    citers,
    hybrid,
    validate_depth,
    _HYBRID_SQL,
    _NEAREST_KEYS_SQL,
)


# ------------------------------------------------------------- candidates --

_CENTROID_EXPR = (
    "(SELECT avg(embedding) FROM citation.work "
    "WHERE kind = 'our-document' AND embedding IS NOT NULL)"
)

# Input for external-literature triage: external-skeleton nodes ranked by
# closeness to a query (or, absent one, to the corpus centroid) and by how
# many CITES edges already tie them to our own documents -- see
# theory/external/ for what happens to a candidate once picked.
#
# Both the ranking and the cut happen in the database, and the SHAPE is the
# whole point rather than a detail:
#
#   `nearest`      ORDER BY <distance> LIMIT :top directly over
#                  citation.work, with nothing joined underneath it. That is
#                  the only shape work_embedding_hnsw can serve: with a join
#                  below the LIMIT, Postgres cannot stop the ordered scan
#                  early and falls back to sorting every row (measured on
#                  this instance with enable_seqscan=off -- the joined shape
#                  plans a plain Index Scan + Sort, this one an Index Scan
#                  using work_embedding_hnsw). The earlier form returned
#                  EVERY external-skeleton row -- most of the graph, by
#                  construction -- computed a 1024-dimension distance for
#                  each, shipped them all through psql and sorted them in
#                  Python for a 20-row answer.
#   `links`        aggregates the two edge directions ONCE, as a UNION ALL
#                  keyed on the endpoint id, and is joined ABOVE the LIMIT,
#                  i.e. for at most :top rows. The earlier form was a
#                  correlated subquery per candidate row whose predicate was
#                  `citing = w.id OR cited = w.id` -- an OR across two
#                  columns no single index scan can serve, so it degraded
#                  toward re-reading citation.cites per candidate.
#   {links_cut}    --min-links, when asked for, is a filter on the base
#                  relation INSIDE `nearest`: the cut has to decide which
#                  rows are eligible for the top-K, not which of the K
#                  survive. It costs a lookup per row the index scan
#                  examines (the plan keeps the HNSW scan and adds a
#                  Filter), which is why it is omitted entirely at the
#                  default 0 rather than written as a tautological >= 0.
#
# {target_expr} is module-owned SQL, never caller input: either the bound
# query vector or the corpus centroid subquery.
#
# At the graph's present size (438 works) the planner still prefers a
# sequential scan with a top-N heapsort, and is right to -- 363 candidate
# rows are cheaper to sort than to walk an index for. The point of the shape
# is that the index becomes available as the graph grows, which the previous
# one never allowed.
_CANDIDATES_SQL = """
WITH links AS (
    SELECT e.id, count(*) AS n
    FROM (SELECT citing AS id, cited AS other FROM citation.cites
          UNION ALL
          SELECT cited AS id, citing AS other FROM citation.cites) e
    JOIN citation.work o ON o.id = e.other AND o.kind = 'our-document'
    GROUP BY e.id
),
nearest AS (
    SELECT w.id, w.key, w.year, w.title,
           1 - (w.embedding <=> {target_expr}) AS score
    FROM citation.work w
    WHERE w.kind = 'external-skeleton' AND w.embedding IS NOT NULL{links_cut}
    ORDER BY w.embedding <=> {target_expr}
    LIMIT :top
)
SELECT n.key, coalesce(n.year::text, ''), coalesce(n.title, ''),
       n.score::text, coalesce(l.n, 0)::text
FROM nearest n
LEFT JOIN links l ON l.id = n.id
ORDER BY n.score DESC;
"""

_LINKS_CUT = ("\n      AND coalesce((SELECT n FROM links WHERE links.id = w.id), 0) "
              ">= :min_links")


def build_candidates_sql(target_expr: str, min_links: int) -> str:
    return _CANDIDATES_SQL.format(
        target_expr=target_expr, links_cut=_LINKS_CUT if min_links > 0 else "")


def rank_candidates(rows: list[dict], min_links: int = 0, top: int | None = None) -> list[dict]:
    """Pure tie-break over the rows the database already ranked and limited.

    SQL orders by distance alone (anything else in the ORDER BY costs the
    HNSW index scan), so equal scores come back in whatever order the scan
    produced them; this settles them by links, then by key, so two runs over
    unchanged data print the same table. The min_links/top arguments are the
    same cut the SQL applied, re-asserted here rather than assumed -- and
    they are what lets the ranking policy be unit-tested against synthetic
    rows with no database. A row with score=None (no target embedding to
    compare against at all) cannot be ranked and is dropped.
    """
    scored = [r for r in rows if r.get("score") is not None and r["links"] >= min_links]
    scored.sort(key=lambda r: (-r["score"], -r["links"], r["key"]))
    return scored[:top] if top is not None else scored


def candidates(env, top: int = 20, query: str | None = None, min_links: int = 0) -> list[dict]:
    if query:
        vec = pg_search.embed_query(query, env)
        if vec is None:
            print("эмбеддинги недоступны, ранжирование по вектору невозможно", file=sys.stderr)
            return []
        sql = build_candidates_sql(":'vec'::vector", min_links)
        variables = {"vec": vec, "top": str(int(top))}
    else:
        sql = build_candidates_sql(_CENTROID_EXPR, min_links)
        variables = {"top": str(int(top))}
    if min_links > 0:
        variables["min_links"] = str(int(min_links))
    result = pg_graph.graph_sql(env, sql, variables=variables,
                                 extra_args=ROW_ARGS)
    rows = []
    for rec in split_records(result.stdout):
        key, year, title, score, links = rec.split(FIELD_SEP, 4)
        rows.append({
            "key": key,
            "year": int(year) if year else None,
            "title": title,
            "score": float(score) if score else None,
            "links": int(links) if links else 0,
        })
    return rank_candidates(rows, min_links=min_links, top=top)



# ------------------------------------------------------------- cocitation --

# Co-citation, not bibliographic coupling: a pair (a, b) counts once per
# THIRD work that cites both -- c1.citing = c2.citing is the shared citer,
# c1.cited < c2.cited both dedupes the unordered pair and orders it so
# (a, b) and (b, a) are never counted as two different pairs.
_COCITATION_SQL = """
SELECT wa.key, coalesce(wa.title, ''), wb.key, coalesce(wb.title, ''), n::text
FROM (
    SELECT LEAST(c1.cited, c2.cited) AS a_id, GREATEST(c1.cited, c2.cited) AS b_id,
           count(DISTINCT c1.citing) AS n
    FROM citation.cites c1
    JOIN citation.cites c2 ON c1.citing = c2.citing AND c1.cited < c2.cited
    GROUP BY a_id, b_id
    HAVING count(DISTINCT c1.citing) >= :min_count
) pairs
JOIN citation.work wa ON wa.id = pairs.a_id
JOIN citation.work wb ON wb.id = pairs.b_id
ORDER BY n DESC;
"""


def cocitation(env, min_count: int = 2) -> list[dict]:
    result = pg_graph.graph_sql(env, _COCITATION_SQL, variables={"min_count": str(min_count)},
                                 extra_args=ROW_ARGS)
    pairs = []
    for rec in split_records(result.stdout):
        a_key, a_title, b_key, b_title, n = rec.split(FIELD_SEP, 4)
        pairs.append({"a_key": a_key, "a_title": a_title, "b_key": b_key, "b_title": b_title, "count": int(n)})
    return pairs


def build_vosviewer_export(pairs: list[dict]) -> tuple[list[str], list[str]]:
    """Formats co-citation pairs as VOSviewer's own text-file pair
    (https://app.vosviewer.com/docs/file-types/map-and-network-file-type/):
    tab-delimited; the map file has a header ('id\\tlabel') and one row per
    distinct item; the network file has NO header and one
    'id1\\tid2\\tweight' row per link. VOSviewer's own ids are sequential
    integers (its manual's own examples use 1, 2, 10, ...) -- `key` is a
    string, so ids are assigned in first-seen order across the pairs, the
    only order available here.
    """
    ids: dict[str, int] = {}
    labels: dict[str, str] = {}

    def node_id(key: str, title: str) -> int:
        if key not in ids:
            ids[key] = len(ids) + 1
            labels[key] = title or key
        return ids[key]

    network_lines = []
    for p in pairs:
        a = node_id(p["a_key"], p["a_title"])
        b = node_id(p["b_key"], p["b_title"])
        network_lines.append(f"{a}\t{b}\t{p['count']}")
    map_lines = ["id\tlabel"] + [f"{ids[k]}\t{labels[k]}" for k in ids]
    return map_lines, network_lines


def write_vosviewer_export(pairs: list[dict], out_dir: Path) -> tuple[Path, Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    map_lines, network_lines = build_vosviewer_export(pairs)
    map_path = out_dir / "VOSviewer_map.txt"
    network_path = out_dir / "VOSviewer_network.txt"
    map_path.write_text("\n".join(map_lines) + "\n", encoding="utf-8")
    network_path.write_text("\n".join(network_lines) + "\n", encoding="utf-8")
    return map_path, network_path, len(map_lines) - 1, len(network_lines)
