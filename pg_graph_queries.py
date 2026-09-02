"""The two relational consumers of the citation graph: candidates and
cocitation (plus the VOSviewer export the latter feeds).

Both read citation.work/citation.cites directly -- a nearest-neighbour
ranking with a 1-hop link count, and a co-citation self-join, are plain SQL
questions, and answering them through Cypher would buy nothing. The two
graph-shaped consumers (citers, hybrid) live in pg_graph_cypher.py and are
imported FROM THERE by whoever needs them -- this module re-exports
nothing. A facade would put the two files back into one surface: a change
to the other module's own private SQL constants would be a change to this
module's exports, which is precisely the coupling the split removed.

Both talk to Postgres through `pg_graph_common.graph_sql()` like everything
else that touches this schema -- see that module's own docstring for why
AGE's LOAD + search_path preamble has to be applied per psql invocation.

Data functions only: CLI argument parsing, dispatch and table printing live
in pg_graph.py, which imports this module and is imported by nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pg_graph_common
import pg_search
from pg_graph_common import FIELD_SEP, ROW_ARGS, split_records


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
#   {links_join}   --min-links, when asked for, turns that LEFT JOIN into
#                  an inner one carrying `l.n >= :min_links`: the answer is
#                  "of the K nearest, the ones with at least N links to our
#                  own documents", so it can be shorter than K. The cut
#                  cannot move below the LIMIT to decide eligibility
#                  instead: measured with EXPLAIN on this instance
#                  (enable_seqscan=off), EVERY membership test against
#                  `links` inside `nearest` -- the correlated subquery this
#                  replaced, `IN (SELECT ...)`, or a join -- makes the
#                  planner drop work_embedding_hnsw and sort the candidates
#                  instead, which is the one thing this shape exists to
#                  prevent. The correlated form also re-scanned the
#                  materialised (hence unindexed) `links` CTE once per row
#                  the index scan examined. Above the LIMIT the index scan
#                  survives and `links` is hashed once. Omitted entirely at
#                  the default 0 rather than written as a tautological >= 0.
#
# {target_expr} is module-owned SQL, never caller input: either the bound
# query vector or the corpus centroid subquery -- and it is spliced ONCE,
# into a MATERIALIZED CTE the nearest scan reads through a LATERAL. Spliced
# into both the score and the ORDER BY, as it was, the centroid became two
# textually distinct subqueries and Postgres aggregated over every
# our-document embedding twice per call, while the query vector's 1024
# floats were serialised into the statement text twice. LATERAL rather
# than a plain cross join because the ordering must belong to the scan:
# EXPLAIN on this instance (enable_seqscan=off) shows Index Scan using
# work_embedding_hnsw with Order By: embedding <=> t.v, i.e. the one shape
# this query exists to keep available.
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
target AS MATERIALIZED (
    SELECT {target_expr} AS v
),
nearest AS (
    SELECT n.*
    FROM target t
    CROSS JOIN LATERAL (
        SELECT w.id, w.key, w.year, w.title,
               1 - (w.embedding <=> t.v) AS score
        FROM citation.work w
        WHERE w.kind = 'external-skeleton' AND w.embedding IS NOT NULL
        ORDER BY w.embedding <=> t.v
        LIMIT :top
    ) n
)
SELECT n.key, coalesce(n.year::text, ''), coalesce(n.title, ''),
       n.score::text, coalesce(l.n, 0)::text
FROM nearest n
{links_join}
ORDER BY n.score DESC;
"""

_LINKS_KEPT = "LEFT JOIN links l ON l.id = n.id"
_LINKS_CUT = "JOIN links l ON l.id = n.id AND l.n >= :min_links"


def build_candidates_sql(target_expr: str, min_links: int) -> str:
    return _CANDIDATES_SQL.format(
        target_expr=target_expr,
        links_join=_LINKS_CUT if min_links > 0 else _LINKS_KEPT)


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
    """The `top` external-skeleton nodes nearest to `query` (or to the
    corpus centroid), with their link counts.

    min_links cuts that top-K rather than the pool it is drawn from, so
    asking for links can return fewer than `top` rows -- see the shape note
    on {links_join} above for the EXPLAIN measurement that decided it.
    """
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
    result = pg_graph_common.graph_sql(env, sql, variables=variables,
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

# A citing work with more outgoing references than this generates no pairs
# at all. The self-join below is quadratic in the per-citer reference count
# -- one citer with k references materialises k*(k-1)/2 pairs BEFORE the
# min_count aggregate can discard any of them -- and the works with the
# largest k are bibliographies, surveys and handbooks, whose "these two are
# cited together" says something about the field rather than about the two.
# Measured on the present graph (2026-09-02): 365 citing works, mean
# out-degree 6.6, maximum 54, so the default cuts nothing here; it exists so
# that a depth-2 crawl pulling in a review with several hundred references
# cannot turn one node into a hundred thousand intermediate pairs.
MAX_OUT_DEGREE = 200
# ... and however many citers survive the cap, the answer itself is a table
# a human reads: the most co-cited pairs, not every pair above min_count
# (2425 edges already yield 1005 pairs at min_count=4).
COCITATION_LIMIT = 500

# Co-citation, not bibliographic coupling: a pair (a, b) counts once per
# THIRD work that cites both -- c1.citing = c2.citing is the shared citer,
# c1.cited < c2.cited both dedupes the unordered pair and orders it so
# (a, b) and (b, a) are never counted as two different pairs.
#
# The cap is a CTE over the aggregate, applied before the join rather than
# as a HAVING over the pairs: filtering afterwards would already have paid
# for generating them. The ORDER BY carries the two keys after the count so
# that two runs over unchanged data return the same LIMIT-ed set, not an
# arbitrary slice of the ties (rank_candidates's reason, applied here in
# SQL because the cut is the database's).
_COCITATION_SQL = """
WITH citers AS (
    SELECT citing
    FROM citation.cites
    GROUP BY citing
    HAVING count(*) <= :max_out_degree
),
pairs AS (
    SELECT c1.cited AS a_id, c2.cited AS b_id, count(DISTINCT c1.citing) AS n
    FROM citation.cites c1
    JOIN citers ON citers.citing = c1.citing
    JOIN citation.cites c2 ON c2.citing = c1.citing AND c1.cited < c2.cited
    GROUP BY c1.cited, c2.cited
    HAVING count(DISTINCT c1.citing) >= :min_count
)
SELECT wa.key, coalesce(wa.title, ''), wb.key, coalesce(wb.title, ''), p.n::text
FROM pairs p
JOIN citation.work wa ON wa.id = p.a_id
JOIN citation.work wb ON wb.id = p.b_id
ORDER BY p.n DESC, wa.key, wb.key
LIMIT :limit;
"""


def cocitation(env, min_count: int = 2, max_out_degree: int = MAX_OUT_DEGREE,
               limit: int = COCITATION_LIMIT) -> list[dict]:
    """The `limit` most co-cited pairs, counting only citers under the
    out-degree cap. Both bounds travel with the result: the VOSviewer
    export is written from exactly the pairs returned here, so the map and
    the printed table can never describe different sets.
    """
    variables = {"min_count": str(int(min_count)),
                 "max_out_degree": str(int(max_out_degree)),
                 "limit": str(int(limit))}
    result = pg_graph_common.graph_sql(env, _COCITATION_SQL, variables=variables,
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
