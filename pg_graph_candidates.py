"""Nearest-neighbour triage over the citation graph: the external-skeleton
nodes closest to a query (or to the corpus centroid), with the count of
CITES edges already tying each to one of our own documents.

A plain SQL question -- a vector ranking with a 1-hop aggregate -- so it is
answered relationally rather than through Cypher, which would buy nothing
here. Its sibling relational consumer, co-citation, lives in
pg_graph_cocitation.py; the two graph-shaped ones (citers, hybrid) in
pg_graph_cypher.py. No module re-exports another: a facade would put the
files back into one surface, so that a change to another module's private
SQL constants would be a change to this module's exports.

Talks to Postgres through `pg_graph_common.graph_sql()` like everything else
that touches this schema -- see that module's own docstring for why AGE's
LOAD + search_path preamble has to be applied per psql invocation.

Data functions only: CLI argument parsing, dispatch and table printing live
in pg_graph.py, which imports this module and is imported by nothing.
"""
from __future__ import annotations

import sys

import pg_graph_common
import pg_search
from citation_vocab import WorkKind
from pg_common import FIELD_SEP, ROW_ARGS, split_records


_CENTROID_EXPR = (
    "(SELECT avg(embedding) FROM citation.work "
    f"WHERE kind = '{WorkKind.OUR_DOCUMENT}' AND embedding IS NOT NULL)"
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
#   `links`        two counts per top-K row, evaluated in a LEFT JOIN
#                  LATERAL ABOVE the LIMIT: measured on this instance
#                  (EXPLAIN, enable_seqscan=off) an Index Only Scan using
#                  cites_pkey downward and a Bitmap Index Scan using
#                  cites_cited_idx upward, both driven from that row's id.
#                  Both earlier forms answered the same question over the
#                  WHOLE of citation.cites: a correlated subquery whose
#                  predicate was `citing = w.id OR cited = w.id` -- an OR
#                  across two columns no single index scan can serve -- and
#                  then a grouped CTE that aggregated every edge in the graph
#                  and was joined on `l.id = n.id`, a qualifier the planner
#                  cannot push into the grouped subquery. So the full O(|E|)
#                  scan ran on every call, however small --top was, to
#                  attach a number to at most :top rows.
#   {links_cut}    --min-links, when asked for, filters that lateral's count
#                  above the LIMIT: the answer is "of the K nearest, the ones
#                  with at least N links to our own documents", so it can be
#                  shorter than K. The cut cannot move below the LIMIT to
#                  decide eligibility instead: measured with EXPLAIN on this
#                  instance (enable_seqscan=off), EVERY membership test
#                  inside `nearest` makes the planner drop
#                  work_embedding_hnsw and sort the candidates instead, which
#                  is the one thing this shape exists to prevent. Omitted
#                  entirely at the default 0 rather than written as a
#                  tautological >= 0.
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
WITH target AS MATERIALIZED (
    SELECT {target_expr} AS v
),
nearest AS (
    SELECT n.*
    FROM target t
    CROSS JOIN LATERAL (
        SELECT w.id, w.key, w.year, w.title,
               1 - (w.embedding <=> t.v) AS score
        FROM citation.work w
        WHERE w.kind = '{external_skeleton}' AND w.embedding IS NOT NULL
        ORDER BY w.embedding <=> t.v
        LIMIT :top
    ) n
)
SELECT n.key, coalesce(n.year::text, ''), coalesce(n.title, ''),
       n.score::text, coalesce(links.n, 0)::text
FROM nearest n
LEFT JOIN LATERAL (
    SELECT (SELECT count(*) FROM citation.cites c
            JOIN citation.work o ON o.id = c.cited AND o.kind = '{our_document}'
            WHERE c.citing = n.id)
         + (SELECT count(*) FROM citation.cites c
            JOIN citation.work o ON o.id = c.citing AND o.kind = '{our_document}'
            WHERE c.cited = n.id) AS n
) links ON true
{links_cut}
ORDER BY n.score DESC;
"""

_LINKS_CUT = "WHERE links.n >= :min_links"


def build_candidates_sql(target_expr: str, min_links: int) -> str:
    return _CANDIDATES_SQL.format(
        target_expr=target_expr,
        links_cut=_LINKS_CUT if min_links > 0 else "",
        our_document=WorkKind.OUR_DOCUMENT,
        external_skeleton=WorkKind.EXTERNAL_SKELETON)


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
