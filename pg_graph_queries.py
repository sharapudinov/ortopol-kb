"""Read-only consumers of the citation graph (pg_schema_citation.sql /
pg_graph.py): citers, candidates, cocitation, hybrid.

Data functions only -- CLI argument parsing, dispatch and table printing
live in pg_graph.py's main() (its docstring explains why: this module
imports pg_graph.graph_sql at its own top level, so pg_graph.py must not
import this module at ITS top level, only lazily from inside main(), or
the two would form an import cycle).

All four talk to Postgres exclusively through `pg_graph.graph_sql()` (AGE's
LOAD + search_path preamble applied once per psql invocation, same
contract every graph-touching query in this repository follows) -- see
pg_graph.py's own docstring for why that preamble cannot be baked into the
server.

candidates/cocitation read citation.work/citation.cites directly (plain
relational SQL is simpler and sufficient for a 1-hop edge count or a
co-citation join); citers and hybrid are the two that actually call
ag_catalog.cypher(), because "who points at this node transitively" and
"what does the graph say about my nearest neighbours" are graph-shaped
questions relational SQL alone answers only awkwardly.

A citation.work.key or title embedded in a Cypher command is untrusted
external-source text (OpenAlex, etc): citers() gets it pre-escaped from
`citation.cypher_literal()` -- the one escaping implementation
pg_schema_citation.sql defines and this module trusts, never re-derives.
hybrid()'s cypher() call carries no variable text at all (it returns the
whole edge set unconditionally), so no escaping question arises there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pg_graph
import pg_search
from pg_common import scalar

FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
MIN_DEPTH, MAX_DEPTH = 1, 3


def _split_records(stdout: str) -> list[str]:
    return [r.strip("\n") for r in stdout.split(RECORD_SEP) if r.strip("\n")]


# ---------------------------------------------------------------- citers --

def validate_depth(depth: int) -> int:
    """Cypher's *1..N variable-length path bound. 1 is the useful floor (a
    *0..N match would also return the seed itself, which is not "who cites
    this"); 3 caps traversal cost -- this SQL runs unindexed over the whole
    label per call, not a claim that citation chains never run deeper.
    """
    if not (MIN_DEPTH <= depth <= MAX_DEPTH):
        raise ValueError(f"--depth must be between {MIN_DEPTH} and {MAX_DEPTH}, got {depth}")
    return depth


def build_citers_sql(escaped_seed_key: str, depth: int) -> str:
    """`escaped_seed_key` MUST already be the output of
    `citation.cypher_literal()` (see citers() below, which fetches it
    pre-escaped) -- this function only splices it into query text, it does
    not escape raw input itself.
    """
    validate_depth(depth)
    cyp = (
        f"MATCH (w:Work {{key: '{escaped_seed_key}'}})<-[:CITES*1..{depth}]-(c:Work) "
        "RETURN DISTINCT c.key, c.title, c.year, c.kind"
    )
    if "$CYPHERQ$" in cyp:
        raise ValueError("seed key collides with the $CYPHERQ$ delimiter")
    return (
        "SELECT c_key::text, c_title::text, c_year::text, c_kind::text "
        f"FROM ag_catalog.cypher('citation_graph', $CYPHERQ${cyp}$CYPHERQ$) "
        "AS (c_key agtype, c_title agtype, c_year agtype, c_kind agtype);"
    )


def citers(env, document_id: str, depth: int = 1) -> list[dict]:
    escaped = scalar(
        env,
        "SELECT citation.cypher_literal(key) FROM citation.work "
        "WHERE document_id = :'doc_id' LIMIT 1;",
        variables={"doc_id": document_id},
    )
    if not escaped:
        return []
    sql = build_citers_sql(escaped, depth)
    result = pg_graph.graph_sql(env, sql, extra_args=["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP])
    rows = []
    for rec in _split_records(result.stdout):
        key, title, year, kind = rec.split(FIELD_SEP, 3)
        rows.append({"key": key, "title": title, "year": int(year) if year else None, "kind": kind})
    rows.sort(key=lambda r: (r["year"] is None, r["year"]))
    return rows


# ------------------------------------------------------------- candidates --

_CENTROID_EXPR = (
    "(SELECT avg(embedding) FROM citation.work "
    "WHERE kind = 'our-document' AND embedding IS NOT NULL)"
)

# Input for external-literature triage: external-skeleton nodes ranked by
# closeness to a query (or, absent one, to the corpus centroid) and by how
# many CITES edges already tie them to our own documents -- see
# theory/external/ for what happens to a candidate once picked.
_CANDIDATES_SQL = """
WITH target AS (
    SELECT {target_expr} AS v
)
SELECT w.key, coalesce(w.year::text, ''), coalesce(w.title, ''),
       CASE WHEN w.embedding IS NULL OR target.v IS NULL THEN ''
            ELSE (1 - (w.embedding <=> target.v))::text END,
       (SELECT count(*)::text FROM citation.cites c
        JOIN citation.work wo
          ON wo.id = CASE WHEN c.citing = w.id THEN c.cited ELSE c.citing END
        WHERE (c.citing = w.id OR c.cited = w.id) AND wo.kind = 'our-document')
FROM citation.work w, target
WHERE w.kind = 'external-skeleton';
"""


def rank_candidates(rows: list[dict], min_links: int = 0, top: int | None = None) -> list[dict]:
    """Pure sort/filter step over already-scored rows (see candidates() for
    where `score`/`links` come from) -- separated out so the ranking policy
    itself (sort key, min-links cut, top-K) has a unit test against
    synthetic data, no database required. A row with score=None (no target
    embedding to compare against) cannot be ranked and is dropped.
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
        sql = _CANDIDATES_SQL.format(target_expr=":'vec'::vector")
        variables = {"vec": vec}
    else:
        sql = _CANDIDATES_SQL.format(target_expr=_CENTROID_EXPR)
        variables = {}
    result = pg_graph.graph_sql(env, sql, variables=variables,
                                 extra_args=["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP])
    rows = []
    for rec in _split_records(result.stdout):
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
                                 extra_args=["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP])
    pairs = []
    for rec in _split_records(result.stdout):
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


# ------------------------------------------------------------------ hybrid --

# The demonstration query for the AGE+pgvector stack: cypher() sits in FROM
# (the `edges` CTE), its output is cast to plain text (agtype's `::text`
# cast strips the JSON-style quoting an agtype column prints by default --
# verified against this instance's AGE 1.7.0) and JOINed against
# citation.work on `key`, alongside a pgvector nearest-neighbour CTE over
# the same table. One SQL statement, two extensions.
_HYBRID_SQL = """
WITH edges AS (
    SELECT citing_key::text AS citing_key, cited_key::text AS cited_key
    FROM ag_catalog.cypher('citation_graph', $CYPHERQ$
        MATCH (a:Work)-[:CITES]->(b:Work)
        RETURN a.key, b.key
    $CYPHERQ$) AS (citing_key agtype, cited_key agtype)
),
nearest AS (
    SELECT key, coalesce(year::text, '') AS year, coalesce(title, '') AS title,
           1 - (embedding <=> :'vec'::vector) AS score
    FROM citation.work
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> :'vec'::vector
    LIMIT :top
)
SELECT n.key, n.year, n.title, n.score::text, 'cites' AS direction,
       w.key, coalesce(w.title, '')
FROM nearest n
JOIN edges e ON e.citing_key = n.key
JOIN citation.work w ON w.key = e.cited_key
UNION ALL
SELECT n.key, n.year, n.title, n.score::text, 'cited_by' AS direction,
       w.key, coalesce(w.title, '')
FROM nearest n
JOIN edges e ON e.cited_key = n.key
JOIN citation.work w ON w.key = e.citing_key
ORDER BY 4 DESC, 1, 5;
"""


def hybrid(env, question: str, top: int = 10) -> list[dict]:
    vec = pg_search.embed_query(question, env)
    if vec is None:
        print("эмбеддинги недоступны, hybrid недоступен", file=sys.stderr)
        return []
    result = pg_graph.graph_sql(env, _HYBRID_SQL, variables={"vec": vec, "top": str(top)},
                                 extra_args=["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP])
    rows = []
    for rec in _split_records(result.stdout):
        key, year, title, score, direction, n_key, n_title = rec.split(FIELD_SEP, 6)
        rows.append({
            "key": key, "year": int(year) if year else None, "title": title,
            "score": float(score), "direction": direction,
            "neighbor_key": n_key, "neighbor_title": n_title,
        })
    return rows
