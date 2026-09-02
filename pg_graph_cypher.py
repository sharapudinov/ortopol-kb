"""The two graph-shaped consumers of citation_graph: citers and hybrid.

Split from pg_graph_queries.py (kb/CLAUDE.md FILE_SIZE) along the line that
already ran through it: these two are the ones that actually issue Cypher.
"Who points at this node transitively" and "what does the graph say about
my nearest neighbours" are questions relational SQL answers only awkwardly;
candidates/cocitation next door are plain relational reads and share none of
what follows.

What they share, and why they live together:

- a citation.work.key spliced into a Cypher command is untrusted
  external-source text (OpenAlex and friends). Both fetch it pre-escaped
  from `citation.cypher_literal()` -- the one escaping implementation
  pg_schema_citation.sql defines, which this module trusts and never
  re-derives -- and both then check the assembled command for the
  $CYPHERQ$ delimiter, because nothing stops a key from containing it;
- both reach Postgres exclusively through `pg_graph.graph_sql()`, which
  applies AGE's session-local LOAD + search_path preamble once per psql
  invocation (see pg_graph.py's docstring for why it cannot be baked into
  the server).

Data functions only: CLI parsing, dispatch and table printing live in
pg_graph.py's main(). This module imports pg_graph at its own top level, so
pg_graph.py imports it (through pg_graph_queries) only lazily from inside
main(), or the two would form a cycle.
"""
from __future__ import annotations

import sys

import pg_graph
import pg_search
from pg_common import scalar
from pg_graph import FIELD_SEP, ROW_ARGS, split_records

MIN_DEPTH, MAX_DEPTH = 1, 3


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
    result = pg_graph.graph_sql(env, sql, extra_args=ROW_ARGS)
    rows = []
    for rec in split_records(result.stdout):
        key, title, year, kind = rec.split(FIELD_SEP, 3)
        rows.append({"key": key, "title": title, "year": int(year) if year else None, "kind": kind})
    rows.sort(key=lambda r: (r["year"] is None, r["year"]))
    return rows



# ------------------------------------------------------------------ hybrid --

# The demonstration query for the AGE+pgvector stack: cypher() sits in FROM
# (the `edges` CTE), its output is cast to plain text (agtype's `::text`
# cast strips the JSON-style quoting an agtype column prints by default --
# verified against this instance's AGE 1.7.0) and JOINed against
# citation.work on `key`, alongside a pgvector nearest-neighbour CTE over
# the same table. One SQL statement, two extensions.
#
# The traversal is BOUNDED by the seeds it will be joined to. An unfiltered
# `MATCH (a:Work)-[:CITES]->(b:Work)` materialised the entire edge set on
# every call -- cast to text, hashed and then joined against `top` (10) rows
# -- so a query touching at most ten seeds and their 1-hop neighbours paid
# for the whole graph. Restricting the MATCH to the seed keys makes the work
# proportional to `top`, and the answer is identical: no edge outside that
# neighbourhood could survive either join below.
#
# The keys are spliced as Cypher literals, not bound: cypher()'s second
# argument must be a dollar-quoted constant (see pg_schema_citation.sql's
# own note), so there is no parameter slot inside it. They arrive
# pre-escaped from citation.cypher_literal(), the one escaping
# implementation this module trusts and never re-derives -- the same
# contract build_citers_sql() follows.
_HYBRID_SQL = """
WITH edges AS (
    SELECT citing_key::text AS citing_key, cited_key::text AS cited_key
    FROM ag_catalog.cypher('citation_graph', $CYPHERQ$
        MATCH (a:Work)-[:CITES]->(b:Work)
        WHERE a.key IN [{keys}] OR b.key IN [{keys}]
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

# The seeds, and their keys already escaped for Cypher by the database's own
# citation.cypher_literal(). One round trip, reusing exactly the CTE the
# statement above re-evaluates, so the two see the same `top` rows.
_NEAREST_KEYS_SQL = """
SELECT citation.cypher_literal(key)
FROM citation.work
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :'vec'::vector
LIMIT :top;
"""


def build_hybrid_sql(escaped_keys: list[str]) -> str | None:
    """`escaped_keys` MUST already be citation.cypher_literal() output (see
    hybrid() below, which fetches them pre-escaped) -- this function only
    splices them into query text. None when there are no seeds at all: an
    empty IN list is not valid Cypher, and the answer is empty anyway.
    """
    if not escaped_keys:
        return None
    keys = ", ".join(f"'{key}'" for key in escaped_keys)
    sql = _HYBRID_SQL.format(keys=keys)
    if "$CYPHERQ$" in keys:
        raise ValueError("seed key collides with the $CYPHERQ$ delimiter")
    return sql


def hybrid(env, question: str, top: int = 10) -> list[dict]:
    vec = pg_search.embed_query(question, env)
    if vec is None:
        print("эмбеддинги недоступны, hybrid недоступен", file=sys.stderr)
        return []
    variables = {"vec": vec, "top": str(int(top))}
    seeds = pg_graph.graph_sql(env, _NEAREST_KEYS_SQL, variables=variables,
                                extra_args=["-t", "-A"]).stdout.split("\n")
    sql = build_hybrid_sql([key for key in seeds if key.strip()])
    if sql is None:
        return []
    result = pg_graph.graph_sql(env, sql, variables=variables,
                                 extra_args=ROW_ARGS)
    rows = []
    for rec in split_records(result.stdout):
        key, year, title, score, direction, n_key, n_title = rec.split(FIELD_SEP, 6)
        rows.append({
            "key": key, "year": int(year) if year else None, "title": title,
            "score": float(score), "direction": direction,
            "neighbor_key": n_key, "neighbor_title": n_title,
        })
    return rows
