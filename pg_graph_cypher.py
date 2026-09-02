"""The two graph-shaped consumers of citation_graph: citers and hybrid.

Split from the relational query modules (kb/CLAUDE.md FILE_SIZE) along the
line that already ran through it: these two actually issue Cypher.
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
- both reach Postgres exclusively through `pg_graph_common.graph_sql()`,
  which applies AGE's session-local LOAD + search_path preamble once per
  psql invocation (see that module's docstring for why it cannot be baked
  into the server).

Data functions only: CLI parsing, dispatch and table printing live in
pg_graph.py, which imports this module directly -- the module that owns a
name is the module a caller imports it from, and the relational pair next
door re-exports nothing of what follows.
"""
from __future__ import annotations

import sys

import pg_graph_common
import pg_search
from citation_vocab import Relation
from pg_common import scalar, sql_literal
from pg_common import FIELD_SEP, ROW_ARGS, split_records

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


def sort_citers(rows: list[dict]) -> list[dict]:
    """Oldest first, undated last, ties broken by key.

    The key is not decoration. Cypher returns the matched vertices in the
    order AGE's label table holds them, and that order changes whenever the
    graph is reprojected (it is a bulk INSERT ... SELECT, see
    citation.project_graph), so citers of the same year came back in a
    different order after every `pg_graph.py project` -- on the same data.
    """
    return sorted(rows, key=lambda r: (r["year"] is None, r["year"] or 0, r["key"]))


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
    result = pg_graph_common.graph_sql(env, sql, extra_args=ROW_ARGS)
    rows = []
    for rec in split_records(result.stdout):
        key, title, year, kind = rec.split(FIELD_SEP, 3)
        rows.append({"key": key, "title": title, "year": int(year) if year else None, "kind": kind})
    return sort_citers(rows)



# ------------------------------------------------------------------ hybrid --

# The demonstration query for the AGE+pgvector stack: cypher() sits in FROM
# (the `edges` CTE), its output is cast to plain text (agtype's `::text`
# cast strips the JSON-style quoting an agtype column prints by default --
# verified against this instance's AGE 1.7.0) and JOINed against
# citation.work on `key`, alongside a pgvector nearest-neighbour CTE over
# the same table. One SQL statement, two extensions.
#
# The MATCH is restricted to the seed keys, and what that buys is the
# MATERIALISATION, not the traversal. AGE 1.7 indexes no vertex property by
# itself (pg_schema_citation_graph.sql's own measured note: even a btree on
# the key property does not rescue a property MATCH), so the WHERE cannot
# reach an index and the label scan happens either way. Measured on this
# instance, EXPLAIN (ANALYZE, BUFFERS) of the edges CTE with and without the
# WHERE, at 438 vertices / 2425 edges: both plans seq-scan "CITES" (2425
# rows) and both "Work" tables (438 rows each); the filtered one is a Hash
# Join with "Rows Removed by Join Filter: 2150", returning 275 rows in
# 2.9 ms against 2425 rows in 4.4 ms. So the traversal stays O(|E|) and only
# the agtype->text cast, the hashing and the joins below shrink to the seed
# neighbourhood. Worth keeping -- the answer is identical, since no edge
# outside that neighbourhood survives either join below -- but the call is
# not proportional to `top`, and the shape will have to change, not merely
# be re-filtered, once |E| starts to matter.
#
# The keys are spliced as Cypher literals, not bound: cypher()'s second
# argument must be a dollar-quoted constant (see pg_schema_citation.sql's
# own note), so there is no parameter slot inside it. They arrive
# pre-escaped from citation.cypher_literal(), the one escaping
# implementation this module trusts and never re-derives -- the same
# contract build_citers_sql() follows.
#
# The seeds arrive as a VALUES list, and this statement does NO vector work
# at all: the nearest-neighbour scan happens once, in _NEAREST_SEEDS_SQL
# below, and its answer -- key, escaped key, score -- is what gets spliced
# here. The `nearest` CTE used to re-run that scan verbatim, so every hybrid
# call paid for two top-K searches over the HNSW index and serialised the
# 1024-float query vector into two psql scripts.
# The two labels the answer carries in its `direction` column. The outgoing
# one IS the crawl's relation word and is imported rather than re-spelled
# (citation_vocab.Relation); the incoming one is its inverse and has no
# crawl_step.relation counterpart, because no journal row is ever about it.
# Substituted at import so `--show-sql` prints the statement, not a template.
_OUTGOING, _INCOMING = Relation.CITES, "cited_by"

_HYBRID_SQL = """
WITH nearest(key, score) AS (
    VALUES {seeds}
),
edges AS (
    SELECT citing_key::text AS citing_key, cited_key::text AS cited_key
    FROM ag_catalog.cypher('citation_graph', $CYPHERQ$
        MATCH (a:Work)-[:CITES]->(b:Work)
        WHERE a.key IN [{keys}] OR b.key IN [{keys}]
        RETURN a.key, b.key
    $CYPHERQ$) AS (citing_key agtype, cited_key agtype)
)
SELECT n.key, coalesce(s.year::text, ''), coalesce(s.title, ''), n.score::text,
       '{outgoing}' AS direction, w.key, coalesce(w.title, '')
FROM nearest n
JOIN citation.work s ON s.key = n.key
JOIN edges e ON e.citing_key = n.key
JOIN citation.work w ON w.key = e.cited_key
UNION ALL
SELECT n.key, coalesce(s.year::text, ''), coalesce(s.title, ''), n.score::text,
       '{incoming}' AS direction, w.key, coalesce(w.title, '')
FROM nearest n
JOIN citation.work s ON s.key = n.key
JOIN edges e ON e.cited_key = n.key
JOIN citation.work w ON w.key = e.citing_key
ORDER BY 4 DESC, 1, 5;
""".replace("{outgoing}", _OUTGOING).replace("{incoming}", _INCOMING)

# The one vector scan of a hybrid call: the seeds, their keys already
# escaped for Cypher by the database's own citation.cypher_literal(), and
# the score the statement above reports -- carried across rather than
# recomputed, so both halves of the answer describe the same `top` rows by
# construction and not by two searches agreeing.
#
# The question vector is spliced ONCE, into a MATERIALIZED CTE the scan
# reads through a LATERAL -- the shape the sibling nearest-neighbour query
# (pg_graph_candidates._CANDIDATES_SQL) already uses, and for the reason given
# there: psql expands a script variable textually, so `:'vec'` in both the
# score and the ORDER BY wrote the 1024 floats into the statement twice and
# cast them twice. AS MATERIALIZED because a single-reference CTE is
# inlined by default since PostgreSQL 12, which would put both copies back.
# LATERAL rather than a plain cross join so the ordering belongs to the
# scan: that is what keeps `Order By: embedding <=> q.v`, i.e. the form
# work_embedding_hnsw can serve, available as the graph grows.
_NEAREST_SEEDS_SQL = """
WITH q AS MATERIALIZED (
    SELECT :'vec'::vector AS v
),
nearest AS (
    SELECT n.*
    FROM q
    CROSS JOIN LATERAL (
        SELECT w.key, citation.cypher_literal(w.key) AS cypher_key,
               1 - (w.embedding <=> q.v) AS score
        FROM citation.work w
        WHERE w.embedding IS NOT NULL
        ORDER BY w.embedding <=> q.v
        LIMIT :top
    ) n
)
SELECT key, cypher_key, score::text FROM nearest ORDER BY score DESC;
"""


def hybrid_sql_template() -> str:
    """The un-substituted statement `pg_graph.py hybrid --show-sql` prints.

    A function rather than the constant itself: the constant is this
    module's private working material, and "show me the statement" is a
    request the CLI is entitled to make without reaching past an
    underscore into another module.
    """
    return _HYBRID_SQL


def build_hybrid_sql(seeds: list[tuple[str, str, str]]) -> str | None:
    """`seeds` are (key, cypher-escaped key, score) triples as
    _NEAREST_SEEDS_SQL returned them -- the escaping is the database's
    (citation.cypher_literal), this function only splices.

    Two literal forms, because the key is read twice by two languages: as a
    Cypher string inside the dollar-quoted command, and as a SQL string in
    the VALUES list, where sql_literal() is the repository's one quoting.
    The score is spliced as the source printed it, after float() has
    confirmed it IS a number -- reformatting it here would move the value.

    None when there are no seeds at all: an empty IN list is not valid
    Cypher, and an empty VALUES list is not valid SQL either.
    """
    if not seeds:
        return None
    keys = ", ".join(f"'{escaped}'" for _key, escaped, _score in seeds)
    if "$CYPHERQ$" in keys:
        raise ValueError("seed key collides with the $CYPHERQ$ delimiter")
    values = []
    for key, _escaped, score in seeds:
        float(score)  # anything but a number must not reach the statement text
        values.append(f"({sql_literal(key)}, {score}::double precision)")
    return _HYBRID_SQL.format(keys=keys, seeds=",\n           ".join(values))


def hybrid(env, question: str, top: int = 10) -> list[dict]:
    vec = pg_search.embed_query(question, env)
    if vec is None:
        print("эмбеддинги недоступны, hybrid недоступен", file=sys.stderr)
        return []
    result = pg_graph_common.graph_sql(
        env, _NEAREST_SEEDS_SQL, variables={"vec": vec, "top": str(int(top))},
        extra_args=ROW_ARGS)
    seeds = [tuple(rec.split(FIELD_SEP, 2)) for rec in split_records(result.stdout)]
    sql = build_hybrid_sql([s for s in seeds if len(s) == 3 and s[0].strip()])
    if sql is None:
        return []
    result = pg_graph_common.graph_sql(env, sql, extra_args=ROW_ARGS)
    rows = []
    for rec in split_records(result.stdout):
        key, year, title, score, direction, n_key, n_title = rec.split(FIELD_SEP, 6)
        rows.append({
            "key": key, "year": int(year) if year else None, "title": title,
            "score": float(score), "direction": direction,
            "neighbor_key": n_key, "neighbor_title": n_title,
        })
    return rows
