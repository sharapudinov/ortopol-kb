"""Exact (not HNSW-approximate) nearest-page and page-rank probes.

Split out of pg_search.py (module size) and living here in deploy/, not
beside pg_search.py/pg_common.py at the repository root: these two
functions serve deploy/ specifically, not general corpus search, and
every non-test caller (manifest_probe.py, vector_probe_check.py,
drift_probe.py) already lives in this same directory -- see
artifact_bundle.py's DEPLOY_FILES, not CORPUS_LIB_FILES.
build_package.py's manifest_probe records the nearest page to a probe query
as its reference (nearest_page), smoke_checks.py later re-derives where
that same (document, page) pair ranks against a freshly computed query
vector (page_rank). Both queries are built from the SAME CTE text, so
"nearest" cannot silently mean two different things (e.g. an
HNSW-approximate top-1 on the build side vs. an exact full-scan ranking on
the verify side).

The CTE having no LIMIT is not enough on its own to guarantee an exact
sort: corpus.pages carries an HNSW index (pg_schema.sql), and a
single-referenced CTE is inlinable, so the planner is free to satisfy the
window function's ORDER BY from that index's approximate candidate list.
SET LOCAL below forces the exact path as a property of the query text
itself rather than of planner cost-model behaviour: both statements run
inside their own BEGIN/COMMIT specifically so the two SET LOCALs apply to
the SELECT that follows them, not to nothing (SET LOCAL outside an
explicit transaction block only lasts for the remainder of the implicit
single-statement transaction psql would otherwise open per statement).

document_id comes straight from corpus.pages.document_id (no JOIN to
corpus.documents): it is the same value by FK definition, and joining a
table that carries a multi-megabyte source_blob column just to re-derive
a column already in hand forces a seq-scan of that table while
_FORCE_EXACT_SCAN_SQL has index scans disabled.

Both the window function's ORDER BY and the outer ORDER BY carry the same
total, data-derived tiebreak (document_id, page_number) after the
distance itself: distance is a computed float, and scanned/blank/
boilerplate pages routinely share an identical embedding and therefore an
identical distance to any given query vector. Without a deterministic
tiebreak, row_number()'s tie-breaking among equal-distance rows is the
executor's unstable sort order -- two runs over identical data can assign
different ranks to a tied pair, which is exactly the rank build_package.py
records and smoke_checks.py later re-derives and compares.
"""
from __future__ import annotations

from pg_common import row_or_none

# distance is computed once, in the inner `dist` CTE, and `ranked` only
# ever refers to that already-computed column -- never rewrites the cosine
# expression itself. Before this split the expression was written twice
# (once in ranked's own target list, once again in row_number()'s ORDER
# BY): Postgres does not common-subexpression-eliminate across a target
# list and a window sort key, so every embedded page paid for the cosine
# distance twice on every call, over the whole table, on the hot path of
# both a package build (once) and a smoke run (once per manifest check,
# n times per drift_probe.measure_drift() repeat).
_RANKED_PAGES_CTE = """
WITH dist AS (
    SELECT p.document_id, p.page_number,
           p.embedding <=> :'vec'::vector AS distance
    FROM corpus.pages p
    WHERE p.embedding IS NOT NULL
), ranked AS (
    SELECT document_id, page_number, distance,
           row_number() OVER (
               ORDER BY distance, document_id, page_number
           ) AS rnk
    FROM dist
)
"""

_FORCE_EXACT_SCAN_SQL = """
SET LOCAL enable_indexscan = off;
SET LOCAL enable_indexonlyscan = off;
"""

_NEAREST_PAGE_SQL = "BEGIN;\n" + _FORCE_EXACT_SCAN_SQL + _RANKED_PAGES_CTE + """
SELECT document_id, page_number, distance, rnk
FROM ranked
ORDER BY rnk, document_id, page_number
LIMIT 1;
COMMIT;
"""

_PAGE_RANK_SQL = "BEGIN;\n" + _FORCE_EXACT_SCAN_SQL + _RANKED_PAGES_CTE + """
SELECT document_id, page_number, distance, rnk
FROM ranked
WHERE document_id = :'doc' AND page_number = :page;
COMMIT;
"""

_RUNNER_UP_DISTANCE_SQL = "BEGIN;\n" + _FORCE_EXACT_SCAN_SQL + _RANKED_PAGES_CTE + """
SELECT distance
FROM ranked
WHERE rnk = 2;
COMMIT;
"""


def nearest_page(env: dict[str, str], vec_json: str) -> dict | None:
    """The single nearest page to vec_json by cosine distance (exact, not
    HNSW-approximate -- see the module docstring). rank is always 1 by
    construction; returned for symmetry with page_rank()'s result shape.

    None if corpus.pages has no embedded rows at all, mirroring page_rank().
    """
    row = row_or_none(env, _NEAREST_PAGE_SQL, variables={"vec": vec_json},
                       expected_columns=4, what="nearest_page")
    if row is None:
        return None
    doc_id, page_number, distance, rnk = row
    return {"document_id": doc_id, "page_number": int(page_number),
            "distance": float(distance), "rank": int(rnk)}


def page_rank(env: dict[str, str], vec_json: str, document_id: str,
              page_number: int) -> dict | None:
    """Rank (1-based) and cosine distance of one specific (document_id,
    page_number) pair among ALL embedded pages, ordered by distance to
    vec_json. None if the pair has no embedding (or no longer exists).
    """
    row = row_or_none(
        env, _PAGE_RANK_SQL,
        variables={"vec": vec_json, "doc": document_id, "page": str(int(page_number))},
        expected_columns=4, what="page_rank",
    )
    if row is None:
        return None
    doc_id, page_no, distance, rnk = row
    return {"document_id": doc_id, "page_number": int(page_no),
            "distance": float(distance), "rank": int(rnk)}


def runner_up_distance(env: dict[str, str], vec_json: str) -> float | None:
    """Cosine distance of the SECOND-nearest page (rnk = 2), from the same
    ranked CTE nearest_page()/page_rank() use.

    manifest_probe.gather_manifest() records this alongside the rank-1
    reference: the gap between the two is the actual margin a cross-process
    embedding perturbation has to cross before rank 1 and rank 2 could
    genuinely swap, which is the numeric quantity
    smoke_checks.VECTOR_PROBE_RANK_TOLERANCE is meant to bound -- a rank
    tolerance with no recorded margin is unfalsifiable: it cannot be told
    apart from one that happens to admit a real nearest-neighbour swap.

    None when fewer than two pages are embedded (mirrors nearest_page's/
    page_rank's own None-on-empty-result contract).
    """
    row = row_or_none(env, _RUNNER_UP_DISTANCE_SQL, variables={"vec": vec_json},
                       expected_columns=1, what="runner_up_distance")
    if row is None:
        return None
    return float(row[0])
