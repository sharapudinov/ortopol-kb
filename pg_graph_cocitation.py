"""Co-citation over citation.cites, and the VOSviewer export it feeds.

"These two works are cited together, N times over" is a self-join with an
aggregate -- a relational question, answered here rather than through
Cypher for the reason pg_graph_candidates.py's docstring gives for its own
half. The two graph-shaped consumers (citers, hybrid) live in
pg_graph_cypher.py, and no module re-exports another.

Talks to Postgres through plain `pg_common.run_sql()`, not through the AGE
session seam: nothing here names ag_catalog, cypher() or a citation_graph
label table, so requiring `LOAD 'age'` would buy a dependency and no
answer -- including in the shipped artifact, where this module travels and
the recipient's role may not be able to load the extension at all.

Data functions only: CLI argument parsing, dispatch and table printing live
in pg_graph.py, which imports this module and is imported by nothing.
"""
from __future__ import annotations

from pathlib import Path

from pg_common import FIELD_SEP, ROW_ARGS, run_sql, split_records


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
    result = run_sql(env, _COCITATION_SQL, variables=variables,
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
