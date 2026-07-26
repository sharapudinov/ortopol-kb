"""Unit tests for deploy/pg_rank_probe.py: no live database.
run_sql is stubbed at pg_common's own module level (not
pg_rank_probe.row_or_none), so the real chain -- row_or_none's None/shape
handling, scalar_row's FIELD_SEP split, and nearest_page/page_rank's own
int()/float() casts and column unpacking -- all run for real under test,
unlike the deploy tests (test_manifest_probe.py, test_deploy_units.py),
which stub nearest_page/page_rank themselves wholesale.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import pg_common
import pg_rank_probe


def _completed(stdout: str):
    return mock.Mock(stdout=stdout, returncode=0)


class ExactScanSqlTests(unittest.TestCase):
    """f9f6548e: exactness must be a property of the query text (SET LOCAL
    disabling the HNSW-satisfiable index paths), not of planner cost-model
    behaviour that happens to pick the exact sort today.
    """

    def test_nearest_page_sql_disables_index_scans_before_the_ranked_cte(self):
        set_local_pos = pg_rank_probe._NEAREST_PAGE_SQL.index("SET LOCAL enable_indexscan")
        cte_pos = pg_rank_probe._NEAREST_PAGE_SQL.index("WITH dist")
        self.assertIn("SET LOCAL enable_indexonlyscan", pg_rank_probe._NEAREST_PAGE_SQL)
        self.assertLess(set_local_pos, cte_pos)

    def test_page_rank_sql_disables_index_scans_before_the_ranked_cte(self):
        set_local_pos = pg_rank_probe._PAGE_RANK_SQL.index("SET LOCAL enable_indexscan")
        cte_pos = pg_rank_probe._PAGE_RANK_SQL.index("WITH dist")
        self.assertIn("SET LOCAL enable_indexonlyscan", pg_rank_probe._PAGE_RANK_SQL)
        self.assertLess(set_local_pos, cte_pos)

    def test_both_statements_wrap_the_set_locals_in_their_own_transaction(self):
        # SET LOCAL outside an explicit transaction block only lasts for the
        # implicit single-statement transaction psql would otherwise open --
        # i.e. it would apply to nothing. BEGIN/COMMIT make it apply to the
        # SELECT that follows.
        for sql in (pg_rank_probe._NEAREST_PAGE_SQL, pg_rank_probe._PAGE_RANK_SQL):
            self.assertTrue(sql.strip().startswith("BEGIN;"))
            self.assertIn("COMMIT;", sql)
            self.assertLess(sql.index("BEGIN;"), sql.index("SET LOCAL"))


class DeterministicOrderingTests(unittest.TestCase):
    """2baf1ba4: row_number() ordered on a computed float alone has no
    guaranteed tiebreak among equal-distance rows (real here: scanned/blank
    pages routinely share an embedding). Both the window function's ORDER BY
    and the outer ORDER BY must carry the same total, data-derived tiebreak.
    """

    def test_window_function_orders_by_distance_then_document_id_then_page_number(self):
        window_clause = pg_rank_probe._RANKED_PAGES_CTE[
            pg_rank_probe._RANKED_PAGES_CTE.index("row_number() OVER"):
            pg_rank_probe._RANKED_PAGES_CTE.index(") AS rnk")
        ]
        # References the already-computed `distance` column (see
        # DistanceComputedOnceTests below), not the raw cosine expression --
        # 238d8f09 moved that expression into the inner `dist` CTE so it is
        # written, and evaluated, exactly once.
        distance_pos = window_clause.index("distance")
        doc_pos = window_clause.index("document_id")
        page_pos = window_clause.index("page_number")
        self.assertLess(distance_pos, doc_pos)
        self.assertLess(doc_pos, page_pos)

    def test_nearest_page_sql_outer_order_by_carries_the_same_tiebreak(self):
        outer_order_by = pg_rank_probe._NEAREST_PAGE_SQL[
            pg_rank_probe._NEAREST_PAGE_SQL.rindex("ORDER BY"):
        ]
        self.assertIn("rnk, document_id, page_number", outer_order_by)

    def test_ranked_cte_selects_document_id_directly_without_joining_documents(self):
        # a5b57ed4: p.document_id is already in hand (FK to corpus.documents);
        # joining that table just to re-derive it forces a seq-scan of a
        # multi-megabyte-blob table while index scans are disabled.
        self.assertNotIn("JOIN corpus.documents", pg_rank_probe._RANKED_PAGES_CTE)
        self.assertIn("SELECT p.document_id, p.page_number", pg_rank_probe._RANKED_PAGES_CTE)


class DistanceComputedOnceTests(unittest.TestCase):
    """238d8f09: the cosine-distance expression must appear exactly once in
    the ranked CTE. Previously it was written twice -- once in the target
    list, once again in row_number()'s ORDER BY -- and Postgres does not
    common-subexpression-eliminate across a target list and a window sort
    key, so every embedded page paid for the distance computation twice on
    every call, over the whole table, with index scans forced off.
    """

    def test_distance_expression_appears_exactly_once(self):
        self.assertEqual(
            pg_rank_probe._RANKED_PAGES_CTE.count("p.embedding <=> :'vec'::vector"), 1,
        )


class RunnerUpDistanceTests(unittest.TestCase):
    def test_parses_a_well_formed_row(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("0.5")):
            result = pg_rank_probe.runner_up_distance({}, "[0.1]")
        self.assertEqual(result, 0.5)

    def test_fewer_than_two_embedded_pages_returns_none(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            result = pg_rank_probe.runner_up_distance({}, "[0.1]")
        self.assertIsNone(result)

    def test_selects_rnk_equals_2_from_the_ranked_cte(self):
        self.assertIn("WHERE rnk = 2", pg_rank_probe._RUNNER_UP_DISTANCE_SQL)


class NearestPageTests(unittest.TestCase):
    def test_parses_a_well_formed_row(self):
        row = "1997_sm280\x1f7\x1f0.123456\x1f1"
        with mock.patch.object(pg_common, "run_sql", return_value=_completed(row)):
            result = pg_rank_probe.nearest_page({}, "[0.1]")
        self.assertEqual(result, {
            "document_id": "1997_sm280", "page_number": 7, "distance": 0.123456, "rank": 1,
        })

    def test_empty_stdout_returns_none(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            result = pg_rank_probe.nearest_page({}, "[0.1]")
        self.assertIsNone(result)

    def test_malformed_row_raises_informative_error_not_a_bare_value_error(self):
        row = "1997_sm280\x1f7\x1f0.123456"  # missing rnk column
        with mock.patch.object(pg_common, "run_sql", return_value=_completed(row)):
            with self.assertRaises(RuntimeError) as ctx:
                pg_rank_probe.nearest_page({}, "[0.1]")
        self.assertIn("nearest_page", str(ctx.exception))
        self.assertIn("expected 4", str(ctx.exception))


class PageRankTests(unittest.TestCase):
    def test_parses_a_well_formed_row(self):
        row = "2015_demr1\x1f69\x1f0.4\x1f3"
        with mock.patch.object(pg_common, "run_sql", return_value=_completed(row)):
            result = pg_rank_probe.page_rank({}, "[0.1]", "2015_demr1", 69)
        self.assertEqual(result, {
            "document_id": "2015_demr1", "page_number": 69, "distance": 0.4, "rank": 3,
        })

    def test_empty_stdout_returns_none(self):
        # No embedding for this (document, page) pair, or the pair no
        # longer exists -- a real, expected outcome, not a malformed row.
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            result = pg_rank_probe.page_rank({}, "[0.1]", "2015_demr1", 69)
        self.assertIsNone(result)

    def test_malformed_row_raises_informative_error(self):
        row = "2015_demr1\x1f69\x1f0.4\x1f3\x1fextra"
        with mock.patch.object(pg_common, "run_sql", return_value=_completed(row)):
            with self.assertRaises(RuntimeError) as ctx:
                pg_rank_probe.page_rank({}, "[0.1]", "2015_demr1", 69)
        self.assertIn("page_rank", str(ctx.exception))

    def test_page_number_and_document_id_are_passed_as_query_variables(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")) as run_sql_mock:
            pg_rank_probe.page_rank({}, "[0.1]", "2015_demr1", 69)
        _, kwargs = run_sql_mock.call_args
        self.assertEqual(kwargs["variables"], {"vec": "[0.1]", "doc": "2015_demr1", "page": "69"})


if __name__ == "__main__":
    unittest.main()
