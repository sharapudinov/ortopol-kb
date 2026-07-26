"""Direct unit tests for pg_common.scalar_row: run_sql is monkeypatched to
return a canned CompletedProcess, so the ACTUAL "-F field_sep" split logic
and the empty-output branch run for real, unlike the deploy tests (which
stub scalar_row itself wholesale and never exercise this code).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401

import pg_common


def _completed(stdout: str):
    return mock.Mock(stdout=stdout, returncode=0)


class ScalarRowSplitTests(unittest.TestCase):
    def test_normal_multi_column_row(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("bge-m3\x1f1024")):
            row = pg_common.scalar_row({}, "SELECT model, dims FROM corpus.embedding_model;")
        self.assertEqual(row, ["bge-m3", "1024"])

    def test_single_column_row(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("42")):
            row = pg_common.scalar_row({}, "SELECT count(*) FROM corpus.documents;")
        self.assertEqual(row, ["42"])

    def test_empty_stdout_returns_empty_list(self):
        # The real falsy-empty-string branch: no matching row at all.
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            row = pg_common.scalar_row({}, "SELECT model, dims FROM corpus.embedding_model;")
        self.assertEqual(row, [])

    def test_custom_field_sep_is_honored(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("a,b,c")) as run_sql_mock:
            row = pg_common.scalar_row({}, "SELECT a, b, c;", field_sep=",")
        self.assertEqual(row, ["a", "b", "c"])
        _, kwargs = run_sql_mock.call_args
        self.assertIn(",", kwargs["extra_args"])


class ScalarRowExpectedColumnsTests(unittest.TestCase):
    """expected_columns turns a missing/malformed row into a RuntimeError
    naming the query and variables, instead of leaving the caller to hit a
    bare, contextless ValueError unpacking too few values (issue: build_
    package.py's gather_manifest used to crash uninformatively on a missing
    corpus.embedding_model row or a missing blob-probe document).
    """

    def test_matching_column_count_passes_through(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("bge-m3\x1f1024")):
            row = pg_common.scalar_row(
                {}, "SELECT model, dims FROM corpus.embedding_model;", expected_columns=2,
            )
        self.assertEqual(row, ["bge-m3", "1024"])

    def test_empty_result_with_expected_columns_raises_informative_error(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            with self.assertRaises(RuntimeError) as ctx:
                pg_common.scalar_row(
                    {}, "SELECT model, dims FROM corpus.embedding_model;",
                    variables={"doc": "1997_sm280"}, expected_columns=2,
                )
        message = str(ctx.exception)
        self.assertIn("expected 2 column", message)
        self.assertIn("1997_sm280", message)
        self.assertIn("corpus.embedding_model", message)

    def test_wrong_column_count_raises(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("only-one")):
            with self.assertRaises(RuntimeError):
                pg_common.scalar_row({}, "SELECT model, dims;", expected_columns=2)


class RowOrNoneTests(unittest.TestCase):
    """row_or_none() sits on top of scalar_row() but, unlike passing
    expected_columns to scalar_row directly, treats a genuinely empty result
    as a legitimate None rather than a RuntimeError -- callers like
    pg_search.embed_query and pg_rank_probe.nearest_page/page_rank all
    document "no such row" as a real, expected outcome distinct from "wrong
    number of columns" (a bug).
    """

    def test_well_formed_row_is_returned_as_is(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("bge-m3\x1f1024")):
            row = pg_common.row_or_none({}, "SELECT model, dims;", None, 2, "test")
        self.assertEqual(row, ["bge-m3", "1024"])

    def test_empty_result_returns_none_not_an_error(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("")):
            row = pg_common.row_or_none({}, "SELECT model, dims;", None, 2, "test")
        self.assertIsNone(row)

    def test_wrong_column_count_raises_naming_the_caller(self):
        with mock.patch.object(pg_common, "run_sql", return_value=_completed("only-one")):
            with self.assertRaises(RuntimeError) as ctx:
                pg_common.row_or_none({}, "SELECT model, dims;", None, 2, "nearest_page")
        message = str(ctx.exception)
        self.assertIn("nearest_page", message)
        self.assertIn("expected 2 column", message)


if __name__ == "__main__":
    unittest.main()
