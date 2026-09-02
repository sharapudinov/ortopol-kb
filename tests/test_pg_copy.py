"""What the bulk-load seam guarantees: streaming, quoting, one script.

run_sql is monkeypatched to a canned CompletedProcess, so the temp file the
\\copy would read is written for real and can be compared byte for byte
with what the csv module produces.
"""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_copy


def _completed(stdout: str):
    return mock.Mock(stdout=stdout, returncode=0)


class _WriteSpy:
    """A temp file that remembers the size of every write it is handed."""

    def __init__(self, handle, sizes: list[int]):
        self._handle, self._sizes = handle, sizes

    def write(self, text: str) -> int:
        self._sizes.append(len(text))
        return self._handle.write(text)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


class CopyCsvRowsTests(unittest.TestCase):
    """copy_csv_rows() writes the batch row by row and runs the whole load
    -- staging DDL, the \\copy itself, the upsert that reads the staged rows
    -- as ONE psql script, hence one session and (with --single-transaction)
    one transaction.

    Both properties are the seam's reason to exist: the crawl's batches are
    the largest in the repository (a depth-2 journal is ~100k rows, a work
    row carries every raw OpenAlex record it came from), and a staging table
    that lives in that one session cannot be observed, clobbered or dropped
    by a second writer running at the same time.
    """

    def _sizes_of(self, rows, **kwargs) -> tuple[list[int], object]:
        sizes: list[int] = []
        real = tempfile.NamedTemporaryFile
        with mock.patch.object(pg_copy, "run_sql", return_value=_completed("")), \
             mock.patch.object(pg_copy.tempfile, "NamedTemporaryFile",
                                side_effect=lambda *a, **kw: _WriteSpy(real(*a, **kw), sizes)):
            result = pg_copy.copy_csv_rows({}, "citation.stage (a, b)", rows, **kwargs)
        return sizes, result

    def test_a_whole_level_never_becomes_one_string(self):
        rows = ([n, "x" * 40] for n in range(10000))
        sizes, result = self._sizes_of(rows)
        self.assertEqual(result.rows, 10000)
        self.assertLess(max(sizes), 200,
                        "the batch is written row by row, not as one buffer")

    def test_the_csv_is_what_the_csv_module_writes(self):
        rows = [["a,b", None, 'quote"inside', "line\nbreak"], [1, 2, 3, 4]]
        seen = {}

        def capture(env, sql, **kwargs):
            path = sql.split("FROM '", 1)[1].split("'", 1)[0]
            seen["text"] = Path(path).read_text(encoding="utf-8")
            seen["args"] = kwargs.get("extra_args")
            return _completed("")

        with mock.patch.object(pg_copy, "run_sql", side_effect=capture):
            result = pg_copy.copy_csv_rows({}, "citation.stage (a, b, c, d)", rows)
        expected = io.StringIO()
        csv.writer(expected, lineterminator="\n").writerows(rows)
        self.assertEqual(seen["text"], expected.getvalue())
        self.assertEqual(result.rows, 2)
        self.assertIn("--single-transaction", seen["args"])

    def test_staging_and_upsert_travel_in_the_same_script(self):
        script = {}

        def capture(env, sql, **kwargs):
            script["sql"] = sql
            return _completed("7")

        with mock.patch.object(pg_copy, "run_sql", side_effect=capture) as run_mock:
            result = pg_copy.copy_csv_rows(
                {}, "stage_work (key)", [["W1"]],
                preamble="CREATE TEMP TABLE stage_work (key TEXT) ON COMMIT DROP;\n",
                epilogue="SELECT count(*) FROM stage_work;\n")
        self.assertEqual(run_mock.call_count, 1)
        body = script["sql"]
        self.assertLess(body.index("CREATE TEMP TABLE"), body.index("\\copy"))
        self.assertLess(body.index("\\copy"), body.index("SELECT count(*)"))
        self.assertEqual(result.stdout.strip(), "7")

    def test_accepted_is_the_last_count_the_script_printed(self):
        self.assertEqual(pg_copy.CopyResult(120, "97\n").accepted(), 97)

    def test_a_script_that_printed_no_count_is_an_error_not_a_zero(self):
        with self.assertRaises(RuntimeError):
            pg_copy.CopyResult(120, "").accepted()

    def test_the_temp_file_is_gone_afterwards(self):
        seen = {}

        def capture(env, sql, **kwargs):
            seen["path"] = Path(sql.split("FROM '", 1)[1].split("'", 1)[0])
            self.assertTrue(seen["path"].is_file())
            return _completed("")

        with mock.patch.object(pg_copy, "run_sql", side_effect=capture):
            pg_copy.copy_csv_rows({}, "citation.stage (a)", [["one"]])
        self.assertFalse(seen["path"].exists())


if __name__ == "__main__":
    unittest.main()
