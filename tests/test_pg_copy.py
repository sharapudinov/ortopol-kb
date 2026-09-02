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


class CopyCsvIntoTests(unittest.TestCase):
    """The form the corpus loaders use: CSV already built as a string.

    Six of them go through it, and nothing exercised it -- neither the
    quoting of the temp path spliced into the \\copy metacommand nor the
    unlink when run_sql raises, which is the one path that leaks a file
    per failed load.
    """

    def _run(self, csv_text: str, side_effect=None):
        seen = {}

        def capture(env, sql, **kwargs):
            seen["sql"] = sql
            seen["path"] = Path(sql.split("FROM '", 1)[1].split("' WITH", 1)[0]
                                .replace("''", "'"))
            seen["content"] = seen["path"].read_text(encoding="utf-8")
            if side_effect is not None:
                raise side_effect
            return _completed("COPY 2\n")

        with mock.patch.object(pg_copy, "run_sql", side_effect=capture):
            if side_effect is None:
                seen["result"] = pg_copy.copy_csv_into(
                    {}, "corpus.pages (document_id, body)", csv_text)
            else:
                with self.assertRaises(type(side_effect)):
                    pg_copy.copy_csv_into(
                        {}, "corpus.pages (document_id, body)", csv_text)
        return seen

    def test_the_file_carries_exactly_what_the_caller_built(self):
        text = ('2009_isu34,"тело, с запятой"\n'
                '2009_isu34,"кавычка "" внутри"\n')
        seen = self._run(text)
        self.assertEqual(seen["content"], text)
        # And it parses back as the same two rows the caller wrote.
        self.assertEqual(list(csv.reader(io.StringIO(seen["content"]))),
                         [["2009_isu34", "тело, с запятой"],
                          ["2009_isu34", 'кавычка " внутри']])
        self.assertEqual(seen["result"].stdout, "COPY 2\n")

    def test_the_metacommand_names_the_table_and_the_csv_format(self):
        seen = self._run("2009_isu34,текст\n")
        self.assertTrue(seen["sql"].startswith("\\copy corpus.pages (document_id, body) "))
        self.assertIn("WITH (FORMAT csv)", seen["sql"])

    def test_a_quote_in_the_path_is_doubled_not_left_to_break_the_command(self):
        """psql reads the path as a single-quoted literal, so an apostrophe
        in the temp directory would end it early -- the load would then
        fail, or worse, read a path nobody chose.
        """
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / "don't"
            odd.mkdir()
            with mock.patch.object(pg_copy.tempfile, "tempdir", str(odd)):
                seen = self._run("a,b\n")
        self.assertIn("don''t", seen["sql"])
        self.assertNotIn("don't", seen["sql"])

    def test_the_temp_file_is_gone_even_when_the_load_fails(self):
        seen = self._run("a,b\n", side_effect=RuntimeError("COPY failed"))
        self.assertFalse(seen["path"].exists(),
                         "провалившаяся загрузка оставила временный файл")

    def test_the_temp_file_exists_while_psql_reads_it(self):
        seen = self._run("a,b\n")
        self.assertFalse(seen["path"].exists())
        self.assertEqual(seen["content"], "a,b\n")


if __name__ == "__main__":
    unittest.main()
