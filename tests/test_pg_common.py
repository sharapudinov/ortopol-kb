"""Direct unit tests for pg_common.scalar_row: run_sql is monkeypatched to
return a canned CompletedProcess, so the ACTUAL "-F field_sep" split logic
and the empty-output branch run for real, unlike the deploy tests (which
stub scalar_row itself wholesale and never exercise this code).

sql_literal is tested twice over: against the text it produces, and -- when
Postgres is reachable -- against what the server makes of that text.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_common
from paths import default_corpus_dir


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


class RowSeparatorsHaveOneHomeTests(unittest.TestCase):
    """The psql row/field separators are one convention, so they are one
    declaration: pg_common's, the module every psql caller already imports.

    Spelled as a check over the sources rather than agreed, because a second
    declaration costs nothing to write and is invisible until the day the
    convention changes in one place only. Tests are exempt: several of them
    stub the wire format psql produces and have to spell the bytes they are
    imitating.
    """

    ROOT = Path(pg_common.__file__).resolve().parent
    # The byte itself, and the two SQL spellings of it. Prose about the
    # separators is fine anywhere; only a value is a second declaration,
    # which is why this reads string constants and not the file's text.
    SPELLINGS = ("\x1f", "\x1e", "chr(31)", "chr(30)")

    def _sources(self):
        for directory in (self.ROOT, self.ROOT / "citations", self.ROOT / "deploy"):
            for path in sorted(directory.glob("*.py")):
                if path.name != "pg_common.py":
                    yield path

    def test_no_other_module_spells_a_separator_byte(self):
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            constants = [node.value for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            for spelling in self.SPELLINGS:
                self.assertTrue(
                    all(spelling not in text for text in constants),
                    f"{path.name} spells {spelling!r}: import it from pg_common")

    def test_no_other_module_declares_the_separator_names(self):
        owned = {"FIELD_SEP", "RECORD_SEP", "ROW_ARGS"}
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            assigned = {
                target.id
                for node in ast.walk(tree) if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
            }
            self.assertEqual(assigned & owned, set(),
                             f"{path.name}: import the separators from pg_common")

    def test_the_row_args_are_the_flags_that_produce_them(self):
        self.assertEqual(
            pg_common.ROW_ARGS,
            ["-t", "-A", "-F", pg_common.FIELD_SEP, "-R", pg_common.RECORD_SEP])

    def test_split_records_keeps_a_multi_line_cell_in_one_record(self):
        stdout = (f"first{pg_common.FIELD_SEP}line one\nline two{pg_common.RECORD_SEP}"
                  f"second{pg_common.FIELD_SEP}plain{pg_common.RECORD_SEP}")
        self.assertEqual(
            pg_common.split_records(stdout),
            [f"first{pg_common.FIELD_SEP}line one\nline two",
             f"second{pg_common.FIELD_SEP}plain"])


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


class SqlLiteralTests(unittest.TestCase):
    """psql script variables (:'name') cover a single value; a list of them
    inside one statement has no binding form, so the literal has to be built
    -- once, here, rather than with an f-string at each call site.
    """

    def test_plain_value_is_quoted(self):
        self.assertEqual(pg_common.sql_literal("30 days"), "E'30 days'")

    def test_single_quote_is_escaped(self):
        self.assertEqual(pg_common.sql_literal("o'zna"), r"E'o\'zna'")

    def test_backslash_is_doubled_before_anything_else(self):
        # Order matters: a backslash inserted while escaping a quote must
        # not itself get doubled afterwards.
        self.assertEqual(pg_common.sql_literal(r"a\'b"), r"E'a\\\'b'")

    def test_nul_byte_is_refused_rather_than_silently_truncated(self):
        with self.assertRaises(ValueError):
            pg_common.sql_literal("a\x00b")


class SqlLiteralRoundTripTests(unittest.TestCase):
    """The only assertion that matters in the end: what the SERVER reads
    back out of the literal is the string that went in.
    """

    HOSTILE = ["o'zna", r"back\slash", r"both\'kinds", 'quote"and\'apostrophe', "перенос\nстроки"]

    @classmethod
    def setUpClass(cls):
        try:
            env = pg_common.load_pgenv(default_corpus_dir() / ".pgenv")
        except pg_common.PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not pg_common.check_postgres_available(env):
            raise unittest.SkipTest("Postgres not reachable")
        cls.env = env

    def test_hostile_strings_survive_the_round_trip(self):
        values = ", ".join(f"({pg_common.sql_literal(v)})" for v in self.HOSTILE)
        out = pg_common.run_sql(
            self.env,
            f"SELECT string_agg(v, chr(31) ORDER BY ord) FROM "
            f"(SELECT row_number() OVER () AS ord, v FROM (VALUES {values}) AS t(v)) s;",
            extra_args=["-t", "-A"],
        ).stdout
        # The newline case makes the whole output multi-line; only the
        # separator-joined payload is compared, not the line structure.
        self.assertEqual(out.rstrip("\n").split("\x1f"), self.HOSTILE)


if __name__ == "__main__":
    unittest.main()
