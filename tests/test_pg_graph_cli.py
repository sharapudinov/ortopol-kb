"""pg_graph.py's own wiring: what main() does with argv, and what it
returns when something refuses.

Split off test_pg_graph.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that file tests the plumbing this CLI drives, and every
function reached from here is well covered there -- which is exactly why
the wiring itself could break unnoticed. A --pgenv fallback that stopped
falling back, a dispatch table that swallowed an exception instead of
returning 2, a subcommand no longer reaching its handler: none of that
touches a tested function.

Nothing here reaches Postgres: load_pgenv, pg_graph_common and the DISPATCH
handlers are stubbed, so what is asserted is the dispatch and the exit
codes, not what the database would have said.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_graph
from pg_common import PostgresUnavailable

ENV = {"PGUSER": "ortopol"}
PGENV = ["--pgenv", "/tmp/does-not-matter/.pgenv"]


class SchemaAndProjectionTests(unittest.TestCase):
    """The two subcommands that are not in DISPATCH, because they answer
    with a number of their own rather than a table.
    """

    def test_init_applies_the_schema(self):
        with mock.patch.object(pg_graph, "load_pgenv", return_value=ENV), \
             mock.patch.object(pg_graph.pg_graph_common, "init_schema") as init, \
             redirect_stdout(io.StringIO()) as out:
            code = pg_graph.main([*PGENV, "init"])
        self.assertEqual(code, 0)
        init.assert_called_once_with(ENV)
        self.assertIn("citation schema applied", out.getvalue())

    def test_project_rebuilds_and_reports_both_counts(self):
        with mock.patch.object(pg_graph, "load_pgenv", return_value=ENV), \
             mock.patch.object(pg_graph.pg_graph_common, "project",
                               return_value=(441, 2427)) as project, \
             mock.patch.object(pg_graph.pg_graph_common, "check") as check, \
             redirect_stdout(io.StringIO()) as out:
            code = pg_graph.main([*PGENV, "project"])
        self.assertEqual(code, 0)
        project.assert_called_once_with(ENV)
        check.assert_not_called()
        self.assertIn("V=441 E=2427", out.getvalue())

    def test_project_check_compares_instead_of_rebuilding(self):
        """--check is the read-only half, and its verdict IS the exit code:
        a stale projection has to fail a script, not print at it.
        """
        with mock.patch.object(pg_graph, "load_pgenv", return_value=ENV), \
             mock.patch.object(pg_graph.pg_graph_common, "project") as project, \
             mock.patch.object(pg_graph.pg_graph_common, "check", return_value=1) as check:
            code = pg_graph.main([*PGENV, "project", "--check"])
        self.assertEqual(code, 1)
        check.assert_called_once_with(ENV)
        project.assert_not_called()


class DispatchTests(unittest.TestCase):
    """Every DISPATCH entry is reached by the subcommand that names it.

    The handlers themselves are tested against their queries elsewhere; what
    is unpinned without this is the table -- a renamed subparser, a key that
    no longer matches, a handler wired to the wrong name.
    """

    ARGV = {
        "citers": ["citers", "1997_sm280"],
        "candidates": ["candidates", "--top", "5"],
        "cocitation": ["cocitation", "--min-count", "3"],
        "hybrid": ["hybrid", "вопрос"],
    }

    def test_the_table_names_every_subcommand_that_has_a_handler(self):
        self.assertEqual(sorted(pg_graph.DISPATCH), sorted(self.ARGV))

    def test_each_subcommand_reaches_its_own_handler(self):
        for command, argv in self.ARGV.items():
            with self.subTest(command=command):
                seen = []

                def handler(args, env, _command=command):
                    seen.append((_command, args.command, env))
                    return 0

                with mock.patch.object(pg_graph, "load_pgenv", return_value=ENV), \
                     mock.patch.dict(pg_graph.DISPATCH, {command: handler}):
                    code = pg_graph.main([*PGENV, *argv])
                self.assertEqual(code, 0)
                self.assertEqual(seen, [(command, command, ENV)])

    def test_a_refused_argument_is_exit_2_and_says_why(self):
        """The handlers raise ValueError for an argument the query cannot
        honour. The CLI owes that a code of its own and the message on
        stderr -- a bare traceback tells a script nothing.
        """
        def handler(_args, _env):
            raise ValueError("--top должен быть положительным")

        with mock.patch.object(pg_graph, "load_pgenv", return_value=ENV), \
             mock.patch.dict(pg_graph.DISPATCH, {"candidates": handler}), \
             redirect_stderr(io.StringIO()) as err:
            code = pg_graph.main([*PGENV, "candidates"])
        self.assertEqual(code, 2)
        self.assertIn("--top должен быть положительным", err.getvalue())

    def test_an_unknown_command_is_argparse_s_refusal(self):
        with redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as ctx:
                pg_graph.main([*PGENV, "citations"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("citations", err.getvalue())

    def test_no_command_at_all_is_refused_too(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pg_graph.main(PGENV)


class PgenvTests(unittest.TestCase):
    """Where the connection comes from when --pgenv is not given.

    Two ways that can fail, and both have to name themselves: the bundled
    artifact has no paths.py at all (the CLI is driven with --pgenv there),
    and a checkout run from outside the data tree has paths.py but nothing
    to find.
    """

    def test_the_default_pgenv_is_the_corpus_directorys(self):
        with mock.patch.object(pg_graph, "default_corpus_dir",
                               return_value=Path("/data/corpus")), \
             mock.patch.object(pg_graph, "load_pgenv", return_value=ENV) as load, \
             mock.patch.object(pg_graph.pg_graph_common, "init_schema"), \
             redirect_stdout(io.StringIO()):
            code = pg_graph.main(["init"])
        self.assertEqual(code, 0)
        load.assert_called_once_with(Path("/data/corpus/.pgenv"))

    def test_without_paths_py_the_missing_pgenv_is_named(self):
        with mock.patch.object(pg_graph, "default_corpus_dir", None), \
             redirect_stderr(io.StringIO()) as err:
            code = pg_graph.main(["init"])
        self.assertEqual(code, 1)
        self.assertIn("--pgenv", err.getvalue())

    def test_outside_the_data_tree_the_reason_is_reported(self):
        with mock.patch.object(pg_graph, "default_corpus_dir",
                               side_effect=RuntimeError("нет theory/iis рядом")), \
             redirect_stderr(io.StringIO()) as err:
            code = pg_graph.main(["init"])
        self.assertEqual(code, 1)
        self.assertIn("нет theory/iis рядом", err.getvalue())

    def test_an_unreachable_postgres_is_exit_1_with_its_own_message(self):
        with mock.patch.object(pg_graph, "load_pgenv",
                               side_effect=PostgresUnavailable("нет .pgenv")), \
             mock.patch.object(pg_graph.pg_graph_common, "init_schema") as init, \
             redirect_stderr(io.StringIO()) as err:
            code = pg_graph.main([*PGENV, "init"])
        self.assertEqual(code, 1)
        self.assertIn("нет .pgenv", err.getvalue())
        init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
