"""Where the crawl's pieces are allowed to live, checked instead of agreed.

The citations package declares a split in its own __init__ (clients do HTTP,
inputs.py does the reads that establish a run, store.py is the write seam,
crawl.py is the traversal) and paths.py declares that 'theory/iis' is spelled
once. Both are the kind of rule a new call site breaks silently: a duplicated
source_dir literal returns zero rows rather than failing, and pipeline code
in the argparse module simply cannot be called without argparse.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import paths
import pg_load_citations
from citations import inputs
from citations.spike_runs import DryRunMeasurementsWriter, MeasurementsWriter
from citations.store import DryRunWriter, PostgresWriter, Writer
from pg_common import FIELD_SEP, RECORD_SEP

CITATIONS_DIR = Path(inputs.__file__).resolve().parent
LOADER = Path(pg_load_citations.__file__).resolve()
SOURCES = sorted(CITATIONS_DIR.glob("*.py")) + [LOADER]


class SourceDirLiteralTests(unittest.TestCase):
    """paths.IIS_SOURCE_DIR exists so the string appears once. A second
    spelling of it queries corpus.documents.source_dir for a directory
    nothing was loaded under: zero rows, no error, a seed set silently
    emptied and a twin index silently unanchored.
    """

    def test_no_module_spells_the_source_directory_itself(self):
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                paths.IIS_SOURCE_DIR, text,
                f"{path.name}: import paths.IIS_SOURCE_DIR instead of the literal",
            )

    def test_the_default_is_the_constant_itself(self):
        tree = ast.parse((CITATIONS_DIR / "inputs.py").read_text(encoding="utf-8"))
        defaults = {
            node.name: [d for d in node.args.defaults]
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in ("corpus_seed_documents", "corpus_document_ids"):
            self.assertTrue(
                any(isinstance(d, ast.Name) and d.id == "IIS_SOURCE_DIR"
                    for d in defaults[name]),
                f"{name}: source_dir must default to paths.IIS_SOURCE_DIR",
            )


class SeedPredicateTests(unittest.TestCase):
    """"What counts as a seed document" is one predicate. Written twice, a
    change applied to one copy produces seeds with no Math-Net title anchor
    -- which is precisely what the twin rule depends on.
    """

    PREDICATE = "extraction_state <> 'metadata'"

    def test_the_predicate_appears_in_exactly_one_module(self):
        carriers = [p.name for p in SOURCES
                    if self.PREDICATE in p.read_text(encoding="utf-8")]
        self.assertEqual(carriers, ["inputs.py"])

    def test_both_readings_go_through_that_one_query(self):
        seen = []
        rows = (f"doc_a{FIELD_SEP}https://mathnet.ru/eng/sm123{RECORD_SEP}"
                f"doc_b{FIELD_SEP}{RECORD_SEP}")

        class _Result:
            stdout = rows

        def fake_run_sql(env, sql, **kwargs):
            seen.append(sql)
            return _Result()

        original = inputs.run_sql
        inputs.run_sql = fake_run_sql
        try:
            self.assertEqual(inputs.corpus_document_ids({}), ["doc_a", "doc_b"])
            self.assertEqual(
                inputs.corpus_seed_documents({}),
                [("doc_a", "https://mathnet.ru/eng/sm123"), ("doc_b", "")],
            )
        finally:
            inputs.run_sql = original
        self.assertEqual(seen, [inputs._SEED_DOCUMENTS_SQL] * 2)


class RowProtocolTests(unittest.TestCase):
    """The psql row protocol is pg_common's, for the crawl too.

    A title, an abstract or a reason can carry a comma, a tab and a newline,
    and third-party titles from OpenAlex/zbMATH/Math-Net are exactly what
    these readers select. FIELD_SEP/RECORD_SEP/ROW_ARGS/split_records() are
    one contract in one place; a module re-spelling "\x1f" and splitting on
    "\n" answers a change to that contract by not noticing it, and unpacks a
    multi-line title into the wrong positions in the meantime.
    """

    MODULES = sorted(CITATIONS_DIR.glob("*.py"))

    def test_no_module_spells_a_separator_itself(self):
        for path in self.MODULES:
            text = path.read_text(encoding="utf-8")
            for sep, name in (("\x1f", "FIELD_SEP"), ("\x1e", "RECORD_SEP")):
                self.assertNotIn(
                    sep, text,
                    f"{path.name}: import {name} from pg_common")

    def test_no_module_cuts_psql_output_into_lines(self):
        for path in self.MODULES:
            text = path.read_text(encoding="utf-8")
            for spelling in ('.split("\n")', ".splitlines("):
                self.assertNotIn(
                    spelling, text,
                    f"{path.name}: rows come from split_records(), not from lines")

    def test_every_read_asks_psql_for_the_shared_flags(self):
        """extra_args is ROW_ARGS or the read is a scalar (pg_common.scalar).

        A hand-written flag list is the other half of the same drift: -F
        without -R produces exactly the newline-delimited output the parse
        above must not assume.
        """
        for path in self.MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.keyword) or node.arg != "extra_args":
                    continue
                self.assertTrue(
                    isinstance(node.value, ast.Name) and node.value.id == "ROW_ARGS",
                    f"{path.name}: extra_args must be pg_common.ROW_ARGS")


class ScoringIsDependencyFreeTests(unittest.TestCase):
    """citations/scoring.py is arithmetic, and the split is only worth
    anything while it stays that: the moment a store read or an HTTP call
    appears in it, the module the calibration imports for its numbers is
    back to carrying a seam, which is the state it was split out of.
    """

    ALLOWED = {"math", "__future__", "typing"}

    def test_scoring_imports_nothing_but_the_standard_arithmetic(self):
        tree = ast.parse((CITATIONS_DIR / "scoring.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], self.ALLOWED)
            elif isinstance(node, ast.ImportFrom):
                self.assertIsNotNone(node.module, "относительный импорт в чистой математике")
                self.assertIn(node.module.split(".")[0], self.ALLOWED,
                              f"scoring.py: {node.module} — это уже не арифметика")
                self.assertEqual(node.level, 0, "относительный импорт в чистой математике")


class DependencyDirectionTests(unittest.TestCase):
    """The crawl package depends on the repository's shared modules, never
    the other way round.

    pg_embed.py covers `pages`, `spikes` and `works`: two of the three have
    nothing to do with citations, and an import of citations/ made the whole
    tool unimportable without the crawl package -- which deploy/
    artifact_bundle.py deliberately does not ship. A shared encoder belongs
    in pg_common.py beside sql_literal, not reached for through a module
    whose own docstring calls itself store.py's plumbing.

    pg_load_citations.py is the one exception, and by definition: it IS the
    crawl's command line.
    """

    ROOT = CITATIONS_DIR.parent
    DISPATCHER = "pg_load_citations.py"

    def test_only_the_crawls_own_cli_imports_the_crawl_package(self):
        for path in sorted(self.ROOT.glob("*.py")):
            if path.name == self.DISPATCHER:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotEqual(
                        name.split(".")[0], CITATIONS_DIR.name,
                        f"{path.name}: {name} -- общий модуль не тянет пакет обхода")

    def test_the_dispatcher_still_does(self):
        """The complement: the guard must not pass because nothing imports
        the package at all any more.
        """
        tree = ast.parse((self.ROOT / self.DISPATCHER).read_text(encoding="utf-8"))
        imported = {node.module.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module}
        self.assertIn(CITATIONS_DIR.name, imported)


class CacheModeIsAnObjectTests(unittest.TestCase):
    """DRY_RUN_WRITES_NOTHING's third channel travels the way the other two
    do: the run picks an http_cache object and hands it over.

    A boolean threaded through the layers instead defaulted, at every one of
    them, to the WRITING implementation -- and seed_metadata.py advertises
    itself as callable with no command line, which is exactly the caller
    that would omit the keyword and write into the data tree under a dry
    run. `writer` and `measurements` have no such default; neither may
    `cache`.
    """

    SEEDERS = ("zbmath_abstracts", "mathnet_names")

    def test_no_module_takes_the_mode_as_a_flag(self):
        for path in SOURCES:
            self.assertNotIn(
                "read_only_cache", path.read_text(encoding="utf-8"),
                f"{path.name}: the mode is an http_cache object, not a flag")

    def test_the_seeding_pipeline_cannot_default_its_cache(self):
        tree = ast.parse((CITATIONS_DIR / "seed_metadata.py").read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        for name in self.SEEDERS:
            arguments = functions[name].args
            self.assertNotIn("cache", [a.arg for a in arguments.args],
                             f"{name}: cache must be keyword-only")
            defaults = dict(zip([a.arg for a in arguments.kwonlyargs],
                                arguments.kw_defaults))
            self.assertIn("cache", defaults, f"{name}: no cache argument at all")
            self.assertIsNone(defaults["cache"],
                              f"{name}: a defaulted cache is a writing cache")


class LoaderIsADispatcherTests(unittest.TestCase):
    """pg_load_citations.py parses flags, constructs and dispatches. The
    run-establishing pipeline (a SQL read, a client, journal writes and the
    reporting around them) belongs beside the rest of the package, where it
    is callable without a command line.
    """

    TREE = ast.parse(LOADER.read_text(encoding="utf-8"))

    def test_the_seed_metadata_pipeline_is_not_defined_here(self):
        defined = {node.name for node in self.TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("zbmath_abstracts", "mathnet_names"):
            self.assertNotIn(name, defined, f"{name}() belongs to citations/seed_metadata.py")

    def test_the_loader_issues_no_sql_of_its_own(self):
        text = LOADER.read_text(encoding="utf-8")
        self.assertNotIn("run_sql", text)
        self.assertNotIn("SELECT ", text)

    def test_it_still_dispatches_to_both(self):
        imported = {alias.name for node in ast.walk(self.TREE)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "citations.seed_metadata"
                    for alias in node.names}
        self.assertEqual(imported, {"mathnet_names", "zbmath_abstracts"})


class WriterModeIsAnObjectTests(unittest.TestCase):
    """Both writer seams answer "did this run write anything" the same way:
    the caller asks the writer it built.

    The graph seam used to be asked of argparse instead, and the two answers
    are already free to disagree -- main() builds a DryRunWriter when
    --calibrate is set with --dry-run unset, and only dispatch order keeps
    that combination away from do_crawl(). Reached, it would have printed a
    live run's acceptance counts and called project()/graph_check() over a
    graph nothing was written to, reporting a faithful projection of a run
    that wrote no rows.
    """

    TREE = ast.parse(LOADER.read_text(encoding="utf-8"))
    SPIKE_CLI = CITATIONS_DIR / "spike_cli.py"
    # The four mode bodies, wherever they live: the two that talk about the
    # graph stayed with the command line, the two that talk about
    # measurements moved beside the writer seam they report on.
    MODES = {LOADER: ("do_crawl", "do_merge_twins"),
             SPIKE_CLI: ("do_hub_report", "do_calibrate")}
    CONSTRUCTORS = ("DryRunWriter", "PostgresWriter",
                    "DryRunMeasurementsWriter", "MeasurementsWriter")

    def test_both_seams_carry_the_mode_on_the_writer(self):
        for writer, dry in ((PostgresWriter({}), False), (DryRunWriter(), True),
                            (MeasurementsWriter({}), False),
                            (DryRunMeasurementsWriter(), True)):
            with self.subTest(writer=type(writer).__name__):
                self.assertIs(writer.dry, dry)

    def test_the_graph_writers_conform_to_the_protocol(self):
        self.assertIn("dry", Writer.__annotations__)
        for writer in (PostgresWriter({}), DryRunWriter()):
            with self.subTest(writer=type(writer).__name__):
                self.assertIsInstance(writer, Writer)

    def test_no_mode_branches_on_the_flag_once_a_writer_exists(self):
        """The flag builds objects and is never consulted again: inside the
        mode functions every mention of args.dry_run is a construction (a
        cache's keyword), and nothing branches on it.
        """
        for path, names in self.MODES.items():
            functions = self._functions(path)
            for name in names:
                with self.subTest(mode=name):
                    self.assertEqual(self._branching_uses(functions[name]), 0,
                                     f"{name}(): режим спрашивается у писателя, не у флага")

    def test_neither_graph_mode_is_even_handed_the_command_line(self):
        functions = self._functions(LOADER)
        for name in ("do_crawl", "do_merge_twins"):
            with self.subTest(mode=name):
                self.assertNotIn("args", [a.arg for a in functions[name].args.args])

    def test_every_writer_is_built_in_one_place(self):
        """The mode -> object rule, spelled once.

        It used to be spelled three times and with two different
        predicates: do_merge_twins() re-derived its own graph writer from
        `args` on --dry-run alone, while main() built the crawl's on
        --dry-run OR --calibrate. A fifth mode that must not write to
        citation.* only had to be added to one of them to be live-writing
        when reached through the other -- the very failure the seam rules
        out, back at the construction site.
        """
        for path, expected in ((LOADER, self.CONSTRUCTORS), (self.SPIKE_CLI, ())):
            built = self._constructor_sites(path)
            with self.subTest(module=path.name):
                self.assertEqual(set(built), set(expected),
                                 f"{path.name}: писателей строит writers_for()")
                for name, enclosing in built.items():
                    self.assertEqual(enclosing, {"writers_for"},
                                     f"{name}() строится вне writers_for(): {enclosing}")

    def test_the_two_predicates_are_the_ones_writers_for_declares(self):
        """The graph writer and the measurements writer genuinely differ --
        --calibrate writes no citation.* row but DOES record a run -- and
        that difference is one expression each, in one function.
        """
        writer, measurements = pg_load_citations.writers_for(
            mock.Mock(dry_run=False, calibrate=True), {})
        self.assertIs(writer.dry, True)
        self.assertIs(measurements.dry, False)
        for dry_run in (True, False):
            writer, measurements = pg_load_citations.writers_for(
                mock.Mock(dry_run=dry_run, calibrate=False), {})
            with self.subTest(dry_run=dry_run):
                self.assertIs(writer.dry, dry_run)
                self.assertIs(measurements.dry, dry_run)

    @staticmethod
    def _functions(path: Path) -> dict:
        return {node.name: node
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef)}

    @classmethod
    def _constructor_sites(cls, path: Path) -> dict:
        """{writer class: the functions that call it} for one module."""
        found: dict[str, set] = {}
        for function in cls._functions(path).values():
            for node in ast.walk(function):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in cls.CONSTRUCTORS):
                    found.setdefault(node.func.id, set()).add(function.name)
        return found

    @staticmethod
    def _is_flag(node) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "dry_run"
                and isinstance(node.value, ast.Name) and node.value.id == "args")

    @classmethod
    def _branching_uses(cls, function: ast.FunctionDef) -> int:
        """How many args.dry_run reads are NOT part of building an object:
        the test of a `X() if flag else Y()` and a keyword argument are
        constructions, anything else decides something after the fact.
        """
        building = set()
        for node in ast.walk(function):
            if (isinstance(node, ast.IfExp) and cls._is_flag(node.test)
                    and isinstance(node.body, ast.Call)
                    and isinstance(node.orelse, ast.Call)):
                building.add(id(node.test))
            if isinstance(node, ast.Call):
                building.update(id(kw.value) for kw in node.keywords
                                if cls._is_flag(kw.value))
        return len([node for node in ast.walk(function)
                    if cls._is_flag(node) and id(node) not in building])


if __name__ == "__main__":
    unittest.main()
