"""DRY_RUN_WRITES_NOTHING's third channel: the caches in the data tree.

Split out of test_citations_layering.py by responsibility (and by
kb/CLAUDE.md FILE_SIZE): that module holds where the crawl's pieces are
allowed to live, this one holds how a mode reaches the tree. Both are
AST scans over the same sources, and both exist because the rule they
check is one a new call site breaks without failing anything.

Two halves, and the second is the one that was missing. Construction:
every cache object is built in pg_load_citations.main() and handed to
its reader as an object, never as a flag. Declaration: the rule "which
mode gets a read-only tree" is spelled ONCE -- tree_read_only(args) --
the way writers_for() spells the writers'. Written out per site, it was
five copies of a predicate whose three earlier copies had already
diverged.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import pg_load_citations
from citations import inputs

CITATIONS_DIR = Path(inputs.__file__).resolve().parent
LOADER = Path(pg_load_citations.__file__).resolve()
SOURCES = sorted(CITATIONS_DIR.glob("*.py")) + [LOADER]


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

    # OpenAlex responses, zbMATH reviews, Math-Net pages, candidate vectors.
    CHANNELS = 4

    def test_only_the_command_line_builds_a_cache(self):
        """Every cache object is built in pg_load_citations.main(), the way
        every writer is built in writers_for(). A second construction site
        is a second answer to "which mode gets which object":
        spike_cli.do_hub_report() built its own from args.cache_dir, in the
        module whose docstring says neither writer nor cache is built there.
        """
        built = []
        for path in SOURCES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            allowed = {id(node) for function in ast.walk(tree)
                       if isinstance(function, ast.FunctionDef)
                       and function.name == "main" and path == LOADER
                       for node in ast.walk(function)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", None) != "cache_for":
                    continue
                self.assertIn(
                    id(node), allowed,
                    f"{path.name}:{node.lineno}: кэш строит pg_load_citations.main")
                built.append(node.lineno)
        self.assertGreaterEqual(len(built), self.CHANNELS,
                                "main() перестал строить кэши — проверять стало нечего")

    def test_no_mode_takes_the_namespace_instead_of_its_objects(self):
        """The other half: a mode handed `args` reaches the flag whatever
        the seam says, and cannot be driven without argparse at all."""
        tree = ast.parse((CITATIONS_DIR / "spike_cli.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("do_"):
                self.assertNotIn("args", [a.arg for a in node.args.args],
                                 f"{node.name}(): режим получает объекты, не Namespace")

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


class NoSeamHasADefaultTests(unittest.TestCase):
    """All THREE channels answer the same way, everywhere under citations/.

    The cache scan above was written on the stated grounds that "`writer`
    and `measurements` have no such default" -- and one function had one:
    zbmath_abstracts(..., writer=None, crawl_id=None), whose error rows a
    programmatic re-seed silently discarded by omitting a keyword. A
    guarantee a test names is a guarantee a test has to check.
    """

    # `cache` is not here: a session given no cache genuinely HAS none
    # (http_session.HttpSession documents that state), so None is a value
    # rather than a silent writing default. Where a cache MUST arrive --
    # the two seeders -- the test above asserts it by name. A writer that
    # is None is not a state; it is a discarded write.
    SEAMS = ("writer", "measurements", "crawl_id")

    def _defaulted(self, path: Path) -> list[str]:
        found = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = node.args.posonlyargs + node.args.args
            defaulted = [a.arg for a in positional[len(positional) - len(node.args.defaults):]]
            defaulted += [a.arg for a, default
                          in zip(node.args.kwonlyargs, node.args.kw_defaults)
                          if default is not None]
            found += [f"{node.name}({name}=...)" for name in defaulted
                      if name in self.SEAMS]
        return found

    def test_no_function_defaults_a_seam(self):
        for path in SOURCES:
            self.assertEqual(
                self._defaulted(path), [],
                f"{path.name}: у канала записи нет умолчания — режим приезжает объектом")

    def test_the_scan_catches_one(self):
        """Positive control: the signature this scan was written for."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8",
                                         delete=False) as handle:
            handle.write("def f(env, *, cache, writer=None, crawl_id=None):\n    pass\n")
            probe = Path(handle.name)
        self.addCleanup(probe.unlink)
        self.assertEqual(self._defaulted(probe),
                         ["f(writer=...)", "f(crawl_id=...)"])

    # Four channels, five construction sites: the response cache is built
    # once for the hub measurement's own branch and once for the crawl.
    SITES = 5

    def test_the_read_only_rule_is_declared_once(self):
        """`read_only=args.dry_run`, written out at every site, is the
        shape writers_for() was introduced to remove one abstraction over
        -- and the shape whose three earlier copies had already diverged
        (`--dry-run` against `--dry-run OR --calibrate`). The rule is a
        function; the sites call it.
        """
        sites = [node for node in ast.walk(ast.parse(LOADER.read_text(encoding="utf-8")))
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "cache_for"]
        self.assertEqual(len(sites), self.SITES,
                         "изменилось число кэшей — проверьте, все ли идут через правило")
        for node in sites:
            keywords = {kw.arg: kw.value for kw in node.keywords}
            with self.subTest(line=node.lineno):
                self.assertIn("read_only", keywords,
                              f"{LOADER.name}:{node.lineno}: у канала нет умолчания")
                value = keywords["read_only"]
                self.assertTrue(
                    isinstance(value, ast.Call)
                    and getattr(value.func, "id", None) == "tree_read_only",
                    f"{LOADER.name}:{node.lineno}: правило объявляет tree_read_only()")

    def test_the_rule_is_the_one_the_declaration_makes(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                self.assertIs(
                    pg_load_citations.tree_read_only(
                        mock.Mock(dry_run=dry_run, calibrate=False)),
                    dry_run)

    def test_calibration_still_writes_its_own_cache(self):
        """The tree rule is NOT the graph writer's: --calibrate writes no
        citation.* row and is exactly the run whose vectors and responses
        must be kept, so the next run does not pay for them again.
        """
        self.assertIs(pg_load_citations.tree_read_only(
            mock.Mock(dry_run=False, calibrate=True)), False)
