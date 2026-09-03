"""Every source file in the repository stays inside its FILE_SIZE cap.

kb/CLAUDE.md states the cap -- code <= 300 lines, tests <= 500, split by
responsibility in the commit that breaches it -- and until now it was held
by whoever happened to run `wc -l`. That is not a gate: the file that went
over did so in the commit that ADDED it, at 303 lines, and it passed every
test in this suite. A cap nothing measures is a preference.

Two regimes, one scan. `tests/` is the looser one, and the directory is the
whole rule -- a fixture module (tests/_citation_fixtures.py) is test code
whatever its name says, and a module that MOVES from tests/ into the
repository root moves into the stricter regime with it.

.sql files count as code and are scanned here too: pg_schema_citation.sql
was split for exactly this cap, and the check that held its four pieces
(test_pg_graph.SchemaFileSplitTests) knew only about the files SCHEMA_PATHS
names. A new schema file nothing has added to that tuple yet is precisely
the one worth catching.

Blank lines and comments count. The cap is about how much a reader has to
hold at once, and a 400-line file of comments is not easier to read than a
400-line file of code -- nor is `wc -l`, the way the rule is checked by
hand, in the habit of distinguishing them.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"
SKIP_DIRECTORIES = {"__pycache__", ".git"}
SUFFIXES = (".py", ".sql")

CODE_LIMIT = 300
TEST_LIMIT = 500


def source_files(root: Path) -> list[Path]:
    return sorted(path for suffix in SUFFIXES for path in root.rglob(f"*{suffix}")
                  if not SKIP_DIRECTORIES & set(path.parts))


def limit_for(path: Path, root: Path = ROOT) -> int:
    """Which cap this file answers to -- decided by where it lives, not by
    what it is named."""
    return TEST_LIMIT if (root / "tests") in path.parents else CODE_LIMIT


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def oversized(root: Path) -> list[str]:
    found = []
    for path in source_files(root):
        lines, limit = line_count(path), limit_for(path, root)
        if lines > limit:
            found.append(f"{path.relative_to(root)}: {lines} lines > {limit}")
    return found


class FileSizeTests(unittest.TestCase):

    def test_every_source_file_is_within_its_cap(self):
        self.assertEqual(
            oversized(ROOT), [],
            "FILE_SIZE (kb/CLAUDE.md): разделить по ответственности "
            "в том же коммите")

    def test_the_scan_catches_a_file_over_each_cap(self):
        """Positive control, both regimes: a scan that silently found
        nothing to look at would pass the test above forever."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "big.py").write_text("x = 1\n" * (CODE_LIMIT + 1), encoding="utf-8")
            (root / "schema.sql").write_text("SELECT 1;\n" * (CODE_LIMIT + 1),
                                             encoding="utf-8")
            (root / "tests" / "test_big.py").write_text(
                "x = 1\n" * (TEST_LIMIT + 1), encoding="utf-8")
            # Right at each cap, and therefore not a finding.
            (root / "edge.py").write_text("x = 1\n" * CODE_LIMIT, encoding="utf-8")
            (root / "tests" / "test_edge.py").write_text(
                "x = 1\n" * TEST_LIMIT, encoding="utf-8")
            found = oversized(root)
        self.assertEqual(
            sorted(entry.split(":")[0] for entry in found),
            ["big.py", "schema.sql", "tests/test_big.py"])

    def test_the_scan_reaches_every_corner_of_the_repository(self):
        """The directories a repository-wide scan is worth having over --
        each one holding a file this suite already talks about."""
        scanned = {path.relative_to(ROOT).as_posix() for path in source_files(ROOT)}
        for expected in ("paths.py", "pg_graph_common.py", "pg_schema_citation.sql",
                         "citations/registry.py", "deploy/artifact_bundle.py",
                         "tests/test_file_size.py"):
            self.assertIn(expected, scanned)

    def test_a_test_module_answers_to_the_looser_cap(self):
        self.assertEqual(limit_for(TESTS_DIR / "test_file_size.py"), TEST_LIMIT)
        self.assertEqual(limit_for(TESTS_DIR / "_citation_fixtures.py"), TEST_LIMIT)
        self.assertEqual(limit_for(ROOT / "paths.py"), CODE_LIMIT)
        self.assertEqual(limit_for(ROOT / "citations" / "registry.py"), CODE_LIMIT)


if __name__ == "__main__":
    unittest.main()
