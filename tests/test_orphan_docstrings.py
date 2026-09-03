"""No test module carries a string literal standing on its own where a
class was meant to open.

A `class SeedingTests(unittest.TestCase):` line went missing and left its
docstring behind as a bare expression statement in the middle of the
PREVIOUS class's body. Python evaluates it and discards it, so nothing
failed: four seeding tests simply became methods of a scoring-memory test
case, inherited fixtures they never use, and disappeared from every
`grep -n "class SeedingTests"` and every `-k SeedingTests` run. The suite
still reported them, under a name that says nothing about what they test --
which is the one way a lost test case does NOT show up as a smaller count.

The rule is narrow on purpose: a docstring is the FIRST statement of a
module, class or function, and every other bare string literal statement is
either that mistake or a comment written with the wrong punctuation. Both
are worth a line in the report; neither is worth guessing about.

tests/ only. The repository's modules are held to the same shape by
tests/test_unused_imports.py and the file-size scan; this failure mode is
about test CASES going missing, and a scan is what tells the difference
between a suite that grew and a suite that moved.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401

TESTS_DIR = Path(__file__).resolve().parent

# The nodes that HAVE a docstring, i.e. whose first statement may be a bare
# string. Everything else with a body (if, for, with, try) never may.
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _orphans(path: Path) -> list[str]:
    """Every bare string-literal statement in `path` that is not the
    docstring of the thing it opens."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        start = 1 if isinstance(node, _DOCUMENTED) and body else 0
        for statement in body[start:]:
            if (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)):
                found.append(f"{path.name}:{statement.lineno}")
    return found


def orphan_docstrings(root: Path) -> list[str]:
    return sorted(entry for path in sorted(root.glob("*.py"))
                  for entry in _orphans(path))


class OrphanDocstringTests(unittest.TestCase):

    def test_no_test_module_has_a_string_statement_of_its_own(self):
        self.assertEqual(
            orphan_docstrings(TESTS_DIR), [],
            "строка-выражение не на первом месте: пропала строка "
            "`class ...:` над докстрингом, и его тесты уехали в соседний "
            "класс молча")

    def test_the_scan_catches_a_class_header_that_went_missing(self):
        """Positive control, in the exact shape the defect had: the second
        class's docstring left standing inside the first class's body.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_lost.py").write_text(
                'import unittest\n\n\n'
                'class FirstTests(unittest.TestCase):\n'
                '    """Its own docstring, which is fine."""\n\n'
                '    def test_one(self):\n'
                '        pass\n\n'
                '    """The second class\'s docstring, and no second class."""\n\n'
                '    def test_two(self):\n'
                '        pass\n',
                encoding="utf-8")
            # A module, class and function docstring apiece, and nothing else.
            (root / "test_fine.py").write_text(
                '"""Module docstring."""\n\n\n'
                'class Tests:\n'
                '    """Class docstring."""\n\n'
                '    def test_one(self):\n'
                '        """Method docstring."""\n'
                '        pass\n',
                encoding="utf-8")
            found = orphan_docstrings(root)
        self.assertEqual(found, ["test_lost.py:10"])


if __name__ == "__main__":
    unittest.main()
