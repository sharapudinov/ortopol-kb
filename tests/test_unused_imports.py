"""No module imports a name it never uses.

A dead import is not a style question here. It is the residue of a
refactoring that moved a subject somewhere else and left the module still
claiming to depend on it -- tests/test_citation_dump.py imported
citation_profile and legal_profile.SHIPPED_SQL long after the tests about
them had moved out, so a reader looking for what that file talks to was
told two things that were no longer true. There is no linter in this
repository's dependencies (no third-party packages at all, deliberately),
so the scan is the standard library's own parser.

A name is used if it appears as a bare name or at the root of an attribute
chain anywhere else in the module. Deliberate re-exports and import-for-
side-effect (tests/_pathfix.py) say so with `# noqa` on the import line,
which is what every one of them already carries.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import _pathfix  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRECTORIES = {"__pycache__", ".git"}


def python_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py")
                  if not SKIP_DIRECTORIES & set(path.parts))


def imported_names(tree: ast.AST):
    """(bound name, line) for every import, minus `from __future__`, whose
    names are compiler directives rather than objects anyone uses.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.asname or alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name != "*":
                    yield alias.asname or alias.name, node.lineno


def used_names(tree: ast.AST) -> set[str]:
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    return used


def unused_in(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    used = used_names(tree)
    return [f"{path.relative_to(ROOT)}:{line}: {name}"
            for name, line in imported_names(tree)
            if name not in used and "noqa" not in lines[line - 1]]


class NoUnusedImportsTests(unittest.TestCase):
    def test_no_module_imports_a_name_it_never_uses(self):
        dead = [entry for path in python_files() for entry in unused_in(path)]
        self.assertEqual(dead, [])

    def test_the_scan_sees_one_that_is_planted(self):
        """The positive control: a scan that silently matched nothing would
        report a clean tree forever."""
        planted = ast.parse("import json\nimport re\nprint(re.escape('x'))\n")
        names = dict(imported_names(planted))
        self.assertEqual(set(names) - used_names(planted), {"json"})

    def test_a_noqa_import_is_left_alone(self):
        """_pathfix and every re-export are imported for their effect, not
        for a name -- and every one of them already says so."""
        self.assertEqual(unused_in(Path(__file__).resolve()), [])


if __name__ == "__main__":
    unittest.main()
