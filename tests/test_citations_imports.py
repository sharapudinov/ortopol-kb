"""A name is imported from the module that owns it.

The rule the repository already states for the graph side
(pg_graph_cypher.py's docstring, and test_pg_graph.py's
test_no_query_module_re_exports_another) applies here for the same reason,
and here it had already been broken in the direction that matters most.

citations/openalex_records.py was split off the HTTP client to hold "what an
OpenAlex record MEANS, apart from how it is asked for" -- and then
openalex_client.py re-exported six of its names, so every production
consumer imported them through the transport. Node identity, edge
derivation and candidate assembly are pure; none of them speaks HTTP. Worst
placed was hub_cache.py, which backs the deliberately network-free
--hub-report path and reached record parsing only through the network
client.

A re-export is invisible where it is used: `from .openalex_client import
short_id` reads as a dependency on the client. Only the client's own import
list says otherwise, which is exactly the thing a reader does not open. So
the scan is over the import statements, not over anyone's intent: a name
imported from a sibling must be DEFINED in that sibling.

Not a style rule -- a dependency-direction one, and the same class the
package's own docstring is organised around. `# noqa` does not exempt an
import here: the marker says a name is deliberately unused (NO_DEAD_IMPORTS),
which is precisely how a re-export hop is spelled.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import _pathfix  # noqa: F401

PACKAGE = Path(__file__).resolve().parent.parent / "citations"


def modules() -> dict[str, Path]:
    return {path.stem: path for path in sorted(PACKAGE.glob("*.py"))
            if path.name != "__init__.py"}


def defined_names(tree: ast.AST) -> set[str]:
    """Every name a module binds at top level -- what it can be said to own.

    Imports are deliberately NOT among them: a name a module merely imported
    belongs to whoever defined it.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets
                      if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def sibling_imports(tree: ast.AST, known: set[str]):
    """(sibling module, imported name, line) for every `from .X import name`
    naming a module of this package."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module in known:
            for alias in node.names:
                yield node.module, alias.name, node.lineno


def hops(sources: dict[str, str]) -> list[str]:
    trees = {name: ast.parse(source) for name, source in sources.items()}
    owned = {name: defined_names(tree) for name, tree in trees.items()}
    found = []
    for name, tree in trees.items():
        for sibling, imported, line in sibling_imports(tree, set(trees)):
            if imported not in owned[sibling]:
                found.append(f"{name}.py:{line}: {imported} is not defined in "
                             f"{sibling}.py")
    return found


class ImportsNameTheOwnerTests(unittest.TestCase):

    def test_no_module_imports_a_name_its_sibling_only_re_exports(self):
        sources = {name: path.read_text(encoding="utf-8")
                   for name, path in modules().items()}
        self.assertEqual(hops(sources), [])

    def test_the_scan_catches_a_re_export_hop(self):
        """Positive control: the exact shape that was here -- a pure helper
        reached through the transport that re-exported it."""
        sources = {
            "records": "def short_id(value):\n    return value\n",
            "client": "from .records import short_id  # noqa: F401\n",
            "consumer": "from .client import short_id\n",
        }
        self.assertEqual(hops(sources),
                         ["consumer.py:1: short_id is not defined in client.py"])

    def test_the_scan_accepts_the_owner_and_the_module_form(self):
        sources = {
            "records": "def short_id(value):\n    return value\n",
            "consumer": "from .records import short_id\n",
            "other": "from . import records\n",
        }
        self.assertEqual(hops(sources), [])

    def test_the_pure_record_layer_is_reached_directly(self):
        """The finding itself, held as a fact rather than as an absence:
        every consumer of a record helper names openalex_records, and the
        transport re-exports nothing."""
        sources = {name: path.read_text(encoding="utf-8")
                   for name, path in modules().items()}
        client = ast.parse(sources["openalex_client"])
        for module, imported, _line in sibling_imports(client, set(sources)):
            with self.subTest(name=imported):
                self.assertIn(imported, {node.id for node in ast.walk(client)
                                         if isinstance(node, ast.Name)}
                              | {node.attr for node in ast.walk(client)
                                 if isinstance(node, ast.Attribute)},
                              f"openalex_client.py imports {imported} from "
                              f"{module} without using it")
        record_names = defined_names(ast.parse(sources["openalex_records"]))
        reached = {name for name, source in sources.items()
                   for module, imported, _line in sibling_imports(
                       ast.parse(source), set(sources))
                   if module == "openalex_records" and imported in record_names}
        self.assertIn("registry", reached)
        self.assertIn("candidates", reached)


if __name__ == "__main__":
    unittest.main()
