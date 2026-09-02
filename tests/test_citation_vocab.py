"""The two closed DB vocabularies, held to one declaration.

citation_vocab.py spells citation.work.kind and citation.crawl_step.action
once on the Python side; pg_schema_citation.sql spells them once on the SQL
side. Two spellings in two languages hold together only if something
compares them, so:

- an AST scan over every module that talks to the citation schema refuses a
  bare literal from either vocabulary (docstrings excepted -- prose about a
  value is not a use of it);
- a live test compares the constants against pg_get_constraintdef() in BOTH
  directions, so an extra value, a missing one and a renamed one all fail.

The comparison is over the VOCABULARY, not the constraint text: the server
renders `x = ANY (ARRAY[...])` however its version likes, and only the
literals inside are ours.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import citation_vocab
from citation_vocab import CrawlAction, WorkKind
from citations import journal
from paths import default_corpus_dir, kb_root
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, scalar

VALUES = tuple(WorkKind.ALL) + tuple(CrawlAction.ALL)
VOCAB_FILE = Path(citation_vocab.__file__).resolve()

_CONSTRAINT_SQL = """
SELECT coalesce(pg_get_constraintdef(c.oid), '')
FROM pg_constraint c
WHERE c.conrelid = '{table}'::regclass AND c.conname = '{name}';
"""


def _schema_modules() -> list[Path]:
    """Every module that talks to the citation schema: the crawl package,
    and the root modules that name the schema themselves. Root modules that
    do not (external_registry.py, say, whose 'excluded' belongs to
    corpus.documents.public_distribution) are a different vocabulary that
    happens to share a word, and are not this test's business.
    """
    root = kb_root()
    modules = sorted((root / "citations").glob("*.py"))
    for path in sorted(root.glob("*.py")):
        if path.resolve() != VOCAB_FILE and "citation." in path.read_text(encoding="utf-8"):
            modules.append(path)
    return modules


def _docstrings(tree: ast.AST) -> set[int]:
    """id() of every docstring node -- prose naming a value is not a use of
    it, and a docstring cannot drift into a query."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _value_positions(tree: ast.AST) -> set[int]:
    """id() of every string constant sitting where a VALUE goes: an
    argument, a keyword, an operand of a comparison, a dict value, the
    right-hand side of an assignment, a returned expression.

    The distinction is what separates `kind="our-document"` from the word
    "seed" as a printed column header (pg_graph.py's hybrid table), which
    is not this vocabulary at all and must not be renamed to please a test.
    A quoted occurrence INSIDE a string -- `'our-document'`, i.e. an SQL
    literal -- is caught regardless of position.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            out |= {id(a) for a in node.args}
            out |= {id(k.value) for k in node.keywords}
        elif isinstance(node, ast.keyword):
            out.add(id(node.value))
        elif isinstance(node, ast.Compare):
            out.add(id(node.left))
            out |= {id(c) for c in node.comparators}
        elif isinstance(node, ast.Dict):
            out |= {id(v) for v in node.values if v is not None}
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
            if getattr(node, "value", None) is not None:
                out.add(id(node.value))
    return out


class NoModuleSpellsTheVocabularyItselfTests(unittest.TestCase):
    def test_every_citation_module_reaches_for_the_constants(self):
        for path in _schema_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            skip = _docstrings(tree)
            positions = _value_positions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in skip:
                    continue
                for value in VALUES:
                    spelled = f"'{value}'" in node.value or (
                        node.value == value and id(node) in positions)
                    self.assertFalse(
                        spelled,
                        f"{path.name}:{node.lineno}: {value!r} spelled here -- import it "
                        f"from citation_vocab (WorkKind/CrawlAction)",
                    )


class JournalActionsAreTheVocabularyTests(unittest.TestCase):
    """The journal is where a new kind of decision arrives, and its bulk
    COPY is where an unknown one would be discovered -- all-or-nothing,
    after the level's work rows and edges are already written. So the check
    happens as the step is built, on the Python side of that COPY.
    """

    def test_every_builder_produces_a_known_action(self):
        steps = [
            journal.seed("c", "doc", "W1"),
            journal.seed_missing("c", "doc"),
            journal.seed_error("c", "doc", "W1"),
            journal.zbmath_error("c", "doc", "zb1", "429"),
            journal.keep("c", 1, "W1", "W1", 0.9, 0.5, "cites"),
            journal.drop("c", 1, "W1", 0.1, 0.5, "cites"),
            journal.fetch("c", 1, "W1", 3, 2),
            journal.hub_skip("c", 1, "W1", 5000, 1000),
            journal.twin("c", "W1", "doc", "W2"),
        ]
        self.assertTrue(steps)
        for step in steps:
            self.assertIn(step["action"], CrawlAction.ALL, step)

    def test_an_action_outside_the_vocabulary_is_refused_before_the_copy(self):
        with self.assertRaises(ValueError) as ctx:
            journal._step("c", 1, "hub-skipped")
        self.assertIn("hub-skipped", str(ctx.exception))


class VocabularyMatchesTheSchemaLiveTests(unittest.TestCase):
    """Both directions, against the constraint the database actually has."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def _check_vocabulary(self, table: str, name: str) -> set[str]:
        definition = scalar(self.env, _CONSTRAINT_SQL.format(table=table, name=name))
        self.assertTrue(definition, f"{name} is not on {table}: apply pg_graph.py init")
        return set(re.findall(r"'([^']*)'", definition))

    def test_the_work_kinds_are_the_ones_the_check_allows(self):
        self.assertEqual(self._check_vocabulary("citation.work", "work_kind_check"),
                         set(WorkKind.ALL))

    def test_the_crawl_actions_are_the_ones_the_check_allows(self):
        self.assertEqual(
            self._check_vocabulary("citation.crawl_step", "crawl_step_action_check"),
            set(CrawlAction.ALL))


if __name__ == "__main__":
    unittest.main()
