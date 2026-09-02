"""JOURNAL_FACTS_ARE_COLUMNS, held over the whole repository at once.

What the pipeline reads out of citation.crawl_step lives in a column
(node_key, score, tau, relation, cited_by_count beside frontier_key /
candidate_key / n_found / n_kept); `reason` is prose for a human. Parsing
it back with substring/split_part/strpos/regexp_* is forbidden as a CLASS:
those take no index, catch a name inside a phrase and break when the
wording is edited -- three consumers lived that way at once.

The sibling invariant VOCABULARY_ONE_DECLARATION has a scan over every
module that names the schema, positive-controlled on a synthetic violating
module. This one had three hardcoded assertions instead, each naming one
known consumer by hand, so a FOURTH consumer -- a new report, a new
profile cut, a new check -- could reintroduce exactly the regression the
invariant exists to prevent and nothing would fail. This is the scan.

Excluded, by name and for a reason: pg_schema_citation_backfill.sql is the
one-time migration that parses the prose written before the columns
existed. It is the sanctioned exception, its own offline guard is next door
(test_backfill_parse.py), and it is skipped here rather than tolerated by a
pattern loose enough to let something else through.

tests/ is not scanned: the assertions that forbid these forms have to spell
them, and a test is not a consumer of the journal.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import _pathfix  # noqa: F401

import paths

ROOT = paths.kb_root()

# The one-time migration, and the only file allowed to hold a parse.
SANCTIONED = "pg_schema_citation_backfill.sql"

SKIPPED_DIRS = {"tests", "__pycache__", ".git"}

# substring(reason ...), split_part(reason, ...), strpos(reason, ...),
# regexp_match/regexp_replace/regexp_substr(reason ...) -- with or without a
# table alias in front, and with the newline a wrapped SQL string puts
# between the parenthesis and the column.
_PARSE = re.compile(
    r"\b(?:substring|split_part|strpos|position|regexp_\w+)\s*\(\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?reason\b",
    re.IGNORECASE)


def sql_faults(text: str, where: str) -> list[str]:
    """Parses of `reason` in SQL text, `--` comments discarded."""
    stripped = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    return [f"{where}: {found.group(0).strip()} — reason разбирается вместо колонки"
            for found in _PARSE.finditer(stripped)]


def module_faults(source: str, where: str) -> list[str]:
    """The same, over the SQL a Python module carries in its strings.

    Docstrings are prose about the ban and are not scanned -- the invariant
    is quoted in several of them, which is the point of writing it down.
    """
    tree = ast.parse(source)
    prose = {id(node.value) for node in ast.walk(tree)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    faults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in prose:
            continue
        faults += sql_faults(node.value, f"{where}:{node.lineno}")
    return faults


def _scanned_files() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in (".py", ".sql") or not path.is_file():
            continue
        if SKIPPED_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.name == SANCTIONED:
            continue
        found.append(path)
    return found


class ReasonIsNotParsedAnywhereTests(unittest.TestCase):
    def test_no_module_and_no_schema_file_parses_the_journal_prose(self):
        faults = []
        for path in _scanned_files():
            rel = str(path.relative_to(ROOT))
            text = path.read_text(encoding="utf-8")
            faults += (module_faults(text, rel) if path.suffix == ".py"
                       else sql_faults(text, rel))
        self.assertEqual(faults, [])

    def test_the_scan_reaches_the_modules_that_consume_the_journal(self):
        """The control for the rule above: it cannot pass by scanning
        nothing, and it must cover the consumers that read crawl_step.
        """
        scanned = {str(path.relative_to(ROOT)) for path in _scanned_files()}
        self.assertGreater(len(scanned), 40)
        for name in ("citation_checks.py", "pg_graph.py", "citations/journal.py",
                     "deploy/citation_profile.py", "pg_schema_citation.sql"):
            self.assertIn(name, scanned)

    def test_the_sanctioned_migration_is_skipped_by_name_not_by_pattern(self):
        scanned = {path.name for path in _scanned_files()}
        self.assertNotIn(SANCTIONED, scanned)
        # And it really does hold a parse, so the exemption is not stale.
        self.assertTrue(sql_faults(
            (ROOT / SANCTIONED).read_text(encoding="utf-8"), SANCTIONED))


class TheScanCatchesEachFormTests(unittest.TestCase):
    """Positive controls, on synthetic sources: the scan must be able to
    fail, in SQL and in a Python module alike.
    """

    def test_a_sql_file_that_parses_reason_is_caught(self):
        for statement in (
            "SELECT substring(reason from 'score=([0-9.]+)') FROM citation.crawl_step;",
            "SELECT split_part(reason, 'node=', 2) FROM citation.crawl_step;",
            "SELECT strpos(reason, 'tau=') FROM citation.crawl_step;",
            "SELECT regexp_match(s.reason, 'seed=(.*)') FROM citation.crawl_step s;",
            "SELECT substring(\n  s.reason from 'node=([^ ]+)') FROM citation.crawl_step s;",
        ):
            with self.subTest(statement=statement):
                self.assertEqual(len(sql_faults(statement, "synthetic.sql")), 1)

    def test_a_module_carrying_such_a_statement_is_caught(self):
        source = ('SQL = "SELECT split_part(reason, \'node=\', 2) '
                  'FROM citation.crawl_step;"\n')
        self.assertEqual(len(module_faults(source, "synthetic.py")), 1)

    def test_a_docstring_may_still_name_the_forbidden_form(self):
        source = ('"""substring(reason from ...) is what this module must '
                  'never do."""\n')
        self.assertEqual(module_faults(source, "synthetic.py"), [])

    def test_a_comment_in_sql_may_name_it_too(self):
        self.assertEqual(
            sql_faults("-- never: substring(reason from 'score=')\n"
                       "SELECT score FROM citation.crawl_step;\n", "synthetic.sql"),
            [])

    def test_reading_the_column_itself_is_not_a_fault(self):
        self.assertEqual(
            sql_faults("SELECT reason, score, node_key FROM citation.crawl_step "
                       "WHERE action = 'keep';", "synthetic.sql"),
            [])
        self.assertEqual(
            sql_faults("SELECT split_part(candidate_key, ':', 1) "
                       "FROM citation.crawl_step;", "synthetic.sql"),
            [])


if __name__ == "__main__":
    unittest.main()
