"""The closed DB vocabularies, held to one declaration.

citation_vocab.py spells citation.work.kind, citation.crawl_step.action/
relation and citation.public_policy.mode once on the Python side;
pg_schema_citation_constraints.sql spells each once on the SQL side, as the
wanted array of an idempotent named-constraint migration (an inline CHECK on
a table that already exists can never be widened again). Two spellings in two languages hold
together only if something compares them, so:

- an AST scan over every module that talks to the citation schema refuses a
  bare literal from either vocabulary (docstrings excepted -- prose about a
  value is not a use of it);
- a static test reads the schema files themselves and compares the literals
  in each CHECK clause against the Python constants in BOTH directions, so a
  value added on one side only fails on ANY checkout, server or not;
- a live test asks the same of pg_get_constraintdef(), which is the constraint
  the database actually carries -- the schema file says what the next
  `pg_graph.py init` will apply, and the two are not the same claim.

The static half is what a machine with no Postgres runs. Without it, adding a
CrawlAction and forgetting the SQL passed the whole suite there, and the
divergence then surfaced where it costs most: the journal travels as one bulk
COPY, so a value the CHECK rejects loses the WHOLE level's audit record after
its work rows and edges are already written.

The comparison is over the VOCABULARY, not the constraint text: the server
renders `x = ANY (ARRAY[...])` however its version likes, and only the
literals inside are ours.

The AST half covers the first two only. The mode vocabulary is read by the
packager (deploy/), which _schema_modules() does not walk -- there the word
"excluded" belongs to corpus.documents.public_distribution and "none" is
not a distinctive literal at all, so a scan over those modules would fail
on vocabularies that merely share a word. What holds the third one is the
live comparison below, plus the single Python declaration itself:
manifest_contract.CitationMode extends PublicPolicyMode rather than
restating its values.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import NamedTuple

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401
import citation_vocab
from citation_vocab import CrawlAction, PublicPolicyMode, Relation, WorkKind
from citations import journal, threshold_store
from manifest_contract import CitationMode
from paths import default_corpus_dir, kb_root
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, scalar

VALUES = tuple(WorkKind.ALL) + tuple(CrawlAction.ALL) + tuple(Relation.ALL)
VOCAB_FILE = Path(citation_vocab.__file__).resolve()

SCHEMA_FILE = kb_root() / "pg_schema_citation.sql"
CONSTRAINTS_FILE = kb_root() / "pg_schema_citation_constraints.sql"

# Each vocabulary as the SQL spells it. All five go through the one
# function that widens a NAMED constraint without a validation scan: an
# inline CHECK cannot be widened on a table that already exists, and CREATE
# TABLE IF NOT EXISTS makes every deployed instance exactly that case.
# Anchored on the constraint name, so a clause elsewhere cannot stand in for
# a missing one, and a clause that stops matching is a failure here rather
# than a silently empty comparison
# (test_every_vocabulary_is_found_and_is_not_empty).
def _wanted_array(constraint: str) -> re.Pattern:
    return re.compile(rf"'{constraint}',\s*ARRAY\[([^\]]*)\]", re.S)


class Declaration(NamedTuple):
    """One closed vocabulary: the column it constrains, the SQL that
    declares it, the constraint name it is declared under, and the Python
    constants it has to equal.

    `sql` is TEXT rather than a path because the fifth declaration is not a
    file at all: citations/threshold_store.py builds the measurements
    table's DDL in Python and the calibration applies it. Same migrator,
    same comparison, so the same row here -- it was the one vocabulary left
    inline, i.e. the one that could never be widened on an instance that
    already ran a calibration.
    """

    column: str
    sql: str
    constraint: str
    python: tuple


CONSTRAINTS_SQL = CONSTRAINTS_FILE.read_text(encoding="utf-8")
DECLARATIONS = {
    "citation.work.kind": Declaration(
        "kind", CONSTRAINTS_SQL, "work_kind_check", WorkKind.ALL),
    "citation.crawl_step.action": Declaration(
        "action", CONSTRAINTS_SQL, "crawl_step_action_check", CrawlAction.ALL),
    "citation.crawl_step.relation": Declaration(
        "relation", CONSTRAINTS_SQL, "crawl_step_relation_check", Relation.ALL),
    "citation.public_policy.mode": Declaration(
        "mode", CONSTRAINTS_SQL, "public_policy_mode_check", PublicPolicyMode.ALL),
    "measurements.citation_frontier_threshold.relation": Declaration(
        "relation", threshold_store.THRESHOLD_DDL,
        threshold_store.RELATION_CONSTRAINT, Relation.ALL),
}

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

    A tuple, list or set is opened up and its elements counted too, because
    that is the most natural way to re-declare a vocabulary and the one form
    this scan used to be blind to: `if action in ("keep", "drop")` and
    `ACTIONS = ["seed", "hub-skip"]` are second declarations exactly as much
    as a bare keyword argument is. Opened up only where the COLLECTION is
    itself a declaring position -- compared against, assigned, returned,
    passed by keyword -- and not when it is a positional argument, which is
    where a row of display text goes (the header list above is one). A dict
    KEY is not a value position either: it names a table or a counter.
    """
    out: set[int] = set()
    collections: list[ast.AST] = []

    def value(node, declaring: bool = True) -> None:
        if node is None:
            return
        out.add(id(node))
        if declaring:
            collections.append(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for argument in node.args:
                value(argument, declaring=False)
            for word in node.keywords:
                value(word.value)
        elif isinstance(node, ast.keyword):
            value(node.value)
        elif isinstance(node, ast.Compare):
            value(node.left)
            for comparator in node.comparators:
                value(comparator)
        elif isinstance(node, ast.Dict):
            for entry in node.values:
                value(entry)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
            value(getattr(node, "value", None))

    while collections:
        node = collections.pop()
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                value(element)
    return out


def bare_vocabulary(source: str) -> list[tuple[int, str]]:
    """(line, value) for every vocabulary word `source` spells itself.

    The scan, as one function, so the modules and the positive control below
    are guarded by the SAME code -- a control exercising a second copy would
    prove nothing about the one that runs.
    """
    tree = ast.parse(source)
    skip = _docstrings(tree)
    positions = _value_positions(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        for value in VALUES:
            if f"'{value}'" in node.value or (node.value == value and id(node) in positions):
                found.append((node.lineno, value))
    return found


class NoModuleSpellsTheVocabularyItselfTests(unittest.TestCase):
    def test_every_citation_module_reaches_for_the_constants(self):
        for path in _schema_modules():
            for line, value in bare_vocabulary(path.read_text(encoding="utf-8")):
                self.fail(f"{path.name}:{line}: {value!r} spelled here -- import it "
                          f"from citation_vocab (WorkKind/CrawlAction/Relation)")

    def test_the_scan_walks_the_modules_that_name_the_schema(self):
        """A guard that scans nothing passes for the wrong reason. The
        crawl, the completeness checks and the graph query modules all name
        citation.*; citation_vocab.py itself is the one declaration and is
        excluded by construction.
        """
        names = {path.name for path in _schema_modules()}
        for expected in ("crawl.py", "gathering.py", "journal.py", "store_sql.py",
                         "hub_report.py", "threshold_store.py", "citation_checks.py",
                         "pg_graph_candidates.py", "pg_graph_cypher.py"):
            self.assertIn(expected, names)
        self.assertNotIn("citation_vocab.py", names)


class TheScanCatchesWhatItIsForTests(unittest.TestCase):
    """Positive control on synthetic sources.

    The scan is the only thing standing between a second declaration and
    the CHECK, and until it was pointed at a planted one, "it passes" and
    "it looks at nothing" were the same observation. Membership in a tuple
    or list is the form it used to miss: the natural way to write the
    re-declaration is exactly `in ("keep", "drop")`.
    """

    def _values(self, source: str) -> set[str]:
        return {value for _line, value in bare_vocabulary(source)}

    def test_a_tuple_membership_test_is_caught(self):
        self.assertEqual(
            self._values('def f(step):\n    return step["action"] in ("keep", "drop")\n'),
            {"keep", "drop"})

    def test_a_list_re_declaration_is_caught(self):
        self.assertEqual(self._values('ACTIONS = ["seed", "hub-skip"]\n'),
                         {"seed", "hub-skip"})

    def test_a_set_of_relations_is_caught(self):
        self.assertEqual(self._values('EXPANDS = {"cites", "referenced"}\n'),
                         {"cites", "referenced"})

    def test_an_sql_literal_anywhere_in_a_string_is_caught(self):
        self.assertEqual(
            self._values("SQL = \"SELECT 1 WHERE kind = 'our-document'\"\n"),
            {"our-document"})

    def test_prose_and_dict_keys_are_not_a_declaration(self):
        """The distinction the scan exists to keep: a word in a docstring is
        prose, and a dict KEY is a name (a table, a counter), not a value
        written to the column.
        """
        self.assertEqual(
            self._values('"""A seed is kept or dropped."""\n'
                         'COUNTS = {"cites": 0, "keep": 0}\n'),
            set())


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


class PackagerSharesTheDeclarationTests(unittest.TestCase):
    """The one consumer outside the modules the AST scan walks.

    deploy/manifest_contract.CitationMode is where the packager reads the
    mode, and it used to spell the three values itself -- a second
    declaration of a CHECK-constrained column, in a module no scan covers
    and no live test compared. It now EXTENDS the vocabulary, adding only
    what a build makes of each mode.
    """

    def test_the_packager_reads_the_column_vocabulary_rather_than_its_own(self):
        self.assertTrue(issubclass(CitationMode, PublicPolicyMode))
        self.assertEqual(CitationMode.ALL, PublicPolicyMode.ALL)

    def test_what_it_adds_is_about_builds_not_about_the_column(self):
        for mode in CitationMode.SHIPPED + CitationMode.FULL_CONTENT:
            self.assertIn(mode, PublicPolicyMode.ALL)


class VocabularyMatchesTheDeclarationTests(unittest.TestCase):
    """The Python constants against the SQL, on any checkout.

    Compared as VOCABULARIES and in both directions: an extra value, a
    missing one and a renamed one each leave the two sets different, and
    nothing else does. What this cannot see is what the SERVER carries --
    a declaration is a promise about the next apply -- which is why the live
    comparison below stays beside it.
    """

    def _literals(self, name: str) -> set[str]:
        declared = DECLARATIONS[name]
        found = _wanted_array(declared.constraint).search(declared.sql)
        self.assertIsNotNone(
            found, f"{name}: словарь не найден рядом с {declared.constraint} -- "
                   "форма объявления изменилась, и сравнивать стало нечего")
        return set(re.findall(r"'([^']*)'", found.group(1)))

    def test_every_vocabulary_is_found_and_is_not_empty(self):
        for name in DECLARATIONS:
            self.assertTrue(self._literals(name), name)

    def test_all_five_vocabularies_go_through_the_migrator(self):
        """The count is the assertion: five CHECK-constrained columns exist
        across the two schemas the crawl writes, and every one of them is
        declared as the `wanted` array of citation.ensure_vocabulary_check.

        An inline CHECK is unwidenable on every instance that already has
        the table, and CREATE TABLE IF NOT EXISTS makes every deployed
        instance exactly that case -- so a vocabulary left inline passes the
        offline tests (they read the declaration) and stays a no-op on the
        database. Three of the five had been left that way in turn; the
        measurements mirror was the last.
        """
        self.assertEqual(len(DECLARATIONS), 5)
        for name, declared in DECLARATIONS.items():
            with self.subTest(vocabulary=name):
                self.assertIn("ensure_vocabulary_check", declared.sql)
                self.assertIn(f"'{declared.constraint}'", declared.sql)

    def test_no_vocabulary_literal_is_left_in_the_data_definition(self):
        """The other half of the same rule, and read as WHICH VALUES the
        file spells rather than as one CHECK shape.

        Anchored on `CHECK (<col> IN (` it saw only the vocabulary form.
        citation.work also stated "an our-document row must name its
        document" and "an excluded row must carry its reason" -- one value
        of the same closed vocabulary each, in a `<>` predicate, and
        anonymous besides, so nothing could even find them by name to
        migrate. Every literal outside a comment is compared now, so any
        shape of them fails here.
        """
        definition = re.sub(r"--[^\n]*", "", SCHEMA_FILE.read_text(encoding="utf-8"))
        spelled = set(re.findall(r"'([^']*)'", definition))
        for name, declared in DECLARATIONS.items():
            with self.subTest(vocabulary=name):
                self.assertFalse(
                    spelled & set(declared.python),
                    f"{name}: значение словаря набрано литералом в "
                    f"{SCHEMA_FILE.name} -- CREATE TABLE IF NOT EXISTS ничего "
                    "не меняет на существующем экземпляре; объявляйте через "
                    "миграцию в pg_schema_citation_constraints.sql")
                self.assertNotRegex(declared.sql, rf"CHECK \({declared.column} IN \(")

    def test_the_work_kinds_are_the_ones_the_declaration_allows(self):
        self.assertEqual(self._literals("citation.work.kind"), set(WorkKind.ALL))

    def test_the_crawl_actions_are_the_ones_the_declaration_allows(self):
        self.assertEqual(self._literals("citation.crawl_step.action"),
                         set(CrawlAction.ALL))

    def test_the_relations_are_the_ones_the_declaration_allows(self):
        self.assertEqual(self._literals("citation.crawl_step.relation"),
                         set(Relation.ALL))

    def test_the_measurements_mirror_declares_the_same_pair(self):
        """The calibration records one row per scored candidate and groups
        its whole verdict by `relation`, so the measurements table mirrors
        the journal's vocabulary -- through the same migrator, on a table
        the first calibration created and nothing recreates.
        """
        self.assertEqual(
            self._literals("measurements.citation_frontier_threshold.relation"),
            set(Relation.ALL))

    def test_the_public_policy_modes_are_the_ones_the_declaration_allows(self):
        self.assertEqual(self._literals("citation.public_policy.mode"),
                         set(PublicPolicyMode.ALL))

    def test_a_value_on_one_side_only_is_what_this_catches(self):
        """The failure this exists for, spelled out: the comparison is over
        sets, so neither direction passes by inclusion.
        """
        for name, declared in DECLARATIONS.items():
            literals = self._literals(name)
            self.assertNotEqual(literals, set(declared.python) | {"invented"}, name)
            self.assertNotEqual(literals, set(declared.python) - {declared.python[0]}, name)


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

    def test_the_relations_are_the_ones_the_check_allows(self):
        """The last column promoted out of `reason` and the only one that
        went unconstrained: the crawl BRANCHES on this value, so a fourth
        spelling reaching the journal is a traversal decision made on a
        typo.
        """
        self.assertEqual(
            self._check_vocabulary("citation.crawl_step", "crawl_step_relation_check"),
            set(Relation.ALL))

    def test_the_public_policy_modes_are_the_ones_the_check_allows(self):
        """The vocabulary the OWNER writes into, and the packager reads out
        of: a mode the CHECK rejects cannot be recorded, and a mode the
        Python side does not know is refused by the build
        (deploy/citation_profile.resolve_citation_mode).
        """
        self.assertEqual(
            self._check_vocabulary("citation.public_policy", "public_policy_mode_check"),
            set(PublicPolicyMode.ALL))

    def test_the_measurements_mirror_carries_the_named_constraint(self):
        """The fifth one, on the table a calibration writes. Live because
        this is the case the offline half cannot reach: the table was
        created with an INLINE check, and whether the migrator adopted that
        constraint or left a second one beside it is a fact about the
        instance, not about the DDL string.
        """
        self.assertEqual(
            self._check_vocabulary("measurements.citation_frontier_threshold",
                                   threshold_store.RELATION_CONSTRAINT),
            set(Relation.ALL))


if __name__ == "__main__":
    unittest.main()
