"""public.ensure_vocabulary_check() -- the migrator itself, driven.

Every closed vocabulary the crawl writes is a NAMED constraint applied
through this one function -- four of them by
pg_schema_citation_constraints.sql, the fifth by the DDL
citations/threshold_store.py hands the measurements writer -- because
CREATE TABLE IF NOT EXISTS changes nothing on an instance that already
carries the table and an inline CHECK therefore can never be widened again.

The function is declared in `public` (pg_schema_vocabulary.sql), not in
`citation`: two schemas with independent lifecycles declare vocabularies
through it, and the packager ships `citation` in three modes including one
that carries nothing at all. HOME below holds that where the calls are, so
a copy re-introduced into a domain schema fails here.
The function reads what the instance currently has and DROP+ADDs only on a
mismatch -- ALTER TABLE ... ADD CONSTRAINT validates every existing row under
an ACCESS EXCLUSIVE lock, and citation.crawl_step grows by ~100k rows per
depth-2 crawl.

Both halves of that sentence were untested. `init_schema()` twice "must not
raise" is satisfied by a function that replaces the constraint every time
(the lock and the scan on every crawl), and test_citation_vocab.py's live
comparison only reads the END STATE of an already-initialised database --
satisfied by a function whose comparison always reports "match", i.e. one
that never applies a widening at all. That second regression is the
expensive one: the vocabulary would then live only in the Python constant
and in the schema file, and reach the database never.

So: seed a constraint with a DIFFERENT vocabulary and require the replace,
then call again and require the constraint's oid to be untouched. On a TEMP
table, whose whole lifetime is the one psql session the script runs in --
the live schema's own constraints are never the subject.
"""
from __future__ import annotations

import ast
import json
import re
import unittest

import _pathfix  # noqa: F401
from citations import threshold_store
from citation_vocab import CrawlAction, PublicPolicyMode, Relation, WorkKind
import pg_graph_common
from paths import default_corpus_dir, kb_root
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

CONSTRAINTS_FILE = kb_root() / "pg_schema_citation_constraints.sql"
VOCABULARY_FILE = kb_root() / "pg_schema_vocabulary.sql"

# The migrator's schema-neutral home, spelled the same by every caller.
HOME = "public.ensure_vocabulary_check"

# Every vocabulary the schema closes, and the constraint each is applied as.
# Named here so a vocabulary added to citation_vocab.py and applied inline
# fails this module rather than passing every offline test and being a no-op
# on the database (which is the failure this migrator exists for).
APPLIED = {
    "work_kind_check": WorkKind.ALL,
    "crawl_step_action_check": CrawlAction.ALL,
    "crawl_step_relation_check": Relation.ALL,
    "public_policy_mode_check": PublicPolicyMode.ALL,
}

# One session, because a TEMP table does not outlive the psql process. The
# log table collects the readings; the last statement is the only one whose
# output is read.
_PROBE_SQL = """
CREATE TEMP TABLE vocab_probe (v text);
ALTER TABLE vocab_probe ADD CONSTRAINT vocab_probe_check CHECK (v IN ('a', 'b'));
CREATE TEMP TABLE vocab_probe_log (n serial, step text, oid oid, def text);

INSERT INTO vocab_probe_log (step, oid, def)
SELECT 'seeded', oid, pg_get_constraintdef(oid)
FROM pg_constraint WHERE conname = 'vocab_probe_check';

SELECT public.ensure_vocabulary_check(
    'vocab_probe', 'v', 'vocab_probe_check', ARRAY['a', 'b', 'c']);
INSERT INTO vocab_probe_log (step, oid, def)
SELECT 'widened', oid, pg_get_constraintdef(oid)
FROM pg_constraint WHERE conname = 'vocab_probe_check';

SELECT public.ensure_vocabulary_check(
    'vocab_probe', 'v', 'vocab_probe_check', ARRAY['c', 'b', 'a']);
INSERT INTO vocab_probe_log (step, oid, def)
SELECT 'unchanged', oid, pg_get_constraintdef(oid)
FROM pg_constraint WHERE conname = 'vocab_probe_check';

SELECT json_object_agg(step, json_build_array(oid::text, def)) FROM vocab_probe_log;
"""


class EveryVocabularyGoesThroughTheMigratorTests(unittest.TestCase):
    """Offline: the shared mechanism is applied to all four, not to some."""

    SOURCE = CONSTRAINTS_FILE.read_text(encoding="utf-8")

    def test_each_closed_vocabulary_is_applied_as_a_named_constraint(self):
        for constraint, values in APPLIED.items():
            found = re.search(rf"'{constraint}',\s*ARRAY\[([^\]]*)\]", self.SOURCE, re.S)
            with self.subTest(constraint=constraint):
                self.assertIsNotNone(found, f"{constraint} не применяется миграцией")
                self.assertEqual(set(re.findall(r"'([^']*)'", found.group(1))), set(values))

    def test_nothing_applies_a_vocabulary_any_other_way(self):
        """The count is the guard: a fifth call with a name this module does
        not know is a vocabulary nobody is comparing against Python.
        """
        calls = re.findall(rf"{re.escape(HOME)}\(\s*'[^']*',\s*'[^']*',\s*'([^']*)'",
                           self.SOURCE)
        self.assertEqual(sorted(calls), sorted(APPLIED))


class TheMigratorHasNoSchemaOfItsOwnTests(unittest.TestCase):
    """Where the function is declared, and that every caller says so.

    In `citation` it was a schema-agnostic tool owned by a domain schema,
    and the measurements table's own DDL depended on it at runtime -- the
    dependency pointing from the research schema, versioned and dumped on
    its own, into one the packager ships in three modes including none and
    which a database can legitimately not carry at all.
    """

    SOURCES = {
        "pg_schema_vocabulary.sql": VOCABULARY_FILE.read_text(encoding="utf-8"),
        "pg_schema_citation_constraints.sql": CONSTRAINTS_FILE.read_text(encoding="utf-8"),
        "threshold_store.py": threshold_store.THRESHOLD_DDL,
    }

    def test_it_is_declared_once_and_outside_every_domain_schema(self):
        declarations = [name for name, source in self.SOURCES.items()
                        if "CREATE OR REPLACE FUNCTION" in source
                        and "ensure_vocabulary_check" in source]
        self.assertEqual(declarations, ["pg_schema_vocabulary.sql"])
        self.assertIn(f"CREATE OR REPLACE FUNCTION {HOME}(",
                      self.SOURCES["pg_schema_vocabulary.sql"])

    def test_the_older_copy_is_dropped_after_the_new_one_exists(self):
        """Idempotent migration, not a rename in the file only: the instance
        this repository runs against carries the citation copy, and two
        definitions are two answers to which comparison decided.
        """
        source = self.SOURCES["pg_schema_vocabulary.sql"]
        drop = source.index("DROP FUNCTION IF EXISTS citation.ensure_vocabulary_check(")
        self.assertGreater(drop, source.index("CREATE OR REPLACE FUNCTION"))
        self.assertIn("(text, text, text, text[])", source[drop:])

    def test_no_caller_names_a_domain_schema(self):
        for name, source in self.SOURCES.items():
            with self.subTest(source=name):
                self.assertNotIn("citation.ensure_vocabulary_check(", source.replace(
                    "DROP FUNCTION IF EXISTS citation.ensure_vocabulary_check(", ""))

    def test_the_migrator_is_applied_before_anything_calls_it(self):
        paths = list(pg_graph_common.SCHEMA_PATHS)
        self.assertEqual(paths[0], VOCABULARY_FILE)
        self.assertIn(CONSTRAINTS_FILE, paths[1:])


class EnsureVocabularyCheckLiveTests(unittest.TestCase):
    """The migrator's two branches, against the real function."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")
        out = run_sql(cls.env, _PROBE_SQL, extra_args=["-t", "-A"]).stdout
        payload = [line for line in out.splitlines() if line.strip()][-1]
        cls.seen = json.loads(payload)

    def _values(self, step: str) -> set[str]:
        return set(re.findall(r"'([^']*)'", self.seen[step][1]))

    def _oid(self, step: str) -> str:
        return self.seen[step][0]

    def test_a_constraint_carrying_another_vocabulary_is_replaced(self):
        self.assertEqual(self._values("seeded"), {"a", "b"})
        self.assertEqual(self._values("widened"), {"a", "b", "c"})
        self.assertNotEqual(self._oid("seeded"), self._oid("widened"),
                            "constraint не пересоздан: расширение словаря не доехало")

    def test_the_same_vocabulary_leaves_the_constraint_untouched(self):
        """Not "converges to the same values": DROP+ADD takes an ACCESS
        EXCLUSIVE lock and re-validates every row, and this runs on every
        `pg_graph.py init` and at the start of every non-dry-run crawl. The
        oid is what tells the two apart.
        """
        self.assertEqual(self._values("unchanged"), {"a", "b", "c"})
        self.assertEqual(self._oid("widened"), self._oid("unchanged"),
                         "constraint пересоздан впустую: полная валидация на каждый init")

    def test_the_comparison_is_of_values_not_of_their_order(self):
        """The second call passes the same three values in another order --
        the function sorts before comparing, so a reordered ARRAY must not
        look like a change.
        """
        self.assertIn("'c', 'b', 'a'", _PROBE_SQL)
        self.assertEqual(self._oid("widened"), self._oid("unchanged"))


class TheLiveSchemaKeepsItsConstraintsTests(unittest.TestCase):
    """The migrator is called on every init, so "no-op when equal" is a
    claim about the real constraints too, not only about a probe table.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def _oids(self) -> dict:
        names = ", ".join(f"'{name}'" for name in APPLIED)
        out = run_sql(
            self.env,
            f"SELECT json_object_agg(conname, oid::text) FROM pg_constraint "
            f"WHERE conname IN ({names});",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        return json.loads(out)

    def test_re_applying_the_constraints_file_replaces_nothing(self):
        before = self._oids()
        self.assertEqual(sorted(before), sorted(APPLIED),
                         "не все словари применены к живой базе: python3 pg_graph.py init")
        run_sql(self.env, CONSTRAINTS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(self._oids(), before)

    def test_the_live_instance_carries_the_migrator_where_the_calls_look(self):
        """The calls are qualified, so a live instance with the function
        only in the old schema fails every init -- and the DROP in
        pg_schema_vocabulary.sql means the old one is gone by then.
        """
        found = run_sql(
            self.env,
            "SELECT n.nspname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE p.proname = 'ensure_vocabulary_check' ORDER BY 1;",
            extra_args=["-t", "-A"],
        ).stdout.split()
        self.assertEqual(found, ["public"], found)


class TheMigratorIsNotCalledFromPythonTests(unittest.TestCase):
    """Where the vocabulary is applied is the schema file, once.

    A module issuing its own ensure_vocabulary_check() would be a second
    place a vocabulary is declared to the database -- the thing
    VOCABULARY_ONE_DECLARATION forbids, one layer down from the literals.

    ONE exception, named by module: citations/threshold_store.py owns the
    DDL of measurements.citation_frontier_threshold, a spike's own data
    table created by the first calibration and applied through the
    measurements writer (EXTENDING procedure D) -- there is no schema file
    for it to be declared in. Its vocabulary is the journal's `relation`,
    and MEASUREMENTS_MIRROR below holds it to citation_vocab, so the
    exception is a place, not a licence.
    """

    MEASUREMENTS_MIRROR = "threshold_store.py"

    def _spelling_modules(self) -> dict:
        root = kb_root()
        modules = sorted(root.glob("*.py")) + sorted((root / "citations").glob("*.py")) \
            + sorted((root / "deploy").glob("*.py"))
        found = {}
        for path in modules:
            source = path.read_text(encoding="utf-8")
            if "ensure_vocabulary_check" not in source:
                continue
            tree = ast.parse(source)
            spelled = [node.lineno for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and "ensure_vocabulary_check(" in node.value]
            if spelled:
                found[path.name] = spelled
        return found

    def test_only_the_named_exception_issues_the_migration_itself(self):
        self.assertEqual(sorted(self._spelling_modules()), [self.MEASUREMENTS_MIRROR],
                         "миграцию применяет pg_schema_citation_constraints.sql, "
                         "кроме таблицы спайка, у которой нет файла схемы")

    def test_the_exception_declares_the_journals_own_vocabulary(self):
        """The measurements table records one row per scored candidate and
        groups its verdict by `relation`: the same closed vocabulary, read
        from citation_vocab rather than spelled a second time.
        """
        found = re.search(rf"'{threshold_store.RELATION_CONSTRAINT}',\s*ARRAY\[([^\]]*)\]",
                          threshold_store.THRESHOLD_DDL, re.S)
        self.assertIsNotNone(found, threshold_store.THRESHOLD_DDL)
        self.assertEqual(set(re.findall(r"'([^']*)'", found.group(1))), set(Relation.ALL))


if __name__ == "__main__":
    unittest.main()
