"""The constraints a CREATE TABLE cannot establish, and what they refuse.

pg_schema_citation_constraints.sql carries the parts of the definition that
have to arrive as an idempotent migration: the closed-vocabulary CHECKs, the
two conditional statements about a work row, and the referential action on
citation.work.document_id. The vocabulary halves are compared in
tests/test_citation_vocab.py; what is here is the FK and the conditional
pair.

Two halves, as everywhere else in this repository: a static one that any
checkout runs (the file says NO ACTION and says it idempotently), and a live
one that asks the server what it actually carries and what a delete actually
does. The static half cannot see the instance; the live half is the only
place "the delete is refused, and demoting the node first lets it through"
is a fact rather than a claim about SQL.

Everything the live half writes happens inside a transaction that is never
committed: the failing case is rolled back by ON_ERROR_STOP, the passing one
ends in an explicit ROLLBACK.
"""
from __future__ import annotations

import re
import unittest

import _pathfix  # noqa: F401

from citation_vocab import WorkKind
from paths import default_corpus_dir, kb_root
from pg_common import (
    PostgresUnavailable,
    check_postgres_available,
    load_pgenv,
    run_sql,
    run_sql_file,
    scalar,
)

CONSTRAINTS_FILE = kb_root() / "pg_schema_citation_constraints.sql"
SCHEMA_FILE = kb_root() / "pg_schema_citation.sql"

DOCUMENT_ID = "test:citation-fk-document"
WORK_KEY = "test:citation-fk-node"
PROBE_KEY = "test:citation-conditional-check"

# The two conditional statements about a work row: the constraint's name,
# the kind value it is about, and the column that value obliges. The value
# comes from citation_vocab and is never spelled here -- one declaration is
# the whole point of moving these out of the table body.
CONDITIONAL_CHECKS = (
    ("work_our_document_has_document_check", WorkKind.OUR_DOCUMENT, "document_id"),
    ("work_excluded_has_reason_check", WorkKind.EXCLUDED, "exclusion_reason"),
)
CHECK_NAMES = sorted([name for name, _kind, _column in CONDITIONAL_CHECKS]
                     + ["work_kind_check"])


def _live_env() -> dict:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    if scalar(env, "SELECT to_regclass('citation.work') IS NOT NULL;") != "t":
        raise unittest.SkipTest("citation schema not applied: python3 pg_graph.py init")
    return env

# One document, one our-document node pointing at it, then the delete the
# loaders perform on every reload. `evidence` is set because a demoted node
# is an external skeleton, and citation_checks requires those to carry it.
_SETUP = f"""
BEGIN;
INSERT INTO corpus.documents (id, filename, extraction_state)
VALUES ('{DOCUMENT_ID}', 'fk-probe.pdf', 'clean');
INSERT INTO citation.work (key, source, kind, document_id, evidence)
VALUES ('{WORK_KEY}', 'test', '{WorkKind.OUR_DOCUMENT}', '{DOCUMENT_ID}', '{{}}'::jsonb);
"""

_DELETE = f"DELETE FROM corpus.documents WHERE id = '{DOCUMENT_ID}';\n"

_DEMOTE = f"""
UPDATE citation.work
   SET kind = '{WorkKind.EXTERNAL_SKELETON}', document_id = NULL
 WHERE document_id = '{DOCUMENT_ID}';
"""


class ConstraintsFileTests(unittest.TestCase):
    """What any checkout can read, server or not."""

    @classmethod
    def setUpClass(cls):
        cls.sql = CONSTRAINTS_FILE.read_text(encoding="utf-8")

    def test_the_column_no_longer_declares_a_referential_action(self):
        """SET NULL and the table's own CHECK cannot both hold: the action
        would write the NULL the CHECK forbids on the very same row.
        """
        self.assertNotIn("ON DELETE SET NULL", SCHEMA_FILE.read_text(encoding="utf-8"))
        self.assertIn("REFERENCES corpus.documents(id),",
                      SCHEMA_FILE.read_text(encoding="utf-8"))

    def test_the_migration_re_adds_the_key_without_an_action(self):
        """Read over the STATEMENTS, not the file: the comment above them
        names the old action, and saying why it went is not carrying it.
        """
        statements = "\n".join(line for line in self.sql.splitlines()
                                if not line.lstrip().startswith("--"))
        self.assertIn("ADD CONSTRAINT work_document_id_fkey", statements)
        self.assertNotIn("ON DELETE", statements)

    def test_the_migration_replaces_only_a_key_that_differs(self):
        """ADD CONSTRAINT on a foreign key re-validates every row, and this
        file runs on every init and every crawl, so the comparison against
        what the instance already carries is the point of the block.
        """
        self.assertIn("confdeltype", self.sql)
        self.assertIn("on_delete = 'a'", self.sql)

    def test_both_conditional_checks_are_declared_through_the_migrator(self):
        """Each states its rule about ONE value of the kind vocabulary, so
        each was unwidenable inline for the reason work_kind_check was:
        CREATE TABLE IF NOT EXISTS is a no-op on every instance that already
        carries the table. Anonymous besides, so nothing could find them by
        name to migrate.

        The value is read out of the declaration and compared with
        citation_vocab, and the CHECK body is built from that same variable
        -- so the literal is spelled once per constraint and a rename cannot
        reach one half only.
        """
        for name, kind, column in CONDITIONAL_CHECKS:
            with self.subTest(constraint=name):
                found = re.search(
                    r"DECLARE kind_value text := '([^']*)';\s*BEGIN\s*"
                    r"PERFORM public\.ensure_check_constraint\(\s*"
                    rf"'citation\.work', '{name}', ARRAY\[kind_value\],\s*"
                    r"format\('([^']*)', kind_value\)\);", self.sql)
                self.assertIsNotNone(
                    found, f"{name}: объявление не найдено или сменило форму")
                self.assertEqual(found.group(1), kind)
                self.assertIn(f"{column} IS NOT NULL", found.group(2))
                self.assertIn("%L", found.group(2))

    def test_the_data_definition_states_neither_of_them(self):
        statements = "\n".join(
            line for line in SCHEMA_FILE.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("--"))
        for _name, kind, _column in CONDITIONAL_CHECKS:
            self.assertNotIn(f"'{kind}'", statements)

    def test_the_file_is_within_the_line_cap(self):
        self.assertLessEqual(self.sql.count("\n"), 300)


class ConditionalChecksLiveTests(unittest.TestCase):
    """The two rules on the instance that carries them: named, singly, and
    enforcing.

    work_check / work_check1 -- Postgres' own names for an inline table
    CHECK -- are what the migration replaces, and a copy left beside the
    named one would go on enforcing the unwidenable half. So the assertion
    is over the WHOLE set of check constraints citation.work carries, not
    over the presence of the two.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _column(self, sql: str) -> str:
        return scalar(
            self.env,
            "SELECT coalesce(string_agg(x, ',' ORDER BY x), '') FROM ("
            f"{sql}) AS t(x);")

    def test_the_instance_carries_exactly_the_named_checks(self):
        self.assertEqual(
            self._column("SELECT conname FROM pg_constraint "
                         "WHERE conrelid = 'citation.work'::regclass AND contype = 'c'"),
            ",".join(CHECK_NAMES),
            "примените схему: python3 pg_graph.py init")

    def test_a_row_breaking_either_rule_is_refused_by_name(self):
        for name, kind, _column in CONDITIONAL_CHECKS:
            with self.subTest(constraint=name):
                with self.assertRaises(RuntimeError) as ctx:
                    run_sql(self.env, "BEGIN;\n"
                            "INSERT INTO citation.work (key, source, kind, evidence) "
                            f"VALUES ('{PROBE_KEY}', 'test', '{kind}', '{{}}'::jsonb);\n"
                            "ROLLBACK;\n")
                self.assertIn(name, str(ctx.exception))

    def test_re_applying_the_file_replaces_nothing(self):
        """ADD CONSTRAINT validates every existing row under an ACCESS
        EXCLUSIVE lock, and this file runs on every init and at the start of
        every non-dry-run crawl. The oid is what says nothing was replaced
        -- the same subject tests/test_vocabulary_migration.py holds the
        vocabulary half to.
        """
        oids = ("SELECT oid::text FROM pg_constraint "
                "WHERE conrelid = 'citation.work'::regclass AND contype = 'c'")
        before = self._column(oids)
        run_sql_file(self.env, CONSTRAINTS_FILE)
        self.assertEqual(self._column(oids), before)

    def test_the_probe_left_nothing_behind(self):
        self.assertEqual(
            scalar(self.env, f"SELECT count(*) FROM citation.work WHERE key = '{PROBE_KEY}';"),
            "0")


class DocumentDeleteIsRefusedLiveTests(unittest.TestCase):
    """The refusal itself, on the instance that carries the constraint."""

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_the_instance_carries_no_action(self):
        self.assertEqual(
            scalar(self.env,
                   "SELECT confdeltype FROM pg_constraint "
                   "WHERE conrelid = 'citation.work'::regclass AND contype = 'f' "
                   "AND confrelid = 'corpus.documents'::regclass;"),
            "a", "apply the schema: python3 pg_graph.py init")

    def test_deleting_a_seeded_document_is_refused_by_the_foreign_key(self):
        """And refused AS a foreign key: under the old SET NULL the same
        delete died on the CHECK of a row the deleter never named, which
        says nothing about what to do next.
        """
        with self.assertRaises(RuntimeError) as ctx:
            run_sql(self.env, _SETUP + _DELETE + "ROLLBACK;\n")
        message = str(ctx.exception)
        self.assertIn("work_document_id_fkey", message)
        self.assertIn("foreign key constraint", message)
        self.assertIn(DOCUMENT_ID, message)
        self.assertNotIn("check constraint", message.lower())

    def test_demoting_the_node_first_lets_the_delete_through(self):
        """EXTENDING.md procedure A's step, run as written."""
        run_sql(self.env, _SETUP + _DEMOTE + _DELETE + "ROLLBACK;\n")

    def test_the_probe_left_nothing_behind(self):
        for statement in (
            f"SELECT count(*) FROM corpus.documents WHERE id = '{DOCUMENT_ID}';",
            f"SELECT count(*) FROM citation.work WHERE key = '{WORK_KEY}';",
        ):
            self.assertEqual(scalar(self.env, statement), "0", statement)


if __name__ == "__main__":
    unittest.main()
