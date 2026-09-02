"""The constraints a CREATE TABLE cannot establish, and what they refuse.

pg_schema_citation_constraints.sql carries the parts of the definition that
have to arrive as an idempotent migration: the closed-vocabulary CHECKs and
the referential action on citation.work.document_id. The vocabulary halves
are compared in tests/test_citation_vocab.py; what is here is the FK.

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

import unittest

import _pathfix  # noqa: F401

from citation_vocab import WorkKind
from paths import default_corpus_dir, kb_root
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql, scalar

CONSTRAINTS_FILE = kb_root() / "pg_schema_citation_constraints.sql"
SCHEMA_FILE = kb_root() / "pg_schema_citation.sql"

DOCUMENT_ID = "test:citation-fk-document"
WORK_KEY = "test:citation-fk-node"

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

    def test_the_file_is_within_the_line_cap(self):
        self.assertLessEqual(self.sql.count("\n"), 300)


class DocumentDeleteIsRefusedLiveTests(unittest.TestCase):
    """The refusal itself, on the instance that carries the constraint."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")
        if scalar(cls.env, "SELECT to_regclass('citation.work') IS NOT NULL;") != "t":
            raise unittest.SkipTest("citation schema not applied: python3 pg_graph.py init")

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
