"""`pg_graph.py init` applies from nothing, in the order it declares.

Every other test of the schema runs against the development database,
where the citation schema has existed for a while: CREATE TABLE IF NOT
EXISTS succeeds there whatever the state, ALTER finds its table, and the
vocabulary migrator finds its function. So the one arrangement nobody
exercised is the recipient's -- an empty database, the five files applied
once, in SCHEMA_PATHS order -- which is exactly the arrangement the order
exists for.

A temporary database, not a rolled-back transaction: CREATE EXTENSION and
CREATE DATABASE do not belong inside one, and the subject here is a
database that has never seen this schema. It is dropped again whatever
happens; a run without permission to create one skips.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph_common
from paths import default_corpus_dir, kb_root
from pg_common import (
    PostgresUnavailable,
    check_postgres_available,
    load_pgenv,
    run_sql,
    run_sql_file,
)

CORPUS_SCHEMA_FILE = kb_root() / "pg_schema.sql"
EXTENSIONS_FILE = kb_root() / "deploy" / "init" / "00_extensions.sql"
PROBE = "kb_init_probe"
ORDER_PROBE = "kb_init_order_probe"


def bootstrap(env: dict[str, str]) -> None:
    """Everything that exists BEFORE the citation schema does: the two
    extensions the container's own init applies, and schema corpus, which
    citation.work.document_id references across.
    """
    run_sql_file(env, EXTENSIONS_FILE)
    run_sql_file(env, CORPUS_SCHEMA_FILE)


class FreshDatabaseInitTests(unittest.TestCase):
    created: list[str] = []

    @classmethod
    def setUpClass(cls):
        try:
            cls.admin = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.admin):
            raise unittest.SkipTest("Postgres not reachable")
        cls.created = []

    @classmethod
    def tearDownClass(cls):
        for name in getattr(cls, "created", []):
            run_sql(cls.admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")

    def _fresh(self, name: str) -> dict[str, str]:
        try:
            run_sql(self.admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")
            run_sql(self.admin, f"CREATE DATABASE {name};")
        except RuntimeError as exc:
            raise unittest.SkipTest(f"no permission to create a probe database: {exc}")
        self.created.append(name)
        return dict(self.admin, PGDATABASE=name)

    def test_the_five_files_apply_to_an_empty_database_in_their_own_order(self):
        env = self._fresh(PROBE)
        bootstrap(env)
        pg_graph_common.init_schema(env)
        self.assertTrue(pg_graph_common.citation_schema_exists(env))
        # init defines the projection; `project` is what first builds it,
        # and on an empty schema that is a graph of nothing rather than a
        # refusal -- the recipient's documented first two commands.
        self.assertEqual(pg_graph_common.project(env), (0, 0))
        self.assertTrue(pg_graph_common.graph_exists(env))

    def test_applying_it_twice_changes_nothing_and_raises_nothing(self):
        """The recipient's second `init` -- and ours, on every schema
        change. Same database, so this is the idempotence of the files as
        they stand rather than of a fresh copy."""
        env = self._fresh(PROBE)
        bootstrap(env)
        pg_graph_common.init_schema(env)
        pg_graph_common.init_schema(env)
        self.assertTrue(pg_graph_common.citation_schema_exists(env))

    def test_the_declared_order_is_load_bearing(self):
        """The positive control. Without it, a test that applies the files
        in the order the tuple happens to hold would pass just as well on a
        tuple that had been reordered.
        """
        env = self._fresh(ORDER_PROBE)
        bootstrap(env)
        reversed_order = tuple(reversed(pg_graph_common.SCHEMA_PATHS))
        with self.assertRaises(RuntimeError):
            for path in reversed_order:
                run_sql_file(env, path)


if __name__ == "__main__":
    unittest.main()
