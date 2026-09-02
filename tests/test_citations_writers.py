"""The two writers as ONE seam, driven through the same call sequence.

Split from test_citations_crawl.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that file is about what the BFS keeps, drops and journals, and
this one about what the two implementations of citations/store.Writer do
with it. The comparison is the point -- crawl.py swaps them by --dry-run,
and the dry run is what decides whether a real, quota-spending crawl is
worth launching, so trusting each separately is exactly how DryRunWriter's
works() came to add the accumulated total on every call.
"""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import work

from citations import journal, store, store_sql
from citations.registry import Node
from citations.store import DryRunWriter, PostgresWriter, Writer
from pg_copy import CopyResult


class WriterConformanceTests(unittest.TestCase):
    """The two writers are one seam: crawl.py swaps them by --dry-run, and
    the dry run is what decides whether a real, quota-spending crawl is
    worth launching. So both are driven through the SAME call sequence and
    their answers compared, rather than each being trusted separately --
    DryRunWriter.works() used to add the accumulated total on every call,
    which nothing noticed until the second batch.
    """

    @staticmethod
    def _nodes(*keys):
        made = []
        for key in keys:
            node = Node(key=key, kind="external-skeleton", depth=1)
            node.absorb(work(key, title=f"Title {key}"))
            made.append(node)
        return made

    @staticmethod
    def _steps(*keys):
        return [journal.fetch("c", 1, key, 1, 1) for key in keys]

    def _drive(self, writer):
        """One fixed sequence; every per-call answer plus the running counts
        after each write."""
        trace = []
        for nodes in (self._nodes("W1", "W2"), self._nodes("W3"), []):
            trace.append(("works", writer.works(nodes), dict(writer.counts)))
        for edges in ([("W1", "W2", "cites", "W1")], [("W3", "W1", "cites", "W3")], []):
            trace.append(("edges", writer.edges(edges), dict(writer.counts)))
        for steps in (self._steps("W1", "W2"), self._steps("W3"), []):
            trace.append(("journal", writer.journal(steps), dict(writer.counts)))
        return trace

    def test_both_implementations_satisfy_the_writer_protocol(self):
        self.assertIsInstance(DryRunWriter(), Writer)
        self.assertIsInstance(PostgresWriter({}), Writer)

    def test_the_same_call_sequence_produces_the_same_counts(self):
        """With nothing in the database to refuse a row, the live writer
        reports what the dry run predicts -- the counts diverge only where
        the upsert actually refuses (an edge already known, a promote key
        with no work row), which is the live half of this pair.
        """
        def accept_everything(env, table_columns, rows, **kwargs):
            written = sum(1 for _ in rows)
            return CopyResult(written, f"{written}\n" if kwargs.get("epilogue") else "")

        with mock.patch.object(store, "copy_csv_rows", side_effect=accept_everything):
            live = self._drive(PostgresWriter({}))
        self.assertEqual(self._drive(DryRunWriter()), live)

    def test_a_write_is_one_script_over_a_temp_staging_table(self):
        """Staging DDL, the \\copy and the upsert in one psql invocation:
        a shared, globally named staging table between three invocations was
        observable -- and droppable -- by any writer running at the time.
        """
        calls = []
        with mock.patch.object(store, "copy_csv_rows",
                                side_effect=lambda env, target, rows, **kw: (
                                    calls.append((target, list(rows), kw)),
                                    CopyResult(2, "2\n"))[1]):
            PostgresWriter({}).works(self._nodes("W1", "W2"))
        self.assertEqual(len(calls), 1)
        target, rows, kwargs = calls[0]
        self.assertTrue(target.startswith("stage_work ("), target)
        self.assertEqual(len(rows), 2)
        self.assertIn("CREATE TEMP TABLE stage_work", kwargs["preamble"])
        self.assertIn("ON COMMIT DROP", kwargs["preamble"])
        self.assertIn("INSERT INTO citation.work", kwargs["epilogue"])

    def test_the_number_reported_is_the_one_the_database_counted(self):
        """Not the number submitted: the cites upsert drops self-edges and
        skips edges the graph already has.
        """
        with mock.patch.object(store, "copy_csv_rows",
                                return_value=CopyResult(3, "1\n")):
            writer = PostgresWriter({})
            accepted = writer.edges([("W1", "W2", "cites", "W1"),
                                     ("W1", "W3", "cites", "W1"),
                                     ("W1", "W4", "cites", "W1")])
        self.assertEqual(accepted, 1)
        self.assertEqual(writer.counts["cites"], 1)

    def test_no_staging_relation_is_named_in_the_shared_schema(self):
        text = pathlib.Path(store_sql.__file__).read_text(encoding="utf-8")
        self.assertNotIn("citation.stage_", text,
                         "staging is TEMP and session-private")

    def test_counts_are_rows_accepted_by_this_call_not_the_running_total(self):
        writer = DryRunWriter()
        self.assertEqual(writer.works(self._nodes("W1", "W2")), 2)
        self.assertEqual(writer.works(self._nodes("W3")), 1)
        self.assertEqual(writer.counts["work"], 3)


if __name__ == "__main__":
    unittest.main()
