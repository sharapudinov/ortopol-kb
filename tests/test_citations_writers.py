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

import contextlib
import io
import json
import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import work

from citations import journal, store, store_sql
from citations.registry import Node
from citations.dry_store import DryRunWriter
from citations.store import PostgresWriter, Writer
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
        after each write. Every method of the protocol is called, and called
        UNCONDITIONALLY -- which is the contract the modes are held to: a
        step a mode short-circuits on writer.dry is a step the seam no
        longer covers.
        """
        trace = []
        for nodes in (self._nodes("W1", "W2"), self._nodes("W3"), []):
            trace.append(("works", writer.works(nodes), dict(writer.counts)))
        for edges in ([("W1", "W2", "cites", "W1")], [("W3", "W1", "cites", "W3")], []):
            trace.append(("edges", writer.edges(edges), dict(writer.counts)))
        for steps in (self._steps("W1", "W2"), self._steps("W3"), []):
            trace.append(("journal", writer.journal(steps), dict(writer.counts)))
        trace.append(("project", writer.project().code, dict(writer.counts)))
        trace.append(("census", isinstance(writer.census(), str), dict(writer.counts)))
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

        with mock.patch.object(store, "copy_csv_rows", side_effect=accept_everything), \
             mock.patch.object(store, "project_graph", return_value=(3, 2)) as projected, \
             mock.patch.object(store, "projection_diff",
                               return_value=mock.Mock(vertex_n=3, edge_n=2)), \
             mock.patch.object(store, "projection_faults", return_value=[]), \
             mock.patch.object(store, "kind_counts", return_value={"indexed": 3}):
            live = self._drive(PostgresWriter({}))
        projected.assert_called_once()
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


class RowPayloadTests(unittest.TestCase):
    """What the writer actually puts on the wire, value by value.

    The counting tests above hold the number of rows; nothing held their
    CONTENT. _work_row() is twelve positional values against a twelve-name
    column list -- two of them swapped is a work whose title is its abstract,
    and every offline test would still pass -- and evidence_of() decides,
    per field, whether it appears at all. The only assertion on that payload
    used to require a live Postgres, i.e. was skipped on every machine
    without the corpus container.
    """

    @staticmethod
    def _captured(call):
        """The rows a writer streamed, materialised from the generator it
        handed copy_csv_rows(). Taken from the seam, not from a return
        value: the generator IS the payload, and it is consumed inside the
        copy the test replaces.
        """
        seen = {}

        def capture(env, target, rows, **kwargs):
            seen["target"] = target
            seen["rows"] = [list(row) for row in rows]
            return CopyResult(len(seen["rows"]), f"{len(seen['rows'])}\n")

        with mock.patch.object(store, "copy_csv_rows", side_effect=capture):
            call()
        return seen

    def _node(self):
        """A node carrying every conditional field evidence_of() knows: an
        abstract recovered from zbMATH rather than OpenAlex, the zbMATH id it
        came from, how the node reached the frontier and from where, and the
        score it passed tau with.
        """
        node = Node(key="W1", kind="external-skeleton", depth=1,
                    relation="cites", discovered_from="W_SEED")
        node.absorb(work("W1", title="Приближение", doi="10.1/x", year=1997))
        node.abstract, node.abstract_source = "обзор zbMATH", "zbmath"
        node.zbmath_id = "1234.56789"
        node.score = 0.6123456789
        node.embedding = [0.5, -0.25]
        return node

    def test_every_one_of_the_twelve_values_is_the_column_it_lands_in(self):
        node = self._node()
        seen = self._captured(lambda: PostgresWriter({}, source="openalex").works([node]))
        self.assertEqual(seen["target"], f"stage_work ({', '.join(store_sql.WORK_COLUMNS)})")
        row = dict(zip(store_sql.WORK_COLUMNS, seen["rows"][0]))
        self.assertEqual(len(seen["rows"][0]), len(store_sql.WORK_COLUMNS))
        self.assertEqual(row["key"], "W1")
        self.assertEqual(row["doi"], "10.1/x")
        self.assertEqual(row["title"], "Приближение")
        self.assertEqual(row["abstract"], "обзор zbMATH")
        self.assertEqual(row["year"], 1997)
        self.assertEqual(json.loads(row["authors"]), ["I. I. Sharapudinov"])
        self.assertEqual(json.loads(row["external_ids"])["openalex"], ["W1"])
        self.assertEqual(row["source"], "openalex")
        self.assertEqual(row["kind"], "external-skeleton")
        self.assertIsNone(row["document_id"])
        self.assertEqual(row["embedding"], "[0.5,-0.25]")

    def test_the_evidence_carries_the_provenance_a_verdict_rests_on(self):
        node = self._node()
        seen = self._captured(lambda: PostgresWriter({}).works([node]))
        row = dict(zip(store_sql.WORK_COLUMNS, seen["rows"][0]))
        evidence = json.loads(row["evidence"])
        self.assertEqual(evidence["abstract_source"], "zbmath")
        self.assertEqual(evidence["zbmath_id"], "1234.56789")
        self.assertEqual(evidence["relation"], "cites")
        self.assertEqual(evidence["discovered_from"], "W_SEED")
        self.assertEqual(evidence["frontier_score"], 0.612346)
        self.assertEqual([r["id"] for r in evidence["records"]],
                         ["https://openalex.org/W1"])
        self.assertNotIn("referenced_works", evidence["records"][0],
                         "список ссылок в evidence — самое объёмное поле записи")

    def test_a_node_with_nothing_to_say_says_nothing(self):
        """Each conditional field is absent, not null: a key present with a
        null value asserts the crawl looked and found nothing, which is a
        different claim from never having asked.
        """
        node = Node(key="W2", kind="external-skeleton", depth=1)
        node.absorb(work("W2", title="Bare"))
        self.assertEqual(set(PostgresWriter({}).evidence_of(node)), {"records"})

    def test_a_seed_score_of_zero_is_still_a_score(self):
        """`if node.score is not None`, not `if node.score`: 0.0 is a
        measured cosine, and the absent key would say it was never measured.
        """
        node = Node(key="W3", kind="external-skeleton", depth=1)
        node.absorb(work("W3", title="Orthogonal"))
        node.score = 0.0
        self.assertEqual(PostgresWriter({}).evidence_of(node)["frontier_score"], 0.0)

    def test_an_edge_row_is_its_two_endpoints_the_source_and_its_evidence(self):
        seen = self._captured(lambda: PostgresWriter({}, source="openalex").edges(
            [("W1", "W2", "cites", "W_SEED")]))
        self.assertEqual(seen["target"], f"stage_cites ({', '.join(store_sql.CITES_COLUMNS)})")
        row = dict(zip(store_sql.CITES_COLUMNS, seen["rows"][0]))
        self.assertEqual(row["citing_key"], "W1")
        self.assertEqual(row["cited_key"], "W2")
        self.assertEqual(row["source"], "openalex")
        self.assertEqual(json.loads(row["evidence"]),
                         {"relation": "cites", "fetched_from": "W_SEED"})

    def test_a_journal_row_is_the_step_read_off_by_column_name(self):
        """Positional again, and against a table this one writes directly:
        the journal has no upsert to refuse a mis-shaped row, and its COPY is
        all-or-nothing for the whole level.
        """
        step = journal.keep("c", 2, "W2", "W_NODE", 0.61, 0.5, "cites", "W_F")
        seen = self._captured(lambda: PostgresWriter({}).journal([step]))
        self.assertEqual(seen["target"],
                         f"citation.crawl_step ({', '.join(store_sql.STEP_COLUMNS)})")
        row = dict(zip(store_sql.STEP_COLUMNS, seen["rows"][0]))
        self.assertEqual(row, {column: step.get(column) for column in store_sql.STEP_COLUMNS})
        self.assertEqual(row["node_key"], "W_NODE")
        self.assertEqual(row["score"], 0.61)
        self.assertIsNone(row["cited_by_count"], "у keep-строки нет счётчика цитирующих")

    def test_a_promotion_row_is_the_four_names_of_the_twin_rule(self):
        merged = [{"key": "W_EN", "document_id": "2019_rm9846",
                   "seed_key": "W_RU", "rule": "doi"}]
        seen = self._captured(lambda: PostgresWriter({}).promote(merged))
        self.assertEqual(seen["target"],
                         f"stage_twin ({', '.join(store_sql.PROMOTE_COLUMNS)})")
        self.assertEqual(dict(zip(store_sql.PROMOTE_COLUMNS, seen["rows"][0])),
                         {"key": "W_EN", "document_id": "2019_rm9846",
                          "seed_key": "W_RU", "rule": "doi"})


class ProjectionIsAWriteThroughTheSeamTests(unittest.TestCase):
    """citation.project_graph() rewrites the citation_graph label tables --
    a database WRITE, and the fourth one a crawl makes.

    It lived in two argparse mode bodies behind `if writer.dry` / `if not
    writer.dry`, i.e. as a flag check in the CLI rather than as a property
    of which object was constructed. A fifth mode, or any programmatic
    driver calling Snowball.run() without a command line, would then either
    reproject under a dry writer or leave the graph stale until
    corpus_completeness.py reported PROJECTION STALE much later.
    """

    def test_the_protocol_declares_it_beside_the_other_writes(self):
        for name in ("works", "edges", "journal", "promote", "project", "census"):
            with self.subTest(method=name):
                self.assertIn(name, Writer.__annotations__ | vars(Writer))

    def test_the_live_writer_projects_and_returns_the_verdict_as_a_value(self):
        """The faults are folded into the outcome, so the caller that
        prints `report` prints the whole answer once."""
        with mock.patch.object(store, "project_graph", return_value=(441, 2427)) as project, \
             mock.patch.object(store, "projection_diff", return_value="reading"), \
             mock.patch.object(store, "projection_faults",
                               return_value=["work=441 vertices=440"]) as faults:
            outcome = PostgresWriter({"PGHOST": "x"}).project()
        project.assert_called_once_with({"PGHOST": "x"})
        faults.assert_called_once_with("reading")
        self.assertEqual(outcome.code, 1)
        self.assertIn("V=441", outcome.report)
        self.assertIn("E=2427", outcome.report)
        self.assertIn("work=441 vertices=440", outcome.report)

    def test_a_faithful_projection_is_code_zero_and_one_report_line(self):
        reading = mock.Mock(vertex_n=441, edge_n=2427)
        with mock.patch.object(store, "project_graph", return_value=(441, 2427)), \
             mock.patch.object(store, "projection_diff", return_value=reading), \
             mock.patch.object(store, "projection_faults", return_value=[]):
            outcome = PostgresWriter({}).project()
        self.assertEqual(outcome.code, 0)
        self.assertEqual(outcome.report.count("\n"), 0)

    def test_a_graph_that_was_never_projected_is_a_fault_not_a_crash(self):
        with mock.patch.object(store, "project_graph", return_value=(0, 0)), \
             mock.patch.object(store, "projection_diff", return_value=None):
            outcome = PostgresWriter({}).project()
        self.assertEqual(outcome.code, 1)
        self.assertIn("не строилась", outcome.report)

    def test_the_library_writes_nothing_to_stdout_or_stderr(self):
        """check() is pg_graph_common's CLI shape -- it prints its own
        verdict and returns an exit code. Called from here, one projection
        produced two overlapping report lines, and any embedder of the
        crawl without a command line got library output it never asked for.
        """
        out, err = io.StringIO(), io.StringIO()
        reading = mock.Mock(vertex_n=441, edge_n=2427)
        with mock.patch.object(store, "project_graph", return_value=(441, 2427)), \
             mock.patch.object(store, "projection_diff", return_value=reading), \
             mock.patch.object(store, "projection_faults", return_value=["mismatch"]), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            PostgresWriter({}).project()
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_the_dry_writer_projects_nothing_and_says_so(self):
        with mock.patch.object(store, "project_graph") as project, \
             mock.patch.object(store, "projection_diff") as diff:
            outcome = DryRunWriter().project()
        project.assert_not_called()
        diff.assert_not_called()
        self.assertEqual(outcome.code, 0)
        self.assertIn("--dry-run", outcome.report)

    def test_the_census_is_read_back_only_by_the_writer_that_filled_the_table(self):
        with mock.patch.object(store, "kind_counts",
                               return_value={"our-document": 72, "indexed": 3}):
            live = PostgresWriter({}).census()
        self.assertIn("our-document=72", live)
        with mock.patch.object(store, "kind_counts") as counts:
            dry = DryRunWriter().census()
        counts.assert_not_called()
        self.assertIn("--dry-run", dry)


if __name__ == "__main__":
    unittest.main()
