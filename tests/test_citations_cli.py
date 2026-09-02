"""pg_load_citations.py's main() and the two spike modes' write seam.

No network. No database except the live class at the bottom, which keys
everything under a `test:` spike and deletes what it wrote -- the one
property no stub carries is what DELETE ... CASCADE does to a run row's
data rows, which is the whole reason update_run_fields() exists.

What is asserted here is what the CLI PROMISES: --dry-run touches nothing
(the schema DDL included), an exhausted quota is journalled before the
non-zero exit, and the crawl refuses to start without a measured tau.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_graph_common
import pg_load_citations
from citations import hub_report, seed_metadata, spike_runs, threshold_store
from citations.openalex_client import QuotaExhausted
from citations.store import DryRunWriter, PostgresWriter
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql, scalar

ENV = {"PGHOST": "test"}


class _MainHarness:
    """Everything main() reaches on its way to the mode under test."""

    def __init__(self, stack, *, schema_exists=True):
        self.init_schema = stack.enter_context(
            mock.patch.object(pg_load_citations, "init_schema"))
        self.run_sql_file = stack.enter_context(
            mock.patch.object(pg_graph_common, "run_sql_file"))
        self.upsert_run = stack.enter_context(
            mock.patch.object(threshold_store, "upsert_run", return_value=1))
        stack.enter_context(mock.patch.object(pg_load_citations, "load_pgenv",
                                              return_value=ENV))
        stack.enter_context(mock.patch.object(pg_load_citations, "citation_schema_exists",
                                              return_value=schema_exists))
        self.writers: list[DryRunWriter] = []
        stack.enter_context(mock.patch.object(pg_load_citations, "DryRunWriter",
                                              side_effect=self._writer))

    def _writer(self, *_args, **_kwargs):
        self.writers.append(DryRunWriter())
        return self.writers[-1]


class DryRunTouchesNothingTests(unittest.TestCase):
    """DryRunWriter covers citation.*; the CLI as a whole has to cover the
    same promise, and it did not: main() applied pg_schema_citation.sql
    (ALTER TABLE, CREATE OR REPLACE FUNCTION, CREATE INDEX) before the flag
    was ever consulted, and the spike modes wrote measurement rows around
    the seam entirely.
    """

    def _run(self, argv, **kwargs):
        with ExitStack() as stack:
            harness = _MainHarness(stack, **kwargs)
            code = pg_load_citations.main(argv)
        return code, harness

    def test_dry_run_hub_report_applies_no_schema_and_writes_no_run(self):
        # A cache with real batches in it: an empty one is refused before
        # the mode does anything at all, which would let this pass for the
        # wrong reason (see HubReportRefusesAnEmptyMeasurementTests).
        with tempfile.TemporaryDirectory() as cache, \
             mock.patch.object(hub_report, "batch_counts", return_value=[51652]):
            code, harness = self._run(["--hub-report", "--dry-run", "--cache-dir", cache])
        self.assertEqual(code, 0)
        harness.init_schema.assert_not_called()
        harness.run_sql_file.assert_not_called()
        harness.upsert_run.assert_not_called()

    def test_a_real_hub_report_does_apply_the_schema(self):
        """The complement, so the guard cannot pass by never applying it."""
        with tempfile.TemporaryDirectory() as cache, \
             mock.patch.object(pg_load_citations, "record_hub_report", return_value=0):
            code, harness = self._run(["--hub-report", "--cache-dir", cache])
        self.assertEqual(code, 0)
        harness.init_schema.assert_called_once()

    def test_dry_run_without_the_schema_refuses_and_names_the_entry_point(self):
        with tempfile.TemporaryDirectory() as cache:
            with mock.patch("sys.stderr") as stderr:
                code, harness = self._run(
                    ["--hub-report", "--dry-run", "--cache-dir", cache], schema_exists=False)
        self.assertEqual(code, 1)
        harness.init_schema.assert_not_called()
        said = "".join(str(call.args[0]) for call in stderr.write.call_args_list)
        self.assertIn("pg_graph.py init", said)

    def test_dry_run_merge_twins_builds_the_writer_that_cannot_write(self):
        """The fourth mode used to take a dry_run flag and guard two raw
        statements with it; it now gets the same object substitution as the
        other three."""
        with ExitStack() as stack:
            harness = _MainHarness(stack)
            merge = stack.enter_context(
                mock.patch.object(pg_load_citations.twin_pass, "merge_twins",
                                  return_value=[]))
            code = pg_load_citations.main(["--merge-twins", "--dry-run"])
        self.assertEqual(code, 0)
        harness.init_schema.assert_not_called()
        self.assertIs(merge.call_args.args[2], harness.writers[0])

    def test_a_real_merge_twins_writes_through_the_database_writer(self):
        with ExitStack() as stack:
            _harness = _MainHarness(stack)
            merge = stack.enter_context(
                mock.patch.object(pg_load_citations.twin_pass, "merge_twins",
                                  return_value=[]))
            pg_load_citations.main(["--merge-twins"])
        self.assertIsInstance(merge.call_args.args[2], PostgresWriter)

    def test_dry_run_crawl_applies_no_schema_either(self):
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = _MainHarness(stack)
            stack.enter_context(mock.patch.object(pg_load_citations, "resolve_model",
                                                  return_value=("bge-m3", 1024)))
            stack.enter_context(mock.patch.object(pg_load_citations, "corpus_document_ids",
                                                  return_value=["doc_a"]))
            stack.enter_context(mock.patch.object(pg_load_citations, "seed_matches",
                                                  return_value={"doc_a": "W1"}))
            stack.enter_context(mock.patch.object(pg_load_citations, "zbmath_abstracts",
                                                  return_value={}))
            stack.enter_context(mock.patch.object(pg_load_citations, "mathnet_names",
                                                  return_value={}))
            stack.enter_context(mock.patch.object(
                pg_load_citations, "Snowball",
                return_value=mock.Mock(seed_keys=[], run=mock.Mock(return_value={}))))
            code = pg_load_citations.main(
                ["--tau", "0.5", "--dry-run", "--cache-dir", cache])
        self.assertEqual(code, 0)
        harness.init_schema.assert_not_called()
        harness.run_sql_file.assert_not_called()


class ModeExclusivityTests(unittest.TestCase):
    """The four modes are alternatives, and argparse is told so.

    They used to be independent booleans dispatched by a fall-through if
    chain, so `--hub-report --calibrate` ran hub-report and discarded the
    other request in silence -- reporting success for a measurement nobody
    asked for. Precedence by statement order is not an interface.
    """

    def _refused(self, argv):
        with self.assertRaises(SystemExit) as ctx, mock.patch("sys.stderr"):
            pg_load_citations.main(argv)
        return ctx.exception.code

    def test_two_modes_at_once_are_refused(self):
        for argv in (["--hub-report", "--calibrate"],
                     ["--merge-twins", "--hub-report"],
                     ["--calibrate", "--merge-twins"]):
            self.assertEqual(self._refused(argv), 2, argv)


class MainFailurePathTests(unittest.TestCase):
    def test_no_tau_is_an_argparse_error(self):
        with self.assertRaises(SystemExit) as ctx, mock.patch("sys.stderr"):
            pg_load_citations.main(["--depth", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_calibrate_is_what_measures_tau_so_it_needs_none(self):
        """The requirement belongs to the crawl alone: --calibrate exists
        precisely to produce the number, and the two offline modes read what
        is already written.
        """
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            _harness = _MainHarness(stack)
            stack.enter_context(mock.patch.object(pg_load_citations, "resolve_model",
                                                  return_value=None))
            code = pg_load_citations.main(
                ["--calibrate", "--dry-run", "--cache-dir", cache])
        # Stopped at the model check, i.e. well past the tau validation.
        self.assertEqual(code, 1)

    def test_an_exhausted_quota_is_journalled_before_the_non_zero_exit(self):
        message = "осталось 3 запросов OpenAlex, окно сбросится через 83942 с"
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = _MainHarness(stack)
            stack.enter_context(mock.patch.object(pg_load_citations, "resolve_model",
                                                  return_value=("bge-m3", 1024)))
            stack.enter_context(mock.patch.object(pg_load_citations, "corpus_document_ids",
                                                  return_value=["doc_a"]))
            stack.enter_context(mock.patch.object(pg_load_citations, "seed_matches",
                                                  return_value={"doc_a": "W1"}))
            stack.enter_context(mock.patch.object(pg_load_citations, "zbmath_abstracts",
                                                  return_value={}))
            stack.enter_context(mock.patch.object(pg_load_citations, "mathnet_names",
                                                  return_value={}))
            snowball = mock.Mock(seed=mock.Mock(side_effect=QuotaExhausted(message)))
            stack.enter_context(mock.patch.object(pg_load_citations, "Snowball",
                                                  return_value=snowball))
            with mock.patch("sys.stderr"):
                code = pg_load_citations.main(
                    ["--tau", "0.5", "--depth", "2", "--dry-run", "--cache-dir", cache])
        self.assertEqual(code, 2)
        steps = harness.writers[0].steps_seen
        self.assertEqual([s["action"] for s in steps], ["error"])
        self.assertEqual(steps[0]["depth"], 2)
        self.assertIn("83942", steps[0]["reason"])


class HubReportWriteOrderTests(unittest.TestCase):
    """The aggregation pass is a full scan of the largest table in the
    schema, twice expanding evidence->'records'. It ran twice because the
    second upsert_run() -- there only to stamp verify_query with numbers the
    first pass produced -- deleted the run row and cascaded away its data.
    """

    class _Writer(spike_runs.DryRunMeasurementsWriter):
        dry = False  # exercise the writing branch without a database

        def upsert_run(self, spike, fields):
            super().upsert_run(spike, fields)
            return 7

    def _run(self):
        writer = self._Writer()
        rows = [["cites", "384", "9000", "1200", "3", "15000"]]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_report, "batch_counts", return_value=[51652]), \
             mock.patch.object(hub_report, "stats", return_value=rows), \
             mock.patch.object(hub_report, "worst_nodes", return_value=[]):
            code = spike_runs.record_hub_report(ENV, tmp, Path(tmp), writer, 1000)
        return code, writer

    def test_the_aggregation_pass_runs_once(self):
        code, writer = self._run()
        self.assertEqual(code, 0)
        self.assertEqual([name for name, _payload in writer.calls].count("populate"), 1)
        self.assertEqual([name for name, _payload in writer.calls].count("upsert_run"), 1)

    def test_verify_query_is_stamped_by_an_in_place_update(self):
        _code, writer = self._run()
        order = [name for name, _payload in writer.calls]
        self.assertEqual(order.index("update_run_fields") > order.index("populate"), True)
        self.assertIn("update_run_fields", order)

    def test_the_stamped_verify_query_names_the_measured_numbers(self):
        seen = {}
        writer = self._Writer()
        writer.update_run_fields = lambda spike, fields: seen.update(fields)
        rows = [["cites", "384", "9000", "1200", "3", "15000"]]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_report, "batch_counts", return_value=[51652]), \
             mock.patch.object(hub_report, "stats", return_value=rows), \
             mock.patch.object(hub_report, "worst_nodes", return_value=[]):
            spike_runs.record_hub_report(ENV, tmp, Path(tmp), writer, 1000)
        self.assertIn("cites 384 узлов", seen["verify_query"])


class DryRunLeavesTheDataTreeAloneTests(unittest.TestCase):
    """The third channel DRY_RUN_WRITES_NOTHING names: the HTTP caches.

    All three clients sit on the startup path of a non-offline run, all
    three cache into the data tree, and none of them had a seam -- the
    writers made the promise structural for citation.* and measurements.*
    while the crawl quietly mkdir'd three directories inside it.
    """

    def _crawl(self, stack, tree: pathlib.Path, *flags: str) -> int:
        """A crawl with no seeds: enough to CONSTRUCT all three clients,
        which is where the directories appear.
        """
        _harness = _MainHarness(stack)
        cache = tree / "cache"
        for module, name, value in (
            (pg_load_citations, "resolve_model", ("bge-m3", 1024)),
            (pg_load_citations, "corpus_document_ids", []),
            (pg_load_citations, "seed_matches", {}),
            (pg_load_citations, "project", (0, 0)),
            (pg_load_citations, "graph_check", 0),
            (seed_metadata, "seed_matches", {}),
            (seed_metadata, "stored_zbmath_abstracts", {}),
            (seed_metadata, "corpus_seed_documents", []),
            (seed_metadata, "default_zbmath_cache_dir", cache / "zbmath"),
            (seed_metadata, "default_mathnet_cache_dir", cache / "mathnet"),
        ):
            stack.enter_context(mock.patch.object(module, name, return_value=value))
        stack.enter_context(mock.patch.object(
            pg_load_citations, "Snowball",
            return_value=mock.Mock(seed_keys=[], run=mock.Mock(return_value={}))))
        return pg_load_citations.main(
            ["--tau", "0.5", *flags, "--cache-dir", str(cache / "openalex")])

    def _run(self, *flags: str) -> tuple[int, list[str]]:
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            tree = pathlib.Path(tmp)
            code = self._crawl(stack, tree, *flags)
            return code, sorted(str(p.relative_to(tree)) for p in tree.rglob("*"))

    def test_dry_run_creates_no_path_under_the_data_tree(self):
        code, leftovers = self._run("--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(leftovers, [], "--dry-run наследил в дереве данных")

    def test_a_real_run_does_create_the_three_caches(self):
        """The complement, so the guard cannot pass by never caching."""
        _code, made = self._run()
        for directory in ("cache/openalex", "cache/mathnet", "cache/zbmath"):
            self.assertIn(directory, made)


class HubReportRefusesAnEmptyMeasurementTests(unittest.TestCase):
    """batch_counts() answers [] for a missing, empty or foreign cache
    directory instead of raising, and the two modes read DIFFERENT caches in
    practice, so "the cache the crawl never wrote" is a reachable input. Its
    sibling record_calibration refuses an empty input; so does this one now.
    """

    def _writer(self):
        writer = spike_runs.DryRunMeasurementsWriter()
        writer.dry = False  # the writing branch, without a database
        return writer

    def test_a_missing_cache_directory_is_refused_and_writes_nothing(self):
        writer = self._writer()
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stderr"):
            code = spike_runs.record_hub_report(
                ENV, str(Path(tmp) / "never-written"), Path(tmp), writer, 1000)
        self.assertEqual(code, 1)
        self.assertEqual(writer.calls, [])

    def test_a_cache_with_no_cites_batches_is_refused_and_writes_nothing(self):
        writer = self._writer()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_report, "batch_counts", return_value=[]), \
             mock.patch("sys.stderr"):
            code = spike_runs.record_hub_report(ENV, tmp, Path(tmp), writer, 1000)
        self.assertEqual(code, 1)
        self.assertEqual(writer.calls, [])

    def test_the_dry_run_branch_is_refused_too(self):
        """--dry-run prints what it WOULD write; there is nothing to print
        about a measurement of nothing either.
        """
        writer = spike_runs.DryRunMeasurementsWriter()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_report, "batch_counts", return_value=[]), \
             mock.patch("sys.stderr"):
            code = spike_runs.record_hub_report(ENV, tmp, Path(tmp), writer, 1000)
        self.assertEqual(code, 1)
        self.assertEqual(writer.calls, [])


class RunRowUpdateLiveTests(unittest.TestCase):
    """UPDATE in place against the real cascade: the data rows have to
    survive the stamping that a second upsert_run() would destroy.
    """

    SPIKE = "test:citations-cli:update-run-fields"

    @classmethod
    def setUpClass(cls):
        try:
            env = load_pgenv(default_corpus_dir() / ".pgenv")
        except (PostgresUnavailable, RuntimeError) as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(env):
            raise unittest.SkipTest("Postgres not reachable")
        cls.env = env

    # Deleting the fixture rows is not enough to leave the base as found:
    # the BIGSERIAL keeps every id this class consumed, and the versioned
    # dump of measurements (lib/tools/measurements) carries the sequence's
    # setval -- so a test run would show up as drift in a repository it
    # never touched. Rewound to the largest surviving id, which is what the
    # sequence said before the fixture.
    _REWIND = ("SELECT setval(pg_get_serial_sequence('measurements.run', 'id'), "
               "coalesce((SELECT max(id) FROM measurements.run), 1), "
               "(SELECT max(id) FROM measurements.run) IS NOT NULL);")

    def setUp(self):
        self.addCleanup(run_sql, self.env, self._REWIND)
        self.addCleanup(run_sql, self.env,
                        "DELETE FROM measurements.run WHERE spike = :'spike';",
                        {"spike": self.SPIKE})
        run_sql(self.env, hub_report.DDL)
        self.run_id = threshold_store.upsert_run(
            self.env, self.SPIKE,
            {"question": "test fixture", "arbiter": "test fixture",
             "reproduce": "python3 -m unittest test_citations_cli",
             "family": ["all-families"]})
        run_sql(
            self.env,
            "INSERT INTO measurements.citation_hub_expansion "
            "(run_id, work_key, relation, cited_by_count, n_references) "
            "VALUES (:run, 'test:citations-cli:W1', 'cites', 12, 34);",
            variables={"run": str(self.run_id)},
        )

    def _rows(self) -> int:
        return int(scalar(
            self.env,
            "SELECT count(*) FROM measurements.citation_hub_expansion WHERE run_id = :run;",
            variables={"run": str(self.run_id)}))

    def test_stamping_a_field_keeps_the_id_and_the_data_rows(self):
        threshold_store.update_run_fields(self.env, self.SPIKE,
                                          {"verify_query": "SELECT 1; -- ждать: 1 узел"})
        same_id = int(scalar(self.env, "SELECT id FROM measurements.run WHERE spike = :'spike';",
                             variables={"spike": self.SPIKE}))
        self.assertEqual(same_id, self.run_id)
        self.assertEqual(self._rows(), 1, "строки данных ушли каскадом при простановке поля")
        stamped = scalar(self.env, "SELECT verify_query FROM measurements.run WHERE spike = :'spike';",
                         variables={"spike": self.SPIKE})
        self.assertIn("ждать: 1 узел", stamped)

    def test_a_second_upsert_would_have_taken_them(self):
        """Why the update exists at all -- the behaviour it replaces."""
        threshold_store.upsert_run(
            self.env, self.SPIKE,
            {"question": "test fixture", "arbiter": "test fixture",
             "reproduce": "python3 -m unittest test_citations_cli"})
        self.assertEqual(self._rows(), 0)


if __name__ == "__main__":
    unittest.main()
