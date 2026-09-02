"""pg_load_citations.py's main(): what the CLI PROMISES and what it says.

No network, no database. The write ORDER of the two spike modes is
test_spike_runs.py's -- here the modes are only driven, and what is asserted
is the loader's own half: --dry-run touches nothing (the schema DDL
included), a refusal names the entry point that fixes it, an exhausted quota
is journalled before the non-zero exit, and the crawl refuses to start
without a measured tau.
"""
from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_graph_common
import pg_load_citations
from citations import hub_cache, hub_report, seed_metadata, spike_runs, threshold_store
from citations.openalex_client import OpenAlexError, QuotaExhausted
from citations.store import DryRunWriter, PostgresWriter

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
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]):
            code, harness = self._run(["--hub-report", "--dry-run", "--cache-dir", cache])
        self.assertEqual(code, 0)
        harness.init_schema.assert_not_called()
        harness.run_sql_file.assert_not_called()
        harness.upsert_run.assert_not_called()

    def test_dry_run_without_the_schema_refuses_and_names_the_entry_point(self):
        """Every mode that runs under --dry-run, not just the one that found
        it: the guard sits above the dispatch, and a test for one branch of
        it says nothing about the other two -- applying the schema IS a
        write, so the refusal is the promise itself for all of them.
        """
        with tempfile.TemporaryDirectory() as cache:
            for argv in (["--hub-report", "--dry-run", "--cache-dir", cache],
                         ["--merge-twins", "--dry-run"],
                         ["--tau", "0.5", "--dry-run", "--cache-dir", cache]):
                with self.subTest(argv=argv), mock.patch("sys.stderr") as stderr:
                    code, harness = self._run(argv, schema_exists=False)
                    self.assertEqual(code, 1)
                    harness.init_schema.assert_not_called()
                    harness.run_sql_file.assert_not_called()
                    said = "".join(str(call.args[0])
                                   for call in stderr.write.call_args_list)
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

    @staticmethod
    def _harness_for(stack) -> _MainHarness:
        """Everything main() reaches before the crawl itself raises."""
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
        return harness

    def test_an_exhausted_quota_is_journalled_before_the_non_zero_exit(self):
        message = "осталось 3 запросов OpenAlex, окно сбросится через 83942 с"
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = self._harness_for(stack)
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

    def test_a_failed_request_is_journalled_before_its_own_non_zero_exit(self):
        """The other way a crawl stops mid-flight: OpenAlex answered with a
        status or a body the client could not use, after every retry. Same
        journal row as an exhausted quota -- the run's own record of why it
        stopped -- and a different code, because the answer differs: a quota
        is waited out, this is looked into.
        """
        message = "GET /works?filter=... -> 500 after 5 attempts"
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = self._harness_for(stack)
            snowball = mock.Mock(seed=mock.Mock(side_effect=OpenAlexError(message)))
            stack.enter_context(mock.patch.object(pg_load_citations, "Snowball",
                                                  return_value=snowball))
            with mock.patch("sys.stderr"):
                code = pg_load_citations.main(
                    ["--tau", "0.5", "--depth", "2", "--dry-run", "--cache-dir", cache])
        self.assertEqual(code, 3)
        steps = harness.writers[0].steps_seen
        self.assertEqual([s["action"] for s in steps], ["error"])
        self.assertEqual(steps[0]["depth"], 2)
        self.assertIn("500", steps[0]["reason"])


class HubReportCliTests(unittest.TestCase):
    """The mode's reporting half, which is the loader's: which refusals the
    process makes, with what exit code, and what a dry run leaves behind.

    The write order the mode follows once it gets that far is
    test_spike_runs.py's; nothing here reaches a writer that writes.
    """

    def _page(self, directory: pathlib.Path, name: str, count: int) -> None:
        body = {"meta": {"count": count, "x_query": {
            "oql": "works where it cites (W1)",
            "url": "/works?filter=referenced_works:W1"}}}
        (directory / name).write_text(json.dumps(body), encoding="utf-8")

    def _main(self, argv: list[str]) -> tuple[int, str, _MainHarness]:
        out = io.StringIO()
        with ExitStack() as stack:
            harness = _MainHarness(stack)
            stack.enter_context(mock.patch("sys.stderr", out))
            stack.enter_context(redirect_stdout(out))
            code = pg_load_citations.main(argv)
        return code, out.getvalue(), harness

    def test_a_missing_cache_directory_is_refused_and_never_created(self):
        """A working cache creates its own directory, so the check has to
        happen before the object does -- otherwise the mode measures the
        empty directory it just made.
        """
        with tempfile.TemporaryDirectory() as tmp:
            absent = pathlib.Path(tmp) / "never-written"
            code, said, harness = self._main(
                ["--hub-report", "--cache-dir", str(absent)])
            self.assertEqual(code, 1)
            self.assertFalse(absent.exists())
        self.assertIn("кэша ответов нет", said)
        harness.upsert_run.assert_not_called()

    def test_a_cache_with_no_cites_batches_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as cache:
            code, said, harness = self._main(["--hub-report", "--cache-dir", cache])
        self.assertEqual(code, 1)
        self.assertIn("ни одного батча cites", said)
        self.assertIn(cache, said)
        harness.upsert_run.assert_not_called()

    def test_a_dry_run_over_a_real_cache_adds_no_sidecar(self):
        """The whole point of handing the mode a cache OBJECT: the reading
        pass writes a sidecar per page it had to parse, and under --dry-run
        that write must not happen -- the data tree is exactly as found.
        """
        with tempfile.TemporaryDirectory() as cache:
            directory = pathlib.Path(cache)
            self._page(directory, "a.json", 18904)
            code, said, harness = self._main(
                ["--hub-report", "--dry-run", "--cache-dir", cache])
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ["a.json"])
        self.assertEqual(code, 0)
        self.assertIn("18904", said)
        harness.upsert_run.assert_not_called()

    def test_a_real_run_prints_the_run_it_wrote(self):
        """Also the complement to the dry-run guard above: a run that is not
        dry DOES apply the schema, so the guard cannot pass by never
        applying it at all.
        """
        record = spike_runs.HubRecord(
            [18904], 7, [["cites", "384", "9000", "1200", "3", "15000"]],
            pathlib.Path("research/citation-hub/report.md"))
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = _MainHarness(stack)
            stack.enter_context(mock.patch.object(
                pg_load_citations, "record_hub_report", return_value=record))
            out = io.StringIO()
            stack.enter_context(redirect_stdout(out))
            code = pg_load_citations.main(["--hub-report", "--cache-dir", cache])
        self.assertEqual(code, 0)
        harness.init_schema.assert_called_once()
        self.assertIn("run 7", out.getvalue())
        self.assertIn("узлов depth-1: 384", out.getvalue())


class CalibrateCliTests(unittest.TestCase):
    """The same division for the other spike mode: the refusal is the
    writer seam's to raise and the loader's to report.
    """

    def _drive(self, record_side_effect):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            _harness = _MainHarness(stack)
            for name, value in (("resolve_model", ("bge-m3", 1024)),
                                ("corpus_document_ids", ["doc_a"]),
                                ("seed_matches", {"doc_a": "W1"}),
                                ("zbmath_abstracts", {}),
                                ("mathnet_names", {})):
                stack.enter_context(mock.patch.object(pg_load_citations, name,
                                                      return_value=value))
            stack.enter_context(mock.patch.object(
                pg_load_citations, "Snowball",
                return_value=mock.Mock(seed_keys=[])))
            stack.enter_context(mock.patch.object(
                pg_load_citations, "record_calibration", **record_side_effect))
            stack.enter_context(mock.patch("sys.stderr", out))
            stack.enter_context(redirect_stdout(out))
            code = pg_load_citations.main(
                ["--calibrate", "--dry-run", "--cache-dir", cache])
        return code, out.getvalue()

    def test_nothing_to_calibrate_is_a_message_and_exit_one(self):
        code, said = self._drive(
            {"side_effect": spike_runs.NothingToMeasure("кандидатов depth-1 нет")})
        self.assertEqual(code, 1)
        self.assertIn("кандидатов depth-1 нет", said)

    def test_a_dry_calibration_says_what_it_would_have_written(self):
        record = spike_runs.CalibrationRecord(0, 0.52, 390,
                                              pathlib.Path("research/threshold.md"))
        code, said = self._drive({"return_value": record})
        self.assertEqual(code, 0)
        self.assertIn("390 записалось бы", said)
        self.assertIn("research/threshold.md", said)


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
            (seed_metadata, "stored_mathnet_titles", {}),
            (seed_metadata, "corpus_seed_documents", []),
            (pg_load_citations, "default_zbmath_cache_dir", cache / "zbmath"),
            (pg_load_citations, "default_mathnet_cache_dir", cache / "mathnet"),
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


if __name__ == "__main__":
    unittest.main()
