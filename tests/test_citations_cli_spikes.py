"""What the two spike modes SAY, and with which exit code.

citations/spike_cli.py's half of the seam spike_runs.py declares: the write
ORDER is test_spike_runs.py's, the loader's own promises (a dry run touching
nothing, the mode exclusivity, the refusals main() makes before dispatch)
are test_citations_cli.py's, and asserted here is only what a user is told
about a measurement and what the process exits with.

Driven through main(), not by calling the mode directly: which object the
mode is handed -- writer, cache, data root -- is part of what is under test,
and only main() decides that.
"""
from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest import mock

import _pathfix  # noqa: F401
import paths
from _loader_harness import MainHarness

import pg_load_citations
from citations import spike_cli, spike_runs


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

    def _main(self, argv: list[str]) -> tuple[int, str, MainHarness]:
        out = io.StringIO()
        with ExitStack() as stack:
            harness = MainHarness(stack)
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

    def test_the_report_root_is_the_one_paths_resolves(self):
        """paths.py owns "where the data tree is", and it locates it by a
        marker (theory/iis/) rather than by position. Inverting
        default_corpus_dir() with .parent re-derives the same fact from an
        assumption about the corpus directory's place, and would put the
        reports one directory off the day that assumption changes.
        """
        record = spike_runs.HubRecord([1], 7, [], pathlib.Path("research/r.md"))
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            MainHarness(stack)
            report = stack.enter_context(mock.patch.object(
                spike_cli, "record_hub_report", return_value=record))
            stack.enter_context(mock.patch.object(
                pg_load_citations, "default_corpus_dir",
                return_value=pathlib.Path(cache) / "elsewhere" / "corpus"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = pg_load_citations.main(["--hub-report", "--cache-dir", cache])
        self.assertEqual(code, 0)
        self.assertEqual(report.call_args[0][1], paths.data_root())

    def test_a_real_run_prints_the_run_it_wrote(self):
        """Also the complement to the dry-run guard above: a run that is not
        dry DOES apply the schema, so the guard cannot pass by never
        applying it at all.
        """
        record = spike_runs.HubRecord(
            [18904], 7, [["cites", "384", "9000", "1200", "3", "15000"]],
            pathlib.Path("research/citation-hub/report.md"))
        with tempfile.TemporaryDirectory() as cache, ExitStack() as stack:
            harness = MainHarness(stack)
            stack.enter_context(mock.patch.object(
                spike_cli, "record_hub_report", return_value=record))
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
            _harness = MainHarness(stack)
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
                spike_cli, "record_calibration", **record_side_effect))
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


if __name__ == "__main__":
    unittest.main()
