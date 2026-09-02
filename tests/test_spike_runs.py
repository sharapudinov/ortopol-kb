"""citations/spike_runs.py: the order two spike modes write in, and nothing
else.

Split out of test_citations_cli.py (kb/CLAUDE.md FILE_SIZE) along the seam
the module itself now keeps: what gets WRITTEN, in which order, is asserted
here; what the user is told about it and which exit code the process takes
is the CLI's and is asserted there. The two used to be one module and one
test file, so a change to a printed line and a change to the write order
were the same edit.

The live class at the bottom keys everything under a `test:` spike, deletes
what it wrote and rewinds the sequence -- the one property no stub carries
is what DELETE ... CASCADE does to a run row's data rows, which is the whole
reason update_run_fields() exists.
"""
from __future__ import annotations

import inspect
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

from citations import hub_cache, hub_report, http_cache, spike_runs, threshold_store
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql, scalar

ENV = {"PGHOST": "test"}


def _cache(directory: str):
    return http_cache.ReadOnlyCache(Path(directory))


class HubReportWriteOrderTests(unittest.TestCase):
    """The aggregation pass is a full scan of the largest table in the
    schema, twice expanding evidence->'records'. It ran twice because the
    second upsert_run() -- there only to stamp verify_query with numbers the
    first pass produced -- deleted the run row and cascaded away its data.
    """

    ROWS = [["cites", "384", "9000", "1200", "3", "15000"]]

    class _Writer(spike_runs.DryRunMeasurementsWriter):
        """A writing writer with no database: the run id and the read-back
        a populated table would have answered."""

        dry = False
        rows: list = []

        def upsert_run(self, spike, fields):
            super().upsert_run(spike, fields)
            return 7

        def hub_stats(self, run_id, hub_cap):
            super().hub_stats(run_id, hub_cap)
            return spike_runs.HubStats(self.rows, [])

    def _run(self):
        writer = self._Writer()
        writer.rows = self.ROWS
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]):
            record = spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)
        return record, writer

    def test_the_aggregation_pass_runs_once(self):
        record, writer = self._run()
        self.assertEqual(record.run_id, 7)
        self.assertEqual([name for name, _payload in writer.calls].count("populate"), 1)
        self.assertEqual([name for name, _payload in writer.calls].count("upsert_run"), 1)

    def test_verify_query_is_stamped_by_an_in_place_update(self):
        _record, writer = self._run()
        order = [name for name, _payload in writer.calls]
        self.assertEqual(order.index("update_run_fields") > order.index("populate"), True)
        self.assertIn("update_run_fields", order)

    def test_the_stamped_verify_query_names_the_measured_numbers(self):
        seen = {}
        writer = self._Writer()
        writer.rows = self.ROWS
        writer.update_run_fields = lambda spike, fields: seen.update(fields)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]):
            spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)
        self.assertIn("cites 384 узлов", seen["verify_query"])

    def test_a_dry_writer_reports_the_counts_and_writes_nothing(self):
        """Every method is still called -- that is what makes .calls the
        mode's answer to "what would you have written" -- and every one of
        them is the writer that writes nothing.
        """
        writer = spike_runs.DryRunMeasurementsWriter()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]):
            record = spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)
        self.assertEqual(record.counts, [51652])
        self.assertEqual((record.run_id, record.rows), (0, []))
        self.assertEqual(record.report, Path(tmp) / hub_report.REPORT_PATH)
        self.assertEqual([name for name, _payload in writer.calls],
                         ["ddl", "upsert_run", "populate", "hub_stats",
                          "update_run_fields", "report"])


class CapIsTheRunsNotTheModulesTests(unittest.TestCase):
    """--hub-cap is what the whole measurement is about, so it has to reach
    the numbers and not only the prose. It reached report()'s closing
    paragraph while the column above it, the table header and the stored
    verify_query all said 1000 -- an artifact contradicting itself in the
    one module whose product is the justification for the cap.
    """

    ROWS = [["cites", "384", "9000", "1200", "3", "15000"]]

    def test_the_stored_verify_query_counts_by_the_runs_cap(self):
        stored = hub_report.run_fields([51652], self.ROWS, 500)["verify_query"]
        self.assertIn("cited_by_count > 500", stored)
        self.assertNotIn("cited_by_count > 1000", stored)

    def test_two_caps_do_not_record_the_same_contract(self):
        self.assertNotEqual(hub_report.run_fields([51652], self.ROWS, 500)["verify_query"],
                            hub_report.run_fields([51652], self.ROWS, 1000)["verify_query"])

    def test_the_command_that_reproduces_it_names_the_cap_too(self):
        self.assertIn("--hub-cap 500", hub_report.run_fields([51652], self.ROWS, 500)["reproduce"])

    def test_the_table_header_names_the_cap_its_column_was_measured_at(self):
        rendered = hub_report.report([51652], self.ROWS, [], 93, 500)
        self.assertIn("за капом 500", rendered)
        self.assertNotIn("за капом 1000", rendered)

    def test_the_aggregation_asks_for_the_cap_as_a_variable(self):
        with mock.patch.object(hub_report, "run_sql",
                                return_value=mock.Mock(stdout="")) as run_sql_mock:
            hub_report.stats(ENV, 93, 500)
        self.assertIn(":cap", run_sql_mock.call_args[0][1])
        self.assertEqual(run_sql_mock.call_args.kwargs["variables"]["cap"], "500")


class RefusesAnEmptyMeasurementTests(unittest.TestCase):
    """batch_counts() answers [] for an empty or foreign cache instead of
    raising, and the two modes read DIFFERENT caches in practice, so "the
    cache the crawl never wrote" is a reachable input. Refusing is what
    keeps a run row whose verify_query "confirms" unobserved numbers out of
    measurements -- and it is a DOMAIN error here, not an exit code.
    """

    def _writer(self, dry: bool):
        writer = spike_runs.DryRunMeasurementsWriter()
        writer.dry = dry  # the writing branch, without a database
        return writer

    def test_a_cache_with_no_cites_batches_is_refused_and_writes_nothing(self):
        for dry in (False, True):
            writer = self._writer(dry)
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(hub_cache, "batch_counts", return_value=[]):
                with self.assertRaises(spike_runs.NothingToMeasure):
                    spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)
            self.assertEqual(writer.calls, [], f"dry={dry}")

    def test_a_calibration_with_no_candidates_is_refused_and_writes_nothing(self):
        writer = spike_runs.DryRunMeasurementsWriter()
        snowball = mock.Mock(calibrate=mock.Mock(return_value=[]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(spike_runs.NothingToMeasure):
                spike_runs.record_calibration(snowball, Path(tmp), writer)
        self.assertEqual(writer.calls, [])


class TheSeamSaysNothingTests(unittest.TestCase):
    """Neither mode prints: the writer seam reports by RETURNING what it
    wrote. A module that also owned the exit codes and the report text was
    two responsibilities, and the CLI kept the other two modes' bodies.
    """

    def test_recording_a_hub_measurement_prints_nothing(self):
        writer = spike_runs.DryRunMeasurementsWriter()
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]), \
             redirect_stdout(out):
            spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)
        self.assertEqual(out.getvalue(), "")

    def test_recording_a_calibration_prints_nothing_and_names_its_report(self):
        writer = spike_runs.DryRunMeasurementsWriter()
        rows = [{"candidate_key": "W1", "score": 0.5, "relation": "cites",
                 "title": "Чебышёв", "year": 1997, "has_abstract": True}]
        snowball = mock.Mock(calibrate=mock.Mock(return_value=rows), candidate_refs={})
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(out):
            record = spike_runs.record_calibration(snowball, Path(tmp), writer)
            self.assertEqual(record.report, Path(tmp) / spike_runs.calibration.REPORT_PATH)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(record.written, 1)


class BothWritersSeeTheSameSequenceTests(unittest.TestCase):
    """One procedure, one call sequence, two implementations behind it.

    record_calibration() never branched: it calls every method and lets the
    dry writer absorb them. record_hub_report() short-circuited on
    writer.dry instead, so the dry path of that mode went round the seam
    entirely -- .calls stayed empty (the CLI had to hand-write the caveat),
    and "a writer method exists => both modes go through it" stopped being
    something a reader could rely on. The one genuinely mode-dependent step
    -- reading statistics back out of a table the dry writer never
    populated -- is hub_stats(), i.e. the writer's own answer.
    """

    class _Recorder:
        """The live implementation's contract with the database taken out:
        what is recorded is which methods the procedure called, in order.
        """

        def __init__(self, dry: bool, stats):
            self.dry, self._stats, self.calls = dry, stats, []

        def __getattr__(self, name):
            def method(*args, **kwargs):
                self.calls.append(name)
                if name == "upsert_run":
                    return 7
                if name == "hub_stats":
                    return self._stats
                if name == "threshold_rows":
                    return len(list(args[1]))
                return None
            return method

    ROWS = [["cites", "384", "9000", "1200", "3", "15000"]]

    def _hub(self, writer):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(hub_cache, "batch_counts", return_value=[51652]):
            spike_runs.record_hub_report(_cache(tmp), Path(tmp), writer, 1000)

    def _calibration(self, writer):
        rows = [{"candidate_key": "W1", "score": 0.5, "relation": "cites",
                 "title": "Чебышёв", "year": 1997, "has_abstract": True}]
        snowball = mock.Mock(calibrate=mock.Mock(return_value=rows), candidate_refs={})
        with tempfile.TemporaryDirectory() as tmp:
            spike_runs.record_calibration(snowball, Path(tmp), writer)

    def _sequences(self, drive) -> list[list[str]]:
        seen = []
        for dry, stats in ((False, spike_runs.HubStats(self.ROWS, [])),
                           (True, spike_runs.HubStats([], []))):
            writer = self._Recorder(dry, stats)
            drive(writer)
            seen.append(writer.calls)
        return seen

    def test_the_hub_measurement_calls_the_same_methods_in_both_modes(self):
        live, dry = self._sequences(self._hub)
        self.assertEqual(live, ["ddl", "upsert_run", "populate", "hub_stats",
                                "update_run_fields", "report"])
        self.assertEqual(dry, live)

    def test_the_calibration_does_too(self):
        """The mode that never branched, asserted the same way, so the two
        are held to one rule rather than to each other's history."""
        live, dry = self._sequences(self._calibration)
        self.assertEqual(live, ["ddl", "upsert_run", "threshold_rows", "report"])
        self.assertEqual(dry, live)

    def test_the_two_shipped_writers_answer_the_same_contract(self):
        """Both directions: a method on one and not the other is a mode
        that reaches the database through an AttributeError, or a dry mode
        quietly skipping a step the live one takes.
        """
        live = {name for name in dir(spike_runs.MeasurementsWriter)
                if not name.startswith("_")}
        dry = {name for name in dir(spike_runs.DryRunMeasurementsWriter)
               if not name.startswith("_") and name != "calls"}
        self.assertEqual(live, dry)
        for name in sorted(live - {"dry"}):
            with self.subTest(method=name):
                self.assertEqual(
                    inspect.signature(getattr(spike_runs.MeasurementsWriter, name)),
                    inspect.signature(getattr(spike_runs.DryRunMeasurementsWriter, name)))

    def test_the_dry_read_back_is_empty_rather_than_absent(self):
        stats = spike_runs.DryRunMeasurementsWriter().hub_stats(0, 1000)
        self.assertEqual((stats.rows, stats.worst), ([], []))


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
             "reproduce": "python3 -m unittest test_spike_runs",
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

    def test_the_aggregation_counts_over_the_callers_cap(self):
        """The "over cap" column is the measurement, not a fixed threshold:
        the same rows aggregate differently at two caps, and a literal in
        the statement made the column disagree with the report around it.
        """
        run_sql(
            self.env,
            "INSERT INTO measurements.citation_hub_expansion "
            "(run_id, work_key, relation, cited_by_count, n_references) "
            "VALUES (:run, 'test:citations-cli:W2', 'cites', 700, 5);",
            variables={"run": str(self.run_id)},
        )
        over = {cap: hub_report.stats(self.env, self.run_id, cap)[0][4]
                for cap in (500, 1000)}
        self.assertEqual(over, {500: "1", 1000: "0"})

    def test_a_second_upsert_would_have_taken_them(self):
        """Why the update exists at all -- the behaviour it replaces."""
        threshold_store.upsert_run(
            self.env, self.SPIKE,
            {"question": "test fixture", "arbiter": "test fixture",
             "reproduce": "python3 -m unittest test_spike_runs"})
        self.assertEqual(self._rows(), 0)


if __name__ == "__main__":
    unittest.main()
