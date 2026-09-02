"""citations/calibration.py: what the threshold measurement prints, what it
computes, and what it refuses to author.

No network, no database. Two properties matter more than the rendering:
a regenerated report must not destroy the hand-written sections below it
(the same bargain the loaders strike with transcribed pages), and the number
the report suggests must come from the distribution, not from a constant --
including the honest "no boundary in the data" when there is none.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

from citations import calibration, spike_runs


def rows_of(scores, **overrides) -> list[dict]:
    out = []
    for index, score in enumerate(scores):
        row = {"candidate_key": f"W{index}", "depth": 1, "relation": "cites",
               "score": score, "title": f"Title {index}", "year": 2000 + index % 5,
               "has_abstract": index % 2 == 0, "n_references": index}
        row.update(overrides)
        out.append(row)
    return out


class SuggestTauTests(unittest.TestCase):
    """The threshold is read off the data or not offered at all."""

    GAPPED = rows_of([0.30, 0.32, 0.34, 0.36, 0.38, 0.40,
                      0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72])

    def test_the_gap_left_of_the_median_is_where_the_line_falls(self):
        tau = calibration.suggest_tau(self.GAPPED)
        self.assertIsNotNone(tau)
        self.assertGreater(tau, 0.40, "порог ниже последнего кандидата слева")
        self.assertLess(tau, 0.60, "порог выше первого кандидата справа")

    def test_a_distribution_without_a_gap_offers_no_boundary(self):
        dense = rows_of([0.30 + 0.005 * i for i in range(80)])
        self.assertIsNone(calibration.suggest_tau(dense))

    def test_the_gap_nearest_the_median_wins_over_an_earlier_one(self):
        """Two gaps below the median: the boundary is the one the kept
        population actually starts after, not the first hole in the tail."""
        two_gaps = rows_of([0.10, 0.12,
                            0.40, 0.42, 0.44,
                            0.70, 0.72, 0.74, 0.76, 0.78, 0.80])
        tau = calibration.suggest_tau(two_gaps)
        self.assertGreater(tau, 0.44, "взят разрыв в хвосте, а не у границы")
        self.assertLess(tau, 0.70)

    def test_a_single_score_is_no_distribution(self):
        self.assertIsNone(calibration.suggest_tau(rows_of([0.5])))
        self.assertIsNone(calibration.suggest_tau([]))

    def test_the_boundary_line_says_so_when_there_is_none(self):
        said = calibration.boundary_line(None)
        self.assertIn("границы в данных нет", said)

    def test_the_boundary_line_names_the_empty_bin_and_its_middle(self):
        said = calibration.boundary_line(0.4945)
        self.assertIn("0.4845…0.5045", said)
        self.assertIn("τ = 0.4945", said)


class CarriedSectionsTests(unittest.TestCase):
    """A rerun of a tool never destroys work the tool did not produce."""

    PREVIOUS = ("# Отчёт\n\nстарые факты\n\n"
                "## Рекомендация исполнителя (вердикт — за оркестратором)\n\n"
                "**τ = 0.50.** потому что\n\n"
                "## Вердикт (оркестратор, 2026-09-02)\n\nпринято\n")

    def _previous(self, tmp: Path, text: str | None) -> Path:
        path = tmp / "threshold.md"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        return path

    def test_no_previous_report_leaves_the_new_text_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._previous(Path(tmp), None)
            self.assertEqual(calibration.carry_over_sections("# новый\n", path), "# новый\n")

    def test_both_hand_written_sections_survive_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._previous(Path(tmp), self.PREVIOUS)
            merged = calibration.carry_over_sections("# новый\n\nновые факты\n", path)
        self.assertIn("новые факты", merged)
        self.assertNotIn("старые факты", merged, "перенесено больше, чем написано руками")
        self.assertLess(merged.index("## Рекомендация исполнителя"),
                        merged.index("## Вердикт"))
        self.assertIn("**τ = 0.50.** потому что", merged)
        self.assertIn("принято", merged)

    def test_a_verdict_without_a_recommendation_is_carried_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._previous(Path(tmp), "# старый\n\n## Вердикт\n\nпринято\n")
            merged = calibration.carry_over_sections("# новый\n", path)
        self.assertIn("## Вердикт\n\nпринято", merged)

    def test_a_new_text_that_already_has_the_section_is_not_doubled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._previous(Path(tmp), self.PREVIOUS)
            merged = calibration.carry_over_sections(
                "# новый\n\n## Рекомендация исполнителя\n\nсвежая\n\n"
                "## Вердикт\n\nсвежий\n", path)
        self.assertEqual(merged.count("## Вердикт"), 1)
        self.assertNotIn("потому что", merged)

    def test_the_generator_writes_neither_section_itself(self):
        """The number and the prose belong to the run that produced them, so
        the report is the only place they live -- not this module.
        """
        text = calibration.calibration_report(rows_of([0.3, 0.4, 0.62, 0.65]), 0.5)
        for heading in calibration.CARRIED_HEADINGS:
            self.assertNotIn(heading, text)
        source = Path(calibration.__file__).read_text(encoding="utf-8")
        self.assertNotIn("RECOMMENDED_TAU", source)

    def test_a_regeneration_keeps_both_sections_of_a_report_like_the_live_one(self):
        rows = rows_of([0.30, 0.32, 0.60, 0.62, 0.64])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._previous(Path(tmp), self.PREVIOUS)
            merged = calibration.carry_over_sections(
                calibration.calibration_report(rows, calibration.suggest_tau(rows)), path)
        self.assertIn("## Квантили", merged)
        self.assertIn("## Рекомендация исполнителя", merged)
        self.assertIn("## Вердикт", merged)


class ReportRenderingTests(unittest.TestCase):
    ROWS = rows_of([0.30, 0.42, 0.61, 0.75])

    def test_cost_table_counts_distinct_references_not_their_sum(self):
        refs = {"W2": {"A", "B"}, "W3": {"B", "C"}}
        table = calibration.cost_table(self.ROWS, refs, taus=(0.50,))
        self.assertEqual(table[-1], "| 0.50 | 2 | 3 | 3 |")

    def test_title_table_is_ordered_by_score_and_escapes_the_separator(self):
        rows = rows_of([0.2, 0.9])
        rows[0]["title"] = "A | B"
        table = calibration.title_table(rows)
        self.assertTrue(table[2].startswith("| 0.9000 "), table)
        self.assertIn("A / B", table[3])

    def test_the_report_carries_the_measured_numbers(self):
        text = calibration.calibration_report(self.ROWS, 0.5, {})
        self.assertIn("Кандидатов depth-1: 4", text)
        self.assertIn("min 0.3000, max 0.7500", text)
        self.assertIn("## Цена depth-2 при разном τ", text)
        self.assertIn("Десять заголовков вокруг неё (τ = 0.50)", text)

    def test_a_report_with_no_boundary_says_so_and_shows_no_table_around_it(self):
        text = calibration.calibration_report(self.ROWS, None, {})
        self.assertIn("границы в данных нет", text)
        self.assertNotIn("Десять заголовков вокруг", text)

    def test_run_fields_name_the_numbers_the_verify_query_must_return(self):
        fields = calibration.run_fields(self.ROWS)
        self.assertIn("ждать: 4 строк", fields["verify_query"])
        self.assertIn("min 0.3000", fields["verify_query"])
        self.assertIn("max 0.7500", fields["verify_query"])
        self.assertEqual(fields["varied"], ["threshold"])
        self.assertIn("citation_frontier_threshold", fields["rules_out"])


class RecordCalibrationTests(unittest.TestCase):
    """The write order of the measurement, on the same seam hub_report uses:
    the run row first, its data rows next, the report last -- and under
    --dry-run none of the three.
    """

    class _Writer(spike_runs.DryRunMeasurementsWriter):
        dry = False  # exercise the writing branch without a database

        def __init__(self):
            super().__init__()
            self.reports: list[tuple[Path, str]] = []

        def upsert_run(self, spike, fields):
            super().upsert_run(spike, fields)
            return 7

        def report(self, path, text):
            super().report(path, text)
            self.reports.append((path, text))

    def _run(self, scores, writer=None):
        writer = writer or self._Writer()
        snowball = mock.Mock(calibrate=mock.Mock(return_value=rows_of(scores)),
                             candidate_refs={})
        client = mock.Mock(n_requests=3, n_cache_hits=1)
        with tempfile.TemporaryDirectory() as tmp:
            code = spike_runs.record_calibration(snowball, client, Path(tmp), writer)
        return code, writer

    def test_no_candidates_is_a_refusal_not_an_empty_report(self):
        writer = self._Writer()
        with mock.patch("sys.stderr"):
            code, writer = self._run([], writer)
        self.assertEqual(code, 1)
        self.assertEqual(writer.calls, [])

    def test_the_run_row_precedes_its_data_rows_and_the_report(self):
        code, writer = self._run([0.30, 0.32, 0.60, 0.62, 0.64])
        self.assertEqual(code, 0)
        order = [name for name, _payload in writer.calls]
        self.assertEqual(order, ["ddl", "upsert_run", "threshold_rows", "report"])
        self.assertEqual(dict(zip(order, [p for _n, p in writer.calls]))["threshold_rows"], 5)

    def test_the_report_goes_to_the_spike_path_in_the_data_tree(self):
        _code, writer = self._run([0.30, 0.32, 0.60, 0.62])
        path, text = writer.reports[0]
        self.assertTrue(str(path).endswith(calibration.REPORT_PATH), path)
        self.assertIn("Кандидатов depth-1: 4", text)

    def test_a_dry_run_writes_nothing_at_all(self):
        writer = spike_runs.DryRunMeasurementsWriter()
        snowball = mock.Mock(calibrate=mock.Mock(return_value=rows_of([0.3, 0.6])),
                             candidate_refs={})
        with tempfile.TemporaryDirectory() as tmp:
            code = spike_runs.record_calibration(
                snowball, mock.Mock(n_requests=0, n_cache_hits=0), Path(tmp), writer)
            self.assertEqual(list(Path(tmp).rglob("*.md")), [])
        self.assertEqual(code, 0)
        self.assertEqual(writer.upsert_run("x", {}), 0, "под --dry-run номера прогона нет")


if __name__ == "__main__":
    unittest.main()
