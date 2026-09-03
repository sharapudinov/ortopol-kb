"""The threshold's polarity, asserted at the boundary and nowhere else.

`score == tau` is the only value that can tell `>= tau` from `> tau`, so
every production reader of the threshold is exercised at exactly that value:
the keep/drop decision and the journal row it writes (citations/crawl.py),
whether a kept candidate holds on to its embedding (citations/candidates.py),
and the cost table a recorded tau verdict rests on
(citations/calibration.py). All three ask citations/scoring.keeps(), and the
AST scan at the bottom is what keeps a fourth spelling from appearing beside
them: three hand-written comparisons already existed, and moving the
boundary at one of them would have desynchronised the journal, the retained
vectors and the calibration's own prediction of what a crawl keeps, with
nothing failing.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, unit, work
from citations import calibration
from citations import scoring
from citations import candidates as candidates_mod
from citations.crawl import Snowball
from citations.dry_store import DryRunWriter
from citations.scoring import keeps, split_by_threshold

CITATIONS_DIR = Path(scoring.__file__).resolve().parent
# Where the comparison is allowed to be written out: the one module that
# declares it.
PREDICATE_HOME = "scoring.py"
TAU = 0.5


class PredicateTests(unittest.TestCase):
    def test_the_boundary_belongs_to_the_kept_side(self):
        self.assertTrue(keeps(TAU, TAU))
        self.assertTrue(keeps(TAU + 1e-12, TAU))
        self.assertFalse(keeps(TAU - 1e-12, TAU))

    def test_split_is_the_predicate_applied_twice(self):
        kept, dropped = split_by_threshold(
            {"at": TAU, "above": TAU + 0.1, "below": TAU - 1e-12}, TAU)
        self.assertEqual(kept, ["above", "at"])
        self.assertEqual(dropped, ["below"])


class ProductionSitesTests(unittest.TestCase):
    """The three readers, each at score == tau."""

    def test_a_candidate_scoring_exactly_tau_keeps_its_vector(self):
        holders = [candidates_mod.scoring_fields(work("W_AT", title="At the boundary"))]
        with mock.patch.object(candidates_mod, "cosine_unit", return_value=TAU):
            scored = candidates_mod.scores_of(
                lambda hs: [(h, unit(0)) for h in hs], unit(0), TAU, holders)
        score, vector = scored["W_AT"]
        self.assertEqual(score, TAU)
        self.assertIsNotNone(vector, "a kept candidate that lost its vector")

    def test_a_candidate_scoring_exactly_tau_is_kept_by_the_crawl(self):
        """tau = 1.0 against axis-aligned vectors: the cosine IS the
        threshold, exactly, with no floating-point margin to hide behind."""
        writer = DryRunWriter()
        client = FakeClient(
            [work("W_SEED_A", title="Seed Chebyshev")],
            citers={"W_SEED_A": [work("W_AT", title="At Chebyshev", refs=["W_SEED_A"])]},
        )
        embedder = PlannedEmbedder({"Seed": unit(0), "At": unit(0)})
        snowball = Snowball(client, embedder, writer, tau=1.0, crawl_id="c",
                            log=lambda *_: None)
        snowball.seed(["doc_a"], {"doc_a": "W_SEED_A"})

        self.assertEqual(snowball.expand(["W_SEED_A"], 1), ["W_AT"])
        kept = [s for s in writer.steps_seen if s["action"] == "keep"]
        self.assertEqual([s["candidate_key"] for s in kept], ["W_AT"])
        self.assertEqual((kept[0]["score"], kept[0]["tau"]), (1.0, 1.0))
        self.assertEqual([s for s in writer.steps_seen if s["action"] == "drop"], [])

    def test_the_cost_table_counts_a_row_scoring_exactly_tau_as_kept(self):
        rows = [{"score": TAU, "candidate_key": "W_AT"},
                {"score": TAU - 1e-12, "candidate_key": "W_BELOW"}]
        refs = {"W_AT": {"W_R1", "W_R2"}, "W_BELOW": {"W_R3"}}
        table = calibration.cost_table(rows, refs, taus=(TAU,))
        self.assertEqual(len(table), 3, table)
        # | τ | kept | distinct refs | ≈ requests |
        fields = [cell.strip() for cell in table[2].strip("|").split("|")]
        self.assertEqual(fields[1], "1")
        self.assertEqual(fields[2], "2")


class OnePolarityTests(unittest.TestCase):
    """No module under citations/ compares a score with a threshold by hand.

    An ordering comparison naming `tau` on either side is the shape all
    three production sites had; scoring.py is where the answer is declared,
    everywhere else it is called.
    """

    ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    @staticmethod
    def _names_tau(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "tau"
        if isinstance(node, ast.Attribute):
            return node.attr == "tau"
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return node.slice.value == "tau"
        return False

    def _hand_written(self, source: str) -> list[int]:
        found = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, self.ORDERING) for op in node.ops):
                continue
            if self._names_tau(node.left) or any(self._names_tau(c) for c in node.comparators):
                found.append(node.lineno)
        return found

    def test_the_comparison_lives_in_one_module(self):
        for path in sorted(CITATIONS_DIR.glob("*.py")):
            if path.name == PREDICATE_HOME:
                continue
            self.assertEqual(
                self._hand_written(path.read_text(encoding="utf-8")), [],
                f"{path.name}: сравнение со счётом пишется один раз — "
                f"citations/scoring.keeps()",
            )

    def test_the_scan_catches_a_hand_written_comparison(self):
        """Positive control: the shape the three sites had."""
        self.assertEqual(self._hand_written("kept = [r for r in rows if r['score'] >= tau]"),
                         [1])
        self.assertEqual(self._hand_written('if item["score"] < self.tau:\n    pass\n'), [1])
        self.assertEqual(self._hand_written('if row["score"] < row["tau"]:\n    pass\n'), [1])
        self.assertEqual(self._hand_written("keeps(score, tau)"), [])


if __name__ == "__main__":
    unittest.main()
