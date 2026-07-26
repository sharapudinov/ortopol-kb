"""Guard pinning smoke_checks.VECTOR_PROBE_RANK_TOLERANCE and
VECTOR_PROBE_DISTANCE_TOLERANCE to the measurements documented in their own
comments (49ab5b3e).

Every other test touching these constants expresses its input relative to
the constant itself (TOLERANCE/2, 10*TOLERANCE, probe["rank"]+TOLERANCE),
so all of them stay green no matter how far a tolerance is widened --
widening a tolerance to make a test pass is an explicitly forbidden pattern
in this project (see CLAUDE.md), and this is the one test that would
actually notice.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import smoke_checks

# python3 drift_probe.py --pgenv corpus/.pgenv \
#   --ollama-url http://127.0.0.1:5471/api/embed -n 10
#   -> max rank shift=0, max |Δdistance|=0.000000 (2026-07-26)
#
# NOT a drift measurement: re-embedding the identical string with the
# identical model in the SAME process, on the SAME host, under CPU-only
# inference is deterministic by construction -- 0 is the only value this
# command can ever produce, whether or not the tolerances below actually
# hold. It contributes no coverage evidence and is not part of
# _WORST_OBSERVED_RANK_SHIFT below; kept only as a determinism control
# (see DeterminismControlTests) -- a nonzero result here would mean this
# host's CPU-only inference stopped being reproducible, which is a
# different, more basic problem than anything a tolerance can bound.
_SAME_PROCESS_DETERMINISM_CONTROL_RANK_SHIFT = 0

# python3 smoke_test.py --artifact corpus/deploy/kb-20260725.tar.zst \
#   --live-pgenv corpus/.pgenv --measure-drift 10
#   -> max rank shift=0, max |Δdistance|=0.000657 (2026-07-26)
#
# This is the actual guarded comparison: a live-built manifest reference
# vs. a freshly deployed kb-smoke stack's own ollama, in a SEPARATE
# process -- the only run that can show nonzero drift, and therefore the
# only evidence these tolerances are sized against.
_WORST_OBSERVED_RANK_SHIFT = 0
_WORST_OBSERVED_DISTANCE_DELTA = 0.000657


class ToleranceGuardTests(unittest.TestCase):
    def test_distance_tolerance_is_pinned_to_the_documented_constant(self):
        # A change here must be a deliberate re-measurement (re-run the
        # drift_probe.py command above, update the comment AND this pin in
        # the same commit), never an incidental widening to unblock an
        # unrelated failure.
        self.assertEqual(smoke_checks.VECTOR_PROBE_DISTANCE_TOLERANCE, 1e-3)

    def test_distance_tolerance_still_exceeds_the_worst_observed_delta(self):
        self.assertLess(
            _WORST_OBSERVED_DISTANCE_DELTA,
            smoke_checks.VECTOR_PROBE_DISTANCE_TOLERANCE,
            "VECTOR_PROBE_DISTANCE_TOLERANCE no longer covers the worst "
            f"cross-process drift measured on 2026-07-26 "
            f"({_WORST_OBSERVED_DISTANCE_DELTA}) -- re-measure with "
            "drift_probe.py (see smoke_checks.py's comment on this constant) "
            "before touching the constant, do not just widen it.",
        )

    def test_rank_tolerance_is_pinned_to_the_documented_constant(self):
        self.assertEqual(smoke_checks.VECTOR_PROBE_RANK_TOLERANCE, 1)

    def test_rank_tolerance_still_covers_the_worst_observed_shift(self):
        # Only the cross-process measurement counts as evidence here -- see
        # _WORST_OBSERVED_RANK_SHIFT's own comment on why the same-process
        # control is excluded rather than folded into a max().
        self.assertGreaterEqual(
            smoke_checks.VECTOR_PROBE_RANK_TOLERANCE, _WORST_OBSERVED_RANK_SHIFT,
            "VECTOR_PROBE_RANK_TOLERANCE no longer covers the worst rank shift "
            f"measured on 2026-07-26 ({_WORST_OBSERVED_RANK_SHIFT}) -- re-measure "
            "with drift_probe.py before touching the constant, do not just widen it.",
        )


class DeterminismControlTests(unittest.TestCase):
    """Not a tolerance guard: pins the same-process re-embed to exactly 0,
    the only value it can legitimately take (see the constant's own
    comment). A failure here means this host's CPU-only inference stopped
    being bit-for-bit reproducible -- re-run the drift_probe.py command
    quoted above and investigate the environment before touching either
    tolerance, since neither one was ever meant to absorb that kind of
    change.
    """

    def test_same_process_reembed_is_exactly_deterministic(self):
        self.assertEqual(_SAME_PROCESS_DETERMINISM_CONTROL_RANK_SHIFT, 0)


if __name__ == "__main__":
    unittest.main()
