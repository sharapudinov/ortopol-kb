"""check_vector() and the two tolerance constants its (ok, detail) verdict
depends on, split out of smoke_checks.py (module size): this is the one
check whose contract carries a whole page of numeric justification for why
VECTOR_PROBE_RANK_TOLERANCE and VECTOR_PROBE_DISTANCE_TOLERANCE are what
they are, which belongs beside the code it justifies rather than diluting
smoke_checks.py's simpler manifest-comparison predicates. Re-exported from
smoke_checks.py so smoke_test.py's `checks.check_vector(...)` etc. keep
working unchanged.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import pg_rank_probe  # noqa: E402
import pg_search  # noqa: E402
from manifest_contract import Key  # noqa: E402

# Reproducible via drift_probe.py (see that module), not by inspection:
#   same-process determinism control, NOT drift evidence -- re-embedding
#   the identical string with the identical model in the same process on
#   the same CPU-only host is deterministic by construction, so 0 is the
#   only value this command can produce and it contributes nothing to the
#   tolerances below (see test_vector_probe_tolerance_guard.py's own
#   DeterminismControlTests) --
#     python3 drift_probe.py --pgenv corpus/.pgenv \
#       --ollama-url http://127.0.0.1:5471/api/embed -n 10
#     -> max rank shift=0, max |Δdistance|=0.000000 (2026-07-26)
#   the actual guarded comparison, and the only one either tolerance below
#   is measured against (live-built manifest vs. a freshly deployed
#   kb-smoke stack, both on this CPU-only host, but in SEPARATE processes) --
#     python3 smoke_test.py --artifact corpus/deploy/kb-20260725.tar.zst \
#       --live-pgenv corpus/.pgenv --measure-drift 10
#     -> max rank shift=0, max |Δdistance|=0.000657 (2026-07-26)
#
# Before 2baf1ba4 added a total, data-derived tiebreak to _RANKED_PAGES_CTE
# (pg_rank_probe.py), the SAME cross-process comparison had shown rank
# shifting from 1 to 2 every time across 3 repeated kb-smoke runs
# (2026-07-25) -- indistinguishable, from the outside, from a genuine
# nearest-neighbour swap. It was in fact the executor's unstable tie-break
# among rows the (then undifferentiated) ORDER BY saw as equal-distance:
# the measurement above, taken immediately after that fix on the same
# corpus and the same probe query, reproduces zero shift over 10 repeats.
# Kept at 1 rather than 0: this host has no GPU to also measure the
# CPU-vs-GPU float-summation-order drift the module docstring calls out,
# and a tolerance of exactly the worst-observed value (0) would leave no
# margin at all for that still-untested dimension.
#
# manifest["vector_probe"]["runner_up_distance"] (rnk = 2 from the SAME
# ranked CTE, see pg_rank_probe.runner_up_distance) makes the actual margin
# for THIS probe checkable, and it is not "margin >> tolerance": measured
# directly against the live corpus (2026-07-26), the runner-up's distance
# equals the reference's own distance bit-for-bit (margin = 0.0). This is
# not noise -- the runner-up is a DIFFERENT document at a DIFFERENT page
# whose body is the same near-empty scanned-page boilerplate ("22", the
# page number and nothing else) as the reference page, so the two pages
# embed identically; a live count found 250 such duplicate-embedding groups
# across the corpus (docstring in pg_rank_probe.py already predicted this
# class of tie). A tolerance of exactly 0 would therefore fail this probe
# the moment cross-process float noise (bounded by VECTOR_PROBE_DISTANCE_
# TOLERANCE below) breaks the tie the "wrong" way -- which the deterministic
# tiebreak (document_id, page_number) makes a coin flip between these two
# specific rows, not a sign of degraded search quality. Tolerance = 1 is
# therefore justified mechanistically, not just empirically: it is exactly
# enough to absorb one exact tie resolving either way, while a shift of 2+
# still fails, since nothing else in the top ranks is anywhere this close --
# rank 3 in the same probe (queried directly against the same ranked CTE,
# 2026-07-26: distance 0.450825 vs. the reference's 0.447759) sits ~0.003066
# away, i.e. about 3x VECTOR_PROBE_DISTANCE_TOLERANCE (1e-3) below, not the
# "three orders of magnitude" (1000x) an earlier version of this comment
# wrongly claimed -- 0.003 vs 0.001 is a factor of three, not a factor of a
# thousand. The corrected number is a real but modest margin, not an
# overwhelming one: even charging the full distance tolerance twice against
# it (once inflating the reference's own distance, once shrinking rank 3's,
# the worst case a single cross-process drift measurement could produce)
# leaves ~0.00107 of headroom before rank 3 could be mistaken for the tie
# tolerance = 1 is meant to absorb -- against the actually measured
# cross-process drift (0.000657, see VECTOR_PROBE_DISTANCE_TOLERANCE below)
# the same gap is a ~4.7x margin. The mechanistic argument for tolerance = 1
# therefore still holds, just far less comfortably than the wrong number
# implied; this is what should be re-measured (not merely re-typed) if the
# corpus, the query, or the model ever change.
VECTOR_PROBE_RANK_TOLERANCE = 1

# Companion tolerance for the actual cosine distance (not just its ordinal
# rank) -- same two measurements as above. max |Δdistance| reproduced
# identically to the pre-tiebreak-fix observation (0.000657, deterministic
# CPU inference on this host): the tiebreak fix changed which row wins a
# tie, not the underlying cross-process embedding drift this tolerance
# bounds. 1e-3 is that worst-observed value with a ~1.5x margin, padding
# specifically for the CPU-vs-GPU drift this single-host measurement cannot
# exercise.
VECTOR_PROBE_DISTANCE_TOLERANCE = 1e-3


def check_vector(env: dict, manifest: dict, ollama_url: str) -> tuple[bool, str]:
    probe = manifest[Key.VECTOR_PROBE]

    # manifest_probe.gather_manifest() refuses to record a probe/page pair
    # with any stemmed lexeme overlap, and always stores [] as a result --
    # but --artifact-dir explicitly lets this script point at a previously
    # built artifact whose build-time guard may predate that refusal (an
    # older manifest schema, or one built before the check existed). A
    # non-empty value here means the invariant this whole check depends on
    # ("a match proves vector search, not word overlap") was never actually
    # true for this artifact -- fail fast rather than print a claim the
    # manifest itself contradicts.
    overlap = probe.get(Key.TOKEN_OVERLAP)
    if overlap:
        return False, (
            f"probe query shares lexeme(s) {overlap} with the reference page -- "
            "this artifact's vector probe cannot distinguish vector search from "
            "plain word overlap (rebuild the package)"
        )

    try:
        vec_json = pg_search.embed_query(probe[Key.QUERY], env, ollama_url=ollama_url)
    except (ValueError, RuntimeError) as exc:
        # ValueError: dims mismatch, raised deliberately by embed_query.
        # RuntimeError: run_sql's internal model/dims lookup failed (any
        # psql failure) -- embed_query only guards the HTTP call with its
        # own try/except, so a psql hiccup here would otherwise propagate
        # uncaught and abort smoke_test.py's whole run before teardown.
        return False, str(exc)
    if vec_json is None:
        return False, "embedding service unreachable"

    try:
        hit = pg_rank_probe.page_rank(env, vec_json, probe[Key.DOCUMENT_ID], probe[Key.PAGE_NUMBER])
    except RuntimeError as exc:
        # Same rationale as the embed_query guard above: page_rank's own
        # run_sql call can fail on a transient psql hiccup, and this
        # function's whole contract (return (ok, detail), never raise) exists
        # so smoke_test.py keeps running every remaining check afterwards.
        return False, str(exc)
    if hit is None:
        return False, f"{probe[Key.DOCUMENT_ID]} p.{probe[Key.PAGE_NUMBER]}: no longer embedded"

    rank_limit = probe[Key.RANK] + VECTOR_PROBE_RANK_TOLERANCE
    rank_ok = hit["rank"] <= rank_limit
    distance_delta = abs(hit["distance"] - probe[Key.DISTANCE])
    distance_ok = distance_delta <= VECTOR_PROBE_DISTANCE_TOLERANCE

    # runner_up_distance (manifest_probe.gather_manifest, rnk = 2 from the
    # same ranked CTE nearest_page/page_rank use) is the quantity
    # VECTOR_PROBE_RANK_TOLERANCE's own docstring justifies tolerance = 1
    # against: either an exact tie (margin == 0, absorbed by design) or a
    # clear separation (margin > VECTOR_PROBE_DISTANCE_TOLERANCE, too far
    # for cross-process float noise to close). margin_ok asserts exactly
    # that -- an ambiguous margin strictly between the two (a near-tie that
    # is NOT provably a tie) is neither case the tolerance's own
    # justification covers, so it fails loudly here instead of silently
    # passing as if it were the measured tie. This is the condition the
    # docstring above has always described in prose; before this check it
    # was printed but never enforced, so a rebuild whose margin drifted
    # into that band would have passed unnoticed.
    runner_up = probe.get(Key.RUNNER_UP_DISTANCE)
    margin_detail = ""
    margin_ok = True
    if runner_up is not None:
        margin = runner_up - probe[Key.DISTANCE]
        margin_detail = f", margin to runner-up={margin:.6f}"
        if 0 < margin <= VECTOR_PROBE_DISTANCE_TOLERANCE:
            margin_ok = False
            margin_detail += (
                f" -- AMBIGUOUS: neither an exact tie nor separated past "
                f"tolerance={VECTOR_PROBE_DISTANCE_TOLERANCE} (rebuild the "
                "package or re-measure VECTOR_PROBE_RANK_TOLERANCE)"
            )

    ok = rank_ok and distance_ok and margin_ok

    return ok, (
        f"{probe[Key.DOCUMENT_ID]} p.{probe[Key.PAGE_NUMBER]}: rank={hit['rank']} "
        f"distance={hit['distance']:.6f} (manifest rank={probe[Key.RANK]}, "
        f"tolerance={VECTOR_PROBE_RANK_TOLERANCE}; manifest distance={probe[Key.DISTANCE]:.6f}, "
        f"delta={distance_delta:.6f}, tolerance={VECTOR_PROBE_DISTANCE_TOLERANCE}{margin_detail})"
    )
