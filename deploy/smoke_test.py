#!/usr/bin/env python3
"""Deploy predicate for the artifact: does this package actually deploy,
and does the deployed result answer the way its manifest says it should.

Deploys the compose stack under a throwaway project name, separate ports
(5474/5475, distinct from both the live instance on 5470/5471 and the
package's own defaults on 5472/5473) and separate named volumes, runs
every checkable item from the task's acceptance list against it, tears
the stack down, and confirms the live ortopol-pg instance answers
normally afterwards (only if asked to -- see --live-pgenv). Exit 0 iff
every check passes.

Runs two ways:
  - standalone, from inside an extracted kb-<profile>-<date>.tar.zst with no access
    to this repository: `python3 smoke_test.py` with no arguments. Defaults
    to this script's own directory (where manifest.json/01_dump.sql.gz sit
    beside it, see build_package.py's artifact_bundle) and skips the
    live-instance check (there is no repository to find a live .pgenv in).
  - from a repository checkout, against a not-yet-extracted artifact:
    `python3 smoke_test.py --artifact PATH` (defaults to the newest
    corpus/deploy/kb-<profile>-*.tar.zst, --profile full unless told
    otherwise); add --live-pgenv PATH to also confirm the developer's live
    instance was untouched.

Profile-aware throughout, but never by assumption: what gets checked comes
from the artifact's own manifest (which schemas it carries, which document
its blob probe names, which documents ship stripped), so one script verifies
both packages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import compose_lifecycle as lifecycle
import drift_probe
import profile_checks
import smoke_checks as checks
from dump_integrity import sha256_file
from manifest_keys import Key
from manifest_contract import Profile
from smoke_stack import (
    OLLAMA_PORT,
    PROJECT,
    ArtifactUnavailable,
    artifact_data_dir,
    check_live_instance_intact,
    psql_env,
    smoke_env,
)

# Backstop for the whole up-to-healthy wait, independent of docker-compose's
# own healthcheck retries/start_period (docker-compose.yml documents why the
# healthcheck now probes TCP specifically -- that's what makes "healthy"
# here actually mean "the restored dump is reachable"). Measured end-to-end
# on this machine (kb-smoke run, 2026-07-25): the full corpus (70 docs, 2462
# pages, 54 MB dump) restores and the real server is reachable in ~23s. This
# is roughly a 17x margin over that measurement, not a guess -- room for the
# corpus to grow well past its current size before this needs revisiting.
KB_PG_HEALTHY_TIMEOUT = 400.0

# kb-ollama's own healthcheck (docker-compose.yml: interval 10s, retries 30,
# start_period 20s) budgets ~320s before Docker itself gives up on a first
# boot that has to pull bge-m3 (~1.2 GB) from the network -- the exact same
# slow-network-pull concern KB_PG_HEALTHY_TIMEOUT above is generous about.
# Leaving this on wait_healthy()'s generic 240s default would let the
# Python-side poll give up (and report a false-negative "kb-ollama not
# healthy", short-circuiting the vector/digest checks) before Docker's own
# retries -- and the pull itself -- would have succeeded. Set to the same
# 400s budget as kb-pg for the same reason, not independently re-measured.
KB_OLLAMA_HEALTHY_TIMEOUT = 400.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument("--artifact", type=Path, default=None,
                                 help="packed kb-<profile>-<date>.tar.zst to extract and test")
    artifact_group.add_argument("--artifact-dir", type=Path, default=None,
                                 help="already-extracted artifact directory "
                                      "(self-contained run, no repository needed)")
    parser.add_argument("--live-pgenv", type=Path, default=None,
                         help="the developer's live instance .pgenv, to confirm it "
                              "was untouched by this run; omit for a standalone run")
    parser.add_argument("--profile", choices=Profile.ALL, default=Profile.FULL,
                         help="which profile's artifact to auto-discover under "
                              "corpus/deploy/ when neither --artifact nor --artifact-dir "
                              "is given; ignored otherwise (the artifact's own manifest "
                              "always decides what is checked)")
    parser.add_argument("--measure-drift", type=int, default=0, metavar="N",
                         help="in addition to the normal pass/fail checks, re-embed the "
                              "manifest's vector_probe query N times against the deployed "
                              "kb-ollama and print the max rank shift / max |Δdistance| "
                              "against the manifest reference (see drift_probe.py) -- "
                              "purely diagnostic, never affects the exit code")
    args = parser.parse_args(argv)

    results: list[tuple[str, bool | None, str]] = []

    def _run_checks(extract_dir: Path, pristine: bool) -> None:
        # Closes over `args` and `results` -- this is the orchestration
        # artifact_data_dir()'s own body deliberately does not perform (see
        # smoke_stack.py's module docstring): everything below decides what
        # to DO with a resolved artifact directory, once resolution itself
        # (and any ArtifactUnavailable it can raise) is out of the way.
        print(f"артефакт: {extract_dir}")
        manifest = json.loads((extract_dir / "manifest.json").read_text())

        # Checked immediately after parsing, before any other manifest key
        # is touched: --artifact-dir explicitly supports pointing this
        # script at a previously extracted (possibly older) artifact, and
        # every access below (manifest[Key.DUMP][Key.FILE], [Key.BLOB_PROBE],
        # ...) assumes the CURRENT shape. A mismatch used to surface as an
        # opaque KeyError mid-run instead of a normal FAIL naming the actual
        # versions involved.
        schema_version = manifest.get(Key.SCHEMA_VERSION)
        schema_ok = schema_version == checks.MANIFEST_SCHEMA_VERSION
        results.append((
            "manifest schema_version совпадает", schema_ok,
            f"manifest={schema_version!r}, ожидается {checks.MANIFEST_SCHEMA_VERSION}"
            + ("" if schema_ok else " -- rebuild the package or use a matching checkout"),
        ))

        if schema_ok:
            # Verify the extracted dump against the manifest BEFORE letting
            # Postgres load it -- a corrupted/truncated/tampered dump should
            # be caught here, not discovered as a mysteriously-wrong row
            # count three checks later.
            dump = manifest[Key.DUMP]
            dump_path = extract_dir / dump[Key.FILE]
            dump_ok = (
                dump_path.is_file()
                and dump_path.stat().st_size == dump[Key.BYTES]
                and sha256_file(dump_path) == dump[Key.SHA256]
            )
            dump_detail = (
                f"{dump_path.name}: {dump_path.stat().st_size} bytes (manifest {dump[Key.BYTES]})"
                if dump_path.is_file() else f"{dump_path.name}: missing"
            )
            results.append(("дамп sha256 совпадает с манифестом", dump_ok, dump_detail))
            results.append(("файлы манифеста совпадают с распаковкой",
                             *checks.check_bundled_files(extract_dir, manifest, pristine=pristine)))

            # Static, before anything is deployed: the profile's content
            # invariants are a property of the artifact, so they are checked
            # against its bytes and hold (or fail) whether or not Docker is
            # available at all. Also runnable on its own --
            # `python3 profile_checks.py --artifact-dir DIR`.
            if dump_ok:
                results.extend(profile_checks.run_checks(extract_dir))
            else:
                results.append(("профиль: содержимое дампа = манифест", False,
                                 "дамп не сошёлся с манифестом, статические проверки пропущены"))

            # The extracted artifact's OWN docker-compose.yml, not this
            # checkout's copy -- their relative bind mounts (init/, the
            # entrypoint script) resolve next to whichever compose file is
            # passed, so a --artifact/--artifact-dir run exercises exactly
            # the files this package shipped, the same ones
            # check_bundled_files just verified above, instead of silently
            # substituting the repo's.
            compose_file = extract_dir / "docker-compose.yml"
            bge_m3_digest = manifest.get(Key.EMBEDDING_MODEL, {}).get(Key.DIGEST, "")
            embed_model = manifest.get(Key.EMBEDDING_MODEL, {}).get(Key.MODEL, "bge-m3")
            compose_env = smoke_env(extract_dir, bge_m3_digest, embed_model)
            up_result = lifecycle.up(PROJECT, compose_env, compose_file=compose_file)
            results.append(("compose up -d", up_result.returncode == 0, up_result.stderr.strip()))

            try:
                if up_result.returncode == 0:
                    pg_healthy = lifecycle.wait_healthy(
                        PROJECT, compose_env, "kb-pg", timeout=KB_PG_HEALTHY_TIMEOUT, compose_file=compose_file,
                    )
                    ollama_healthy = lifecycle.wait_healthy(
                        PROJECT, compose_env, "kb-ollama", timeout=KB_OLLAMA_HEALTHY_TIMEOUT, compose_file=compose_file,
                    )
                else:
                    # `up` itself failed (bad image, port conflict, disk
                    # full) -- the containers never exist, so both
                    # health-polls would otherwise spin for their entire
                    # timeout before reporting exactly this same failure,
                    # wasting ~13 minutes for nothing.
                    pg_healthy = ollama_healthy = False
                results.append(("kb-pg healthcheck зелёный", pg_healthy, ""))
                results.append(("kb-ollama healthcheck зелёный", ollama_healthy, ""))

                if pg_healthy:
                    penv = psql_env(compose_env)
                    ollama_url = f"http://127.0.0.1:{OLLAMA_PORT}/api/embed"
                    results.append(("число документов/страниц = манифест", *checks.check_counts(penv, manifest)))
                    results.append(("fulltext находит документы", *checks.check_fulltext(penv, manifest)))
                    results.append(("citation-граф спроецирован",
                                     *checks.check_citation_projection(penv, manifest)))
                    if ollama_healthy:
                        results.append(("vector находит релевантное без совпадения слов",
                                         *checks.check_vector(penv, manifest, ollama_url)))
                        results.append(("bge-m3 digest совпадает с манифестом",
                                         *checks.check_embedding_model_digest(manifest, ollama_url)))
                        if args.measure_drift:
                            report = drift_probe.measure_drift(
                                penv, ollama_url, manifest[Key.VECTOR_PROBE], args.measure_drift,
                            )
                            print(
                                f"[MEASURE] vector_probe drift over {report['n']} re-embeds: "
                                f"max rank shift={report['max_rank_shift']}, "
                                f"max |Δdistance|={report['max_distance_delta']:.6f} "
                                f"(shifts={report['rank_shifts']}, deltas={report['distance_deltas']})"
                            )
                    else:
                        results.append(("vector находит релевантное без совпадения слов", False, "kb-ollama not healthy"))
                        results.append(("bge-m3 digest совпадает с манифестом", False, "kb-ollama not healthy"))
                    results.append(("embedding_model заполнена, размерность совпадает",
                                     *checks.check_embedding_model_dims(penv)))
                    # SKIP, not FAIL, when the artifact declares no
                    # measurements schema (the public profile ships schema
                    # corpus only): querying measurements.run there is a
                    # psql error about a missing relation, which says nothing
                    # about the package's health. The absence itself is
                    # already asserted statically, against the dump, by
                    # profile_checks.check_schemas.
                    if "measurements" in manifest.get(Key.SCHEMAS, []):
                        results.append(("measurements.run присутствует полностью",
                                         *checks.check_measurements_run(penv, manifest)))
                    else:
                        results.append(("measurements.run присутствует полностью", None,
                                         f"профиль {manifest.get(Key.PROFILE)!r} не несёт схему "
                                         "measurements -- проверять нечего"))
                    probe_doc = manifest[Key.BLOB_PROBE][Key.DOCUMENT_ID]
                    blob_result = checks.blob_sha256(penv, probe_doc)
                    # Named from the manifest, not hardcoded: each profile
                    # probes a document whose blob it actually ships (see
                    # manifest_probe.blob_probe_doc).
                    results.append((f"round-trip блоба {probe_doc}",
                                     *checks.check_blob_roundtrip(penv, manifest, blob_result)))
                    results.append(("порча sha256 -> HASH MISMATCH",
                                     *checks.check_blob_corruption_detected(penv, manifest, blob_result)))
                elif up_result.returncode == 0:
                    print("kb-pg never became healthy; skipping DB-level checks", file=sys.stderr)
            finally:
                down_result = lifecycle.down(PROJECT, compose_env, compose_file=compose_file)
                remaining = lifecycle.volumes_remaining(PROJECT)
                teardown_ok = down_result.returncode == 0 and not remaining
                results.append(("down -v чистый снос", teardown_ok,
                                 down_result.stderr.strip() or f"remaining volumes: {remaining}"))

    try:
        with artifact_data_dir(args.artifact, args.artifact_dir, args.profile) as (extract_dir, pristine):
            _run_checks(extract_dir, pristine)
    except ArtifactUnavailable as exc:
        # A resolution failure (empty/missing deploy dir, no artifact given
        # and no repository to auto-discover one from) is exactly as
        # reportable as any other check here -- it used to raise SystemExit
        # straight out of artifact_data_dir/latest_artifact instead, which
        # skipped this results list and the OK/FAIL loop below entirely.
        results.append(("артефакт найден", False, str(exc)))

    results.append(("живой инстанс не задет", *check_live_instance_intact(args.live_pgenv)))

    all_ok = True
    for name, ok, detail in results:
        status = "SKIP" if ok is None else ("OK" if ok else "FAIL")
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        if ok is not None:
            all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
