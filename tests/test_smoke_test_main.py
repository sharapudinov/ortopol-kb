"""Unit tests for smoke_test.py's main() orchestration: no Docker, no live
Postgres. Every lifecycle.*/checks.* collaborator is monkeypatched (each has
its own dedicated test module already: test_deploy_units.py, test_pg_rank_
probe.py, test_blob_integrity_checks.py) so only main()'s own branch logic
-- the dump-integrity gate, the up-failure/pg-unhealthy/ollama-unhealthy
short-circuits, and the teardown_ok computation -- is under test here.
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import smoke_test

_DEFAULT_CHECK_RESULTS = {
    "check_bundled_files": (True, "ok"),
    "check_counts": (True, "ok"),
    "check_fulltext": (True, "ok"),
    "check_vector": (True, "ok"),
    "check_embedding_model_digest": (True, "ok"),
    "check_embedding_model_dims": (True, "ok"),
    "check_measurements_run": (True, "ok"),
    "check_blob_roundtrip": (True, "ok"),
    "check_blob_corruption_detected": (True, "ok"),
}


def _completed(returncode=0, stderr=""):
    return mock.Mock(returncode=returncode, stderr=stderr)


class ArtifactFixture:
    """The stub artifact directory and the collaborator patching that both
    test classes below share. A plain mixin, not a TestCase: making the
    profile tests inherit from SmokeTestMainTests re-ran every one of its
    tests a second time under the subclass's name.
    """

    def _artifact_dir(self, tmp: str, *, dump_bytes: bytes = b"fake-dump-contents",
                       schema_version: int | None = None,
                       profile: str = "full",
                       schemas: list[str] | None = None) -> Path:
        artifact_dir = Path(tmp) / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "01_dump.sql.gz").write_bytes(dump_bytes)
        manifest = {
            "schema_version": (
                smoke_test.checks.MANIFEST_SCHEMA_VERSION if schema_version is None else schema_version
            ),
            "profile": profile,
            "schemas": ["corpus", "measurements"] if schemas is None else schemas,
            "files": {},
            "dump": {"file": "01_dump.sql.gz", "bytes": len(dump_bytes), "sha256": "REAL_SHA"},
            "embedding_model": {"model": "bge-m3", "digest": "sha256:abc"},
            "blob_probe": {"document_id": "1997_sm280"},
            "vector_probe": {"query": "q", "document_id": "doc", "page_number": 1, "rank": 1, "distance": 0.4},
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
        return artifact_dir

    @contextlib.contextmanager
    def _patched(self, *, up_ok=True, pg_healthy=True, ollama_healthy=True,
                 down_ok=True, remaining_volumes=None, check_overrides=None):
        checks_values = dict(_DEFAULT_CHECK_RESULTS)
        checks_values.update(check_overrides or {})
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                smoke_test.lifecycle, "up",
                return_value=_completed(0 if up_ok else 1, stderr="" if up_ok else "boom"),
            ))
            wait_healthy_mock = stack.enter_context(mock.patch.object(smoke_test.lifecycle, "wait_healthy"))
            wait_healthy_mock.side_effect = [pg_healthy, ollama_healthy]
            stack.enter_context(mock.patch.object(
                smoke_test.lifecycle, "down", return_value=_completed(0 if down_ok else 1),
            ))
            stack.enter_context(mock.patch.object(
                smoke_test.lifecycle, "volumes_remaining", return_value=remaining_volumes or [],
            ))
            for name, value in checks_values.items():
                stack.enter_context(mock.patch.object(smoke_test.checks, name, return_value=value))
            stack.enter_context(mock.patch.object(smoke_test.checks, "blob_sha256", return_value=(100, "sha")))
            # profile_checks reads the real dump bytes (see its own test
            # module); the fixture's dump is a stub string, and what is under
            # test here is main()'s branch logic, not the scan.
            stack.enter_context(mock.patch.object(
                smoke_test.profile_checks, "run_checks",
                return_value=[("профиль: содержимое = манифест", True, "stub")],
            ))
            yield


class SmokeTestMainTests(ArtifactFixture, unittest.TestCase):
    def test_dump_sha256_mismatch_fails_even_when_everything_else_is_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="DIFFERENT_SHA"), \
                 self._patched():
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)

    def test_up_failure_short_circuits_health_waits_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(up_ok=False), \
                 mock.patch.object(smoke_test.lifecycle, "wait_healthy") as wait_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)
        wait_mock.assert_not_called()

    def test_pg_unhealthy_skips_db_level_checks_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(pg_healthy=False, ollama_healthy=False), \
                 mock.patch.object(smoke_test.checks, "check_counts") as check_counts_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)
        check_counts_mock.assert_not_called()

    def test_ollama_unhealthy_short_circuits_vector_and_digest_without_calling_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(pg_healthy=True, ollama_healthy=False), \
                 mock.patch.object(smoke_test.checks, "check_vector") as check_vector_mock, \
                 mock.patch.object(smoke_test.checks, "check_embedding_model_digest") as digest_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)  # ollama-unhealthy itself fails the run
        check_vector_mock.assert_not_called()
        digest_mock.assert_not_called()

    def test_teardown_ok_requires_clean_down_and_no_remaining_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(remaining_volumes=["kb-smoke-pg-data"]):
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)

    def test_all_green_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched():
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)

    def test_schema_version_mismatch_fails_before_compose_up(self):
        # c5d48e2f: checked immediately after json.loads(), before any other
        # manifest key is touched -- previously a mismatch surfaced only as
        # an opaque KeyError mid-run (or worse, ran compose up against a
        # manifest shape this checkout doesn't understand).
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp, schema_version=999)
            with mock.patch.object(smoke_test, "sha256_file") as sha256_mock, \
                 mock.patch.object(smoke_test.lifecycle, "up") as up_mock, \
                 mock.patch.object(smoke_test.lifecycle, "down") as down_mock, \
                 mock.patch.object(smoke_test.checks, "check_bundled_files") as bundled_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)
        up_mock.assert_not_called()
        down_mock.assert_not_called()
        bundled_mock.assert_not_called()
        sha256_mock.assert_not_called()

    def test_matching_schema_version_proceeds_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp, schema_version=smoke_test.checks.MANIFEST_SCHEMA_VERSION)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched():
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)

    def test_measure_drift_flag_calls_drift_probe_and_never_affects_exit_code(self):
        # dcae1b3c: --measure-drift is purely diagnostic -- it must run
        # (against the manifest's own vector_probe reference) whenever pg
        # and ollama are healthy, but a large/irrelevant drift report must
        # never fail an otherwise-green run.
        report = {
            "n": 3, "rank_shifts": [0, 1, 0], "distance_deltas": [0.0, 0.0001, 0.0],
            "max_rank_shift": 1, "max_distance_delta": 0.0001,
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test.drift_probe, "measure_drift", return_value=report) as drift_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir), "--measure-drift", "3"])
        self.assertEqual(exit_code, 0)
        drift_mock.assert_called_once()
        (_env, _url, probe, n), _kwargs = drift_mock.call_args
        self.assertEqual(n, 3)
        self.assertEqual(probe, {"query": "q", "document_id": "doc", "page_number": 1, "rank": 1, "distance": 0.4})

    def test_measure_drift_not_called_when_ollama_unhealthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(ollama_healthy=False), \
                 mock.patch.object(smoke_test.drift_probe, "measure_drift") as drift_mock:
                smoke_test.main(["--artifact-dir", str(artifact_dir), "--measure-drift", "3"])
        drift_mock.assert_not_called()

    def test_measure_drift_defaults_to_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test.drift_probe, "measure_drift") as drift_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)
        drift_mock.assert_not_called()


class ProfileAwareMainTests(ArtifactFixture, unittest.TestCase):
    """main() must verify whichever profile the artifact declares, and it
    must learn that from the artifact -- not from its own flags.
    """

    def test_static_profile_checks_run_before_compose_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(
                     smoke_test.profile_checks, "run_checks",
                     return_value=[("профиль", True, "")]) as checks_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)
        checks_mock.assert_called_once_with(artifact_dir)

    def test_a_failing_static_profile_check_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(
                     smoke_test.profile_checks, "run_checks",
                     return_value=[("metadata-only: ни блоба, ни текста", False,
                                     "blobs present for ['1997_sm280']")]):
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 1)

    def test_public_artifact_skips_the_measurements_check_instead_of_failing(self):
        # The public package ships schema corpus only, so measurements.run
        # does not exist there: querying it is a psql error that says nothing
        # about the package. The absence itself is asserted statically.
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp, profile="public", schemas=["corpus"])
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test.checks, "check_measurements_run") as measurements_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)
        measurements_mock.assert_not_called()

    def test_full_artifact_still_checks_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test.checks, "check_measurements_run",
                                    return_value=(True, "ok")) as measurements_mock:
                exit_code = smoke_test.main(["--artifact-dir", str(artifact_dir)])
        self.assertEqual(exit_code, 0)
        measurements_mock.assert_called_once()

    def test_blob_probe_document_comes_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp, profile="public", schemas=["corpus"])
            manifest_path = artifact_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["blob_probe"] = {"document_id": "2009_isu34"}
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test.checks, "blob_sha256",
                                    return_value=(10, "sha")) as blob_mock:
                smoke_test.main(["--artifact-dir", str(artifact_dir)])
        (_env, probe_doc), _kwargs = blob_mock.call_args
        self.assertEqual(probe_doc, "2009_isu34")

    def test_profile_flag_only_steers_auto_discovery(self):
        # --profile picks which artifact to FIND when none is given; it never
        # overrides what the found artifact says about itself.
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = self._artifact_dir(tmp)

            @contextlib.contextmanager
            def fake_data_dir(artifact, artifact_dir_arg, profile=None):
                fake_data_dir.seen = (artifact, artifact_dir_arg, profile)
                yield artifact_dir, False

            with mock.patch.object(smoke_test, "sha256_file", return_value="REAL_SHA"), \
                 self._patched(), \
                 mock.patch.object(smoke_test, "artifact_data_dir", fake_data_dir):
                smoke_test.main(["--profile", "public"])
        self.assertEqual(fake_data_dir.seen, (None, None, "public"))


if __name__ == "__main__":
    unittest.main()
