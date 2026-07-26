"""Unit tests for deploy/smoke_stack.py: the throwaway
kb-smoke deploy target's artifact-resolution helpers (latest_artifact,
artifact_data_dir, check_live_instance_intact). No Docker, no live
database. smoke_test.py's own orchestration (main()) has its dedicated
test_smoke_test_main.py, and the check_*/compose_lifecycle collaborators
have theirs (test_smoke_checks*.py, test_compose_lifecycle.py).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import smoke_stack


class LatestArtifactTests(unittest.TestCase):
    def test_empty_directory_raises_artifact_unavailable(self):
        # Not SystemExit: this is a domain error the library layer raises,
        # not a process-exit decision -- see ArtifactUnavailable's docstring.
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            (corpus_dir / "deploy").mkdir()
            with self.assertRaises(smoke_stack.ArtifactUnavailable):
                smoke_stack.latest_artifact(corpus_dir)

    def test_picks_lexicographically_last_same_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            deploy_dir = corpus_dir / "deploy"
            deploy_dir.mkdir()
            for name in ["kb-20260101.tar.zst", "kb-20260615.tar.zst", "kb-20260228.tar.zst"]:
                (deploy_dir / name).touch()
            picked = smoke_stack.latest_artifact(corpus_dir)
            self.assertEqual(picked.name, "kb-20260615.tar.zst")

    def test_picks_correctly_across_a_year_boundary(self):
        # Lexicographic sort of YYYYMMDD is also chronological across a
        # year rollover -- this pins that assumption down explicitly.
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            deploy_dir = corpus_dir / "deploy"
            deploy_dir.mkdir()
            for name in ["kb-20261231.tar.zst", "kb-20270101.tar.zst", "kb-20260601.tar.zst"]:
                (deploy_dir / name).touch()
            picked = smoke_stack.latest_artifact(corpus_dir)
            self.assertEqual(picked.name, "kb-20270101.tar.zst")


class ArtifactDataDirTests(unittest.TestCase):
    """artifact_data_dir()'s resolution modes: explicit artifact_dir, a
    packed artifact tar.zst, the standalone default of "this script's own
    directory when a manifest.json sits beside it" (the mode a foreign
    agent uses after `tar --zstd -xf kb-<date>.tar.zst -C somewhere`), and
    the dev-loop default of neither given. Each also yields `pristine` --
    whether bundled_files_check.check_bundled_files may treat ANY
    unaccounted file as tampering, or must additionally tolerate
    bundled_files_check.OPERATIONAL_ALLOWLIST (951bf93a/fcbc1fa1).

    Explicit Path | None arguments, not a fabricated argparse.Namespace:
    03f1b2d3 changed artifact_data_dir's own signature to match -- see that
    function's docstring for why.
    """

    def test_explicit_artifact_dir_is_used_as_is_not_pristine_and_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            given = Path(tmp)
            with smoke_stack.artifact_data_dir(artifact=None, artifact_dir=given) as (resolved, pristine):
                self.assertEqual(resolved, given)
                self.assertFalse(pristine)
            self.assertTrue(given.is_dir())  # still exists after the context exits

    def test_no_args_and_no_repository_and_no_local_manifest_raises(self):
        # Not SystemExit: a domain error the library layer raises, not a
        # process-exit decision -- see ArtifactUnavailable's docstring.
        with mock.patch.object(smoke_stack, "try_default_corpus_dir", return_value=None), \
             mock.patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(smoke_stack.ArtifactUnavailable) as ctx:
                with smoke_stack.artifact_data_dir(artifact=None, artifact_dir=None):
                    pass
        self.assertIn("--artifact", str(ctx.exception))

    def test_no_args_with_a_local_manifest_is_in_place_and_not_pristine(self):
        # The extract-and-run-in-place mode itself: `tar --zstd -xf ... -C
        # somewhere && cd somewhere && python3 smoke_test.py`, exactly the
        # sequence AGENT_GUIDE.md documents. extract_dir IS this script's
        # own directory, already touched by the operator (fcbc1fa1's
        # `cp .pgenv.example .pgenv`) before this check ever runs.
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp).resolve()
            (here / "manifest.json").write_text("{}")
            with mock.patch.object(smoke_stack, "__file__", str(here / "smoke_stack.py")):
                with smoke_stack.artifact_data_dir(artifact=None, artifact_dir=None) as (resolved, pristine):
                    self.assertEqual(resolved, here)
                    self.assertFalse(pristine)

    def test_artifact_tar_path_extracts_via_tar_zstd_into_a_pristine_temp_dir(self):
        # 951bf93a: the packed --artifact branch was never exercised by any
        # test. subprocess.run is mocked (no real tar --zstd binary
        # required) so this asserts the exact argv and that the yielded
        # directory is the fresh temp extraction target -- pristine=True,
        # since nothing but that extraction has touched it.
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "kb-20260101.tar.zst"
            tar_path.touch()
            captured = {}
            with mock.patch.object(smoke_stack.subprocess, "run") as run_mock:
                with smoke_stack.artifact_data_dir(artifact=tar_path, artifact_dir=None) as (resolved, pristine):
                    captured["dir"] = resolved
                    self.assertTrue(pristine)
                    self.assertTrue(resolved.is_dir())
            (argv,), kwargs = run_mock.call_args
            self.assertEqual(argv, ["tar", "--zstd", "-xf", str(tar_path), "-C", str(captured["dir"])])
            self.assertTrue(kwargs.get("check"))
            # tempfile.TemporaryDirectory cleans up on context exit.
            self.assertFalse(captured["dir"].exists())

    def test_dev_loop_default_picks_the_latest_artifact_and_extracts_it_pristinely(self):
        # 951bf93a: the "neither --artifact nor --artifact-dir, no local
        # manifest.json" convenience path composes try_default_corpus_dir()
        # and latest_artifact() -- previously only unit-tested in isolation
        # from each other, never as they actually run together inside
        # artifact_data_dir().
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            deploy_dir = corpus_dir / "deploy"
            deploy_dir.mkdir(parents=True)
            tar_path = deploy_dir / "kb-20260101.tar.zst"
            tar_path.touch()
            with mock.patch.object(smoke_stack, "try_default_corpus_dir", return_value=corpus_dir), \
                 mock.patch("pathlib.Path.is_file", return_value=False), \
                 mock.patch.object(smoke_stack.subprocess, "run") as run_mock:
                with smoke_stack.artifact_data_dir(artifact=None, artifact_dir=None) as (resolved, pristine):
                    self.assertTrue(pristine)
                    self.assertTrue(resolved.is_dir())
            (argv,), _kwargs = run_mock.call_args
            self.assertEqual(argv[3], str(tar_path))


class CheckLiveInstanceIntactTests(unittest.TestCase):
    def test_none_pgenv_is_skipped_not_failed(self):
        ok, detail = smoke_stack.check_live_instance_intact(None)
        self.assertIsNone(ok)
        self.assertIn("--live-pgenv", detail)

    def test_missing_pgenv_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist.pgenv"
            ok, detail = smoke_stack.check_live_instance_intact(missing)
        self.assertFalse(ok)

    def test_run_sql_failure_after_successful_load_pgenv_fails(self):
        # The third of check_live_instance_intact's three branches, and the
        # actual "live instance unreachable after smoke run" scenario this
        # check exists to catch -- previously untested, unlike the
        # pgenv-None and load_pgenv-raising branches above.
        with tempfile.TemporaryDirectory() as tmp:
            pgenv = Path(tmp) / "live.pgenv"
            pgenv.write_text("PGUSER=ortopol\n")
            with mock.patch.object(
                smoke_stack, "run_sql", side_effect=RuntimeError("connection refused"),
            ):
                ok, detail = smoke_stack.check_live_instance_intact(pgenv)
        self.assertFalse(ok)
        self.assertIn("unreachable after smoke run", detail)
        self.assertIn("connection refused", detail)


if __name__ == "__main__":
    unittest.main()
