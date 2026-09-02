"""Unit tests for deploy/artifact_bundle.py: bundling the
runtime files into the artifact workdir, and streaming pg_dump through
gzip without a real Postgres/pg_dump.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import artifact_bundle


class BundleRuntimeFilesTests(unittest.TestCase):
    def test_copies_every_declared_file_with_matching_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            files = artifact_bundle.bundle_runtime_files(workdir)

            expected_relpaths = set(artifact_bundle.DEPLOY_FILES) | {
                f"corpus_lib/{name}" for name in artifact_bundle.CORPUS_LIB_FILES
            }
            self.assertEqual(set(files), expected_relpaths)

            for rel, want_sha in files.items():
                dst = workdir / rel
                self.assertTrue(dst.is_file(), f"{rel} was not copied")
                self.assertEqual(hashlib.sha256(dst.read_bytes()).hexdigest(), want_sha)

    def test_bundled_corpus_lib_matches_the_real_source_modules(self):
        # A tampered/stale copy would still "exist" but disagree byte-for-
        # byte with the repository's own pg_common.py/pg_search.py.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            for name in artifact_bundle.CORPUS_LIB_FILES:
                bundled = (workdir / "corpus_lib" / name).read_bytes()
                source = (artifact_bundle.CORPUS_DIR / name).read_bytes()
                self.assertEqual(bundled, source)


class FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdout = io.BytesIO(stdout)
        self._stderr = io.BytesIO(stderr)
        self._returncode = returncode

    @property
    def stderr(self):
        return self._stderr

    def wait(self):
        return self._returncode


class DumpSchemasTests(unittest.TestCase):
    def test_full_schemas_include_citation(self):
        # The full profile is the owner's own backup -- it carries the whole
        # citation schema unconditionally, the same as corpus/measurements.
        self.assertEqual(artifact_bundle.FULL_SCHEMAS, ("corpus", "measurements", "citation"))

    def test_dumps_never_carry_age_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(b"-- fake\n")) as popen_mock:
                artifact_bundle.dump_schemas({}, gz_path)
            (argv,), _kwargs = popen_mock.call_args
        self.assertIn("--exclude-schema=citation_graph", argv)
        self.assertIn("--exclude-schema=ag_catalog", argv)
        self.assertIn("--schema=citation", argv)

    def test_streams_pg_dump_stdout_through_gzip(self):
        payload = b"-- fake pg_dump output\n" * 1000
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(payload)):
                artifact_bundle.dump_schemas({}, gz_path)
            self.assertTrue(gz_path.is_file())
            with gzip.open(gz_path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_pg_dump_failure_raises_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(
                artifact_bundle.subprocess, "Popen",
                return_value=FakeProc(b"partial", stderr=b"connection refused", returncode=1),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    artifact_bundle.dump_schemas({}, gz_path)
            self.assertIn("connection refused", str(ctx.exception))
            self.assertFalse(gz_path.exists())

    def test_large_stderr_output_does_not_deadlock(self):
        # 16bd7012: with both stdout and stderr as pipes and nothing
        # draining stderr while the main thread blocks copying stdout, a
        # child that writes enough to stderr (NOTICE/WARNING lines are
        # routine on a large multi-schema pg_dump) to fill the OS pipe
        # buffer (~64KB on Linux) would deadlock -- the child blocks
        # writing to stderr, the parent blocks reading stdout, forever. A
        # real subprocess is required to reproduce that: FakeProc's
        # in-memory BytesIO has no OS pipe buffer to fill. Popen is
        # redirected to a real python3 -c script instead of the real
        # "pg_dump" argv (not installed, and irrelevant to the pipe
        # mechanics under test); the whole call is run on a background
        # thread with a bounded join() so an actual deadlock fails this
        # test instead of hanging the suite forever.
        script = (
            "import sys\n"
            "sys.stderr.write('E' * 200000)\n"
            "sys.stderr.flush()\n"
            "sys.stdout.buffer.write(b'ok')\n"
        )
        real_argv = [sys.executable, "-c", script]
        real_popen = subprocess.Popen  # captured before patching -- see below

        def fake_popen(_argv, **kwargs):
            # NOT subprocess.Popen: mock.patch.object below patches the
            # Popen attribute on the actual subprocess module (the same
            # object artifact_bundle.subprocess IS), so calling
            # subprocess.Popen from inside this function would recurse into
            # itself instead of spawning the real interpreter.
            return real_popen(real_argv, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            done = threading.Event()
            with mock.patch.object(artifact_bundle.subprocess, "Popen", side_effect=fake_popen):
                worker = threading.Thread(
                    target=lambda: (artifact_bundle.dump_schemas({}, gz_path), done.set()),
                )
                worker.start()
                worker.join(timeout=10)
            self.assertTrue(done.is_set(), "dump_schemas deadlocked on large stderr output")
            with gzip.open(gz_path, "rb") as f:
                self.assertEqual(f.read(), b"ok")


class PackageTests(unittest.TestCase):
    """package() (the tar --zstd invocation producing the shipped
    kb-<date>.tar.zst) had zero test coverage anywhere: test_build_package.py
    always monkeypatches it with a stub. These exercise the real
    subprocess.run call end to end (round-trip contents through a real tar
    --zstd extraction) and its CalledProcessError failure path.
    """

    def test_real_tar_zstd_round_trips_workdir_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            (workdir / "manifest.json").write_text('{"schema_version": 3}')
            (workdir / "01_dump.sql.gz").write_bytes(b"\x1f\x8b" + b"fake-gzip-payload")
            nested = workdir / "corpus_lib"
            nested.mkdir()
            (nested / "pg_common.py").write_text("# fake module\n")

            out_path = Path(tmp) / "kb-test.tar.zst"
            artifact_bundle.package(workdir, out_path)
            self.assertTrue(out_path.is_file())

            extract_dir = Path(tmp) / "extracted"
            extract_dir.mkdir()
            subprocess.run(
                ["tar", "--zstd", "-xf", str(out_path), "-C", str(extract_dir)], check=True,
            )
            self.assertEqual(
                (extract_dir / "manifest.json").read_text(), '{"schema_version": 3}',
            )
            self.assertEqual(
                (extract_dir / "01_dump.sql.gz").read_bytes(), b"\x1f\x8b" + b"fake-gzip-payload",
            )
            self.assertEqual(
                (extract_dir / "corpus_lib" / "pg_common.py").read_text(), "# fake module\n",
            )

    def test_nonzero_exit_raises_calledprocesserror(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            out_path = Path(tmp) / "kb-test.tar.zst"
            with mock.patch.object(
                artifact_bundle.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, ["tar"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    artifact_bundle.package(workdir, out_path)

    def test_argv_uses_tar_zstd_create_rooted_at_workdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            out_path = Path(tmp) / "kb-test.tar.zst"
            with mock.patch.object(artifact_bundle.subprocess, "run") as run_mock:
                artifact_bundle.package(workdir, out_path)
            (args,), kwargs = run_mock.call_args
            self.assertEqual(args, ["tar", "--zstd", "-cf", str(out_path), "-C", str(workdir), "."])
            self.assertTrue(kwargs.get("check"))


class ImportClosureTests(unittest.TestCase):
    """168db9fb: DEPLOY_FILES/CORPUS_LIB_FILES are a hand-maintained list
    with no structural guarantee that every module smoke_test.py's import
    chain actually needs is declared -- exactly the drift already seen once
    (blob_integrity_checks.py, ollama_registry.py, pg_rank_probe.py all
    landed as new imports without anything catching a missing declaration).
    Runs a real, separate `python3 -c "import smoke_test"` with the bundle
    directory as the only entry on sys.path (Python adds '' -- the cwd --
    for -c), so a genuinely undeclared/missing module fails exactly the way
    a foreign agent's standalone `python3 smoke_test.py` run would, rather
    than a simulation of it. A real subprocess is required, not an in-
    process import: sys.modules caches by name, so re-importing e.g.
    "pg_common" in-process would silently return whatever copy an earlier
    test already imported from the real repository checkout, regardless
    of what sys.path says -- never actually exercising bundle resolution.
    """

    def test_smoke_test_import_chain_resolves_entirely_inside_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            result = subprocess.run(
                [sys.executable, "-c", "import smoke_test"],
                cwd=workdir, capture_output=True, text=True,
            )
        self.assertEqual(
            result.returncode, 0,
            "smoke_test.py's import chain did not resolve entirely inside the bundle "
            f"(missing from DEPLOY_FILES/CORPUS_LIB_FILES?): {result.stderr}",
        )


    def test_graph_query_modules_resolve_inside_the_bundle(self):
        """AGENT_GUIDE.md documents `pg_graph.py citers|candidates|
        cocitation|hybrid` for artifact recipients, and those four
        subcommands import pg_graph_queries (which imports pg_graph_cypher,
        and both import pg_graph_common). Bundling the plumbing alone --
        enough for the smoke check that only needs graph_exists/graph_counts
        -- left every documented query command failing with
        ModuleNotFoundError on a package that declares it can answer them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            result = subprocess.run(
                [sys.executable, "-c",
                 "import pg_graph_common, pg_graph, pg_graph_queries"],
                cwd=workdir / "corpus_lib", capture_output=True, text=True,
            )
        self.assertEqual(
            result.returncode, 0,
            "the graph query commands AGENT_GUIDE.md documents do not import "
            f"inside the bundle: {result.stderr}",
        )

    def test_deploy_scripts_reach_the_graph_modules_through_the_pathfix(self):
        """The other entry point: a deploy script (smoke_checks) importing
        the graph modules must find them through deploy_pathfix's corpus_lib
        shim, from the deploy directory rather than from corpus_lib itself.
        """
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            result = subprocess.run(
                [sys.executable, "-c",
                 "from deploy_pathfix import ensure_corpus_importable; "
                 "ensure_corpus_importable(); "
                 "import pg_common, pg_search, pg_graph_common, pg_graph, "
                 "pg_graph_queries"],
                cwd=workdir, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
