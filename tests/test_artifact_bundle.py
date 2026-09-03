"""Unit tests for deploy/artifact_bundle.py: bundling the runtime files
into the artifact workdir and packaging them.

The other half of the module -- dump_schemas(), the full profile's pg_dump
streamed through gzip -- is tests/test_dump_schemas.py (kb/CLAUDE.md
FILE_SIZE, and two different questions).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import artifact_bundle
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv


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
        subcommands import pg_graph_candidates, pg_graph_cocitation and
        pg_graph_cypher (all of which import pg_graph_common). Bundling the plumbing alone --
        enough for the smoke check that only needs graph_exists/projection_reading
        -- left every documented query command failing with
        ModuleNotFoundError on a package that declares it can answer them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            result = subprocess.run(
                [sys.executable, "-c",
                 "import pg_graph_common, pg_graph, pg_graph_candidates, "
                 "pg_graph_cocitation, pg_graph_cypher"],
                cwd=workdir / "corpus_lib", capture_output=True, text=True,
            )
        self.assertEqual(
            result.returncode, 0,
            "the graph query commands AGENT_GUIDE.md documents do not import "
            f"inside the bundle: {result.stderr}",
        )

    def test_every_module_relative_file_dependency_is_bundled_too(self):
        """An import graph does not mention a data file a module opens next
        to itself (pg_graph_common.SCHEMA_PATHS are the ones here), so no
        import test can catch its absence: `pg_graph.py init` shipped
        pointing at a schema file that was not in the package.
        """
        pattern = re.compile(r'parent\s*/\s*"([^"]+)"')
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            files = artifact_bundle.bundle_runtime_files(workdir)
            for rel in files:
                if not rel.endswith(".py"):
                    continue
                source = (workdir / rel).read_text(encoding="utf-8")
                for needed in pattern.findall(source):
                    self.assertTrue(
                        (workdir / rel).parent.joinpath(needed).is_file(),
                        f"{rel} opens {needed} beside itself, and it is not bundled",
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
                 "pg_graph_candidates, pg_graph_cocitation"],
                cwd=workdir, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


class BundledGraphCliLiveTests(unittest.TestCase):
    """The recipient's first documented graph command, run the way a
    recipient runs it: out of the extracted bundle, with an explicit
    --pgenv and no repository on sys.path. Applying the schema is
    idempotent (CREATE ... IF NOT EXISTS / CREATE OR REPLACE / DROP
    CONSTRAINT IF EXISTS + ADD), which is what makes this safe to point at
    the live instance -- and what the recipient's own re-run relies on.
    """

    @classmethod
    def setUpClass(cls):
        try:
            corpus_dir = default_corpus_dir()
            env = load_pgenv(corpus_dir / ".pgenv")
        except (PostgresUnavailable, RuntimeError) as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(env):
            raise unittest.SkipTest("Postgres not reachable")
        cls.pgenv = corpus_dir / ".pgenv"

    def test_bundled_init_applies_the_schema_and_repeats_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            artifact_bundle.bundle_runtime_files(workdir)
            for attempt in (1, 2):
                result = subprocess.run(
                    [sys.executable, "pg_graph.py", "--pgenv", str(self.pgenv), "init"],
                    cwd=workdir / "corpus_lib", capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0,
                                 f"bundled `pg_graph.py init` run {attempt}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
