"""Unit tests for deploy/artifact_bundle.py: bundling the
runtime files into the artifact workdir, and streaming pg_dump through
gzip without a real Postgres/pg_dump.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import re
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
from manifest_contract import CitationMode, Profile, schemas_for
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
    """pg_dump is stubbed here; nothing else is. The full profile's row
    counts are READ BACK OFF the file it wrote, so a stubbed pg_dump output
    carrying real COPY blocks is the whole input the count needs -- there is
    no second database read left to stub.
    """

    # A miniature pg_dump output: two COPY blocks, one per schema the
    # manifest is stamped from, plus a table of neither.
    DUMP = (b"CREATE SCHEMA corpus;\n"
            b"COPY corpus.documents (id) FROM stdin;\n2009_isu34\n1997_sm280\n\\.\n"
            b"COPY citation.work (id) FROM stdin;\n1\n2\n3\n\\.\n"
            b"COPY measurements.run (id) FROM stdin;\n7\n\\.\n")

    def test_the_dump_reports_what_the_manifest_will_declare(self):
        """Both writers answer with {table: rows} per schema: the packager
        stamps it into the manifest, and the recipient's gate holds the
        shipped bytes to it. Here the answer IS the shipped bytes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(self.DUMP)):
                carried = artifact_bundle.dump_schemas(
                    {}, gz_path, CitationMode.FULL_SKELETON)
        self.assertEqual(carried.citation, {"work": 3})
        self.assertEqual(carried.corpus, {"documents": 2})

    def test_a_dump_carrying_no_citation_block_reports_no_citation_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(b"-- fake\n")):
                carried = artifact_bundle.dump_schemas({}, gz_path, CitationMode.NONE)
        self.assertEqual(carried.citation, {})
        self.assertEqual(carried.corpus, {})

    def test_no_count_is_asked_of_the_live_database_at_all(self):
        """The read that used to follow pg_dump is gone, not merely
        unused: a count from a fresh connection describes whatever the
        database held at that moment, and the gate demands it equal the
        file.
        """
        self.assertFalse(hasattr(artifact_bundle, "live_row_counts"))

    def test_the_dump_asks_for_exactly_what_the_manifest_declares(self):
        """One source of truth for "which schemas does this profile ship":
        manifest.json's schemas[] and pg_dump's --schema arguments are the
        same list from schemas_for(), not two lists kept equal by hand.
        """
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                gz_path = Path(tmp) / "dump.sql.gz"
                with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                        return_value=FakeProc(b"-- fake\n")) as popen_mock:
                    artifact_bundle.dump_schemas({}, gz_path, citation_mode=mode)
                (argv,), _kwargs = popen_mock.call_args
                asked = [a.split("=", 1)[1] for a in argv if a.startswith("--schema=")]
                self.assertEqual(asked, schemas_for(Profile.FULL, mode))

    def test_the_owners_own_backup_carries_the_citation_schema(self):
        # The full profile applies no legal or policy cut: whenever the
        # schema exists at all (i.e. the mode is not NONE) it ships whole.
        self.assertIn("citation", schemas_for(Profile.FULL, CitationMode.FULL_SKELETON))
        self.assertNotIn("citation", schemas_for(Profile.FULL, CitationMode.NONE))

    def test_dumps_never_carry_age_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(b"-- fake\n")) as popen_mock:
                artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
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
                artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
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
                    artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
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
                    target=lambda: (artifact_bundle.dump_schemas(
                        {}, gz_path, CitationMode.FULL_SKELETON), done.set()),
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
