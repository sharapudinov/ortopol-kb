"""Unit tests for deploy/build_package.py's main(): no Docker,
no live Postgres, no real pg_dump/tar. Every collaborator main() orchestrates
(gather_manifest, bundle_runtime_files, dump_schemas, package, sha256_file)
has its own dedicated test module already -- this one covers main()'s own
control flow instead: argument/path defaults, the PostgresUnavailable ->
print+return 1 error path, and the happy-path sequencing that assembles
manifest['files']/manifest['dump'] before writing manifest.json.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import build_package
from citation_profile import CitationUnclassified
from legal_profile import Unclassified
from pg_common import PostgresUnavailable


def _fake_manifest(**overrides) -> dict:
    """What gather_manifest returns, in the shape main() relies on: since
    manifest schema 4 that always includes the profile and the schemas the
    dump carries (main() prints them and profile_checks.py verifies them
    against the dump). Built here rather than inline so the two happy-path
    tests below cannot drift apart.
    """
    manifest = {
        "schema_version": 5,
        "profile": "full",
        "schemas": ["corpus", "measurements"],
        "documents_count": 70,
        "pages_count": 2462,
    }
    manifest.update(overrides)
    return manifest


class MainPostgresUnavailableTests(unittest.TestCase):
    def test_returns_1_and_prints_error_without_touching_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv",
                                    side_effect=PostgresUnavailable("no .pgenv")), \
                 mock.patch.object(build_package, "gather_manifest") as gather_mock, \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main([])
        self.assertEqual(exit_code, 1)
        gather_mock.assert_not_called()


class MainHappyPathTests(unittest.TestCase):
    def test_writes_manifest_json_with_files_and_dump_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            out_dir = corpus_dir / "deploy"
            captured = {}

            def fake_dump_schemas(env, gz_path):
                gz_path.write_bytes(b"\x1f\x8b\x08\x00fake-gzip")

            def fake_package(workdir, out_path):
                captured["manifest"] = json.loads((workdir / "manifest.json").read_text())
                out_path.write_bytes(b"fake-tar-zst")

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "gather_manifest",
                                    return_value=_fake_manifest()), \
                 mock.patch.object(build_package, "bundle_runtime_files",
                                    return_value={"docker-compose.yml": "abc123"}), \
                 mock.patch.object(build_package, "dump_schemas", side_effect=fake_dump_schemas), \
                 mock.patch.object(build_package, "sha256_file", return_value="deadbeef" * 8), \
                 mock.patch.object(build_package, "package", side_effect=fake_package):
                exit_code = build_package.main(["--out-dir", str(out_dir)])
            self.assertTrue(out_dir.is_dir())  # main() creates --out-dir itself

        self.assertEqual(exit_code, 0)
        manifest = captured["manifest"]
        self.assertEqual(manifest["documents_count"], 70)
        self.assertEqual(manifest["files"], {"docker-compose.yml": "abc123"})
        self.assertEqual(manifest["dump"]["file"], "01_dump.sql.gz")
        self.assertEqual(manifest["dump"]["sha256"], "deadbeef" * 8)
        self.assertGreater(manifest["dump"]["bytes"], 0)


class ProfileDispatchTests(unittest.TestCase):
    """--profile picks BOTH the dump writer and the artifact's name. Getting
    either wrong is the expensive kind of mistake here: a full dump named
    kb-public-<date> is an unpublishable file that looks publishable.
    """

    def _build(self, argv, profile_in_manifest):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            seen = {}

            def fake_full(env, gz_path):
                seen["writer"] = "full"
                gz_path.write_bytes(b"full")

            def fake_public(env, gz_path, citation_mode_override=None):
                seen["writer"] = "public"
                seen["citation_mode_override"] = citation_mode_override
                gz_path.write_bytes(b"public")

            def fake_package(workdir, out_path):
                seen["manifest"] = json.loads((workdir / "manifest.json").read_text())
                out_path.write_bytes(b"fake-tar-zst")

            def fake_gather(env, ollama_url, profile="full", citation_mode_override=None):
                seen["gathered_profile"] = profile
                seen["gathered_citation_override"] = citation_mode_override
                return _fake_manifest(
                    profile=profile_in_manifest,
                    schemas=["corpus"] if profile_in_manifest == "public"
                    else ["corpus", "measurements"],
                )

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_schemas", side_effect=fake_full), \
                 mock.patch.object(build_package, "dump_public", side_effect=fake_public), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package):
                exit_code = build_package.main(argv)
            seen["written"] = sorted(p.name for p in (corpus_dir / "deploy").glob("*.tar.zst"))
        return exit_code, seen

    def test_default_profile_is_full_and_uses_the_pg_dump_writer(self):
        exit_code, seen = self._build([], "full")
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["gathered_profile"], "full")
        self.assertEqual(seen["writer"], "full")
        self.assertEqual(len(seen["written"]), 1)
        self.assertTrue(seen["written"][0].startswith("kb-full-"), seen["written"])

    def test_public_profile_uses_the_filtered_writer_and_its_own_name(self):
        exit_code, seen = self._build(["--profile", "public"], "public")
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["gathered_profile"], "public")
        self.assertEqual(seen["writer"], "public")
        self.assertTrue(seen["written"][0].startswith("kb-public-"), seen["written"])

    def test_unclassified_corpus_refuses_to_build_without_writing_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "gather_manifest",
                                    side_effect=Unclassified("2026_new: class=None")), \
                 mock.patch.object(build_package, "dump_public") as public_mock, \
                 mock.patch.object(build_package, "package") as package_mock, \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main(["--profile", "public"])
            written = list((corpus_dir / "deploy").glob("*.tar.zst"))
        self.assertEqual(exit_code, 1)
        public_mock.assert_not_called()
        package_mock.assert_not_called()
        self.assertEqual(written, [])

    def test_unclassified_citation_schema_refuses_to_build_without_writing_anything(self):
        # This gate adds: unlike an unclassified DOCUMENT, this is
        # a whole-schema policy (citation.public_policy has no row) --
        # test_public_build_refuses_unclassified_citation_schema.
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "gather_manifest",
                                    side_effect=CitationUnclassified("citation.public_policy: no row")), \
                 mock.patch.object(build_package, "dump_public") as public_mock, \
                 mock.patch.object(build_package, "package") as package_mock, \
                 mock.patch("sys.stderr") as stderr_mock:
                exit_code = build_package.main(["--profile", "public"])
            written = list((corpus_dir / "deploy").glob("*.tar.zst"))
        self.assertEqual(exit_code, 1)
        public_mock.assert_not_called()
        package_mock.assert_not_called()
        self.assertEqual(written, [])
        printed = "".join(call.args[0] for call in stderr_mock.write.call_args_list)
        self.assertIn("схема citation не классифицирована", printed)

    def test_explicit_pgenv_overrides_the_corpus_dir_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "unused-corpus-dir"  # never created
            out_dir = Path(tmp) / "out"
            explicit_pgenv = Path(tmp) / "explicit.pgenv"
            explicit_pgenv.write_text("PGUSER=ortopol\n")
            seen_pgenv_paths = []

            def fake_load_pgenv(path):
                seen_pgenv_paths.append(path)
                return {"PGUSER": "ortopol"}

            def fake_package(workdir, out_path):
                out_path.write_bytes(b"fake-tar-zst")

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", side_effect=fake_load_pgenv), \
                 mock.patch.object(build_package, "gather_manifest", return_value=_fake_manifest()), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_schemas",
                                    side_effect=lambda env, gz: gz.write_bytes(b"x")), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package):
                exit_code = build_package.main(["--pgenv", str(explicit_pgenv), "--out-dir", str(out_dir)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen_pgenv_paths, [explicit_pgenv])


class CitationPolicyOverrideTests(unittest.TestCase):
    """--policy-override (TEST ONLY, see build_package.py's --help): bypasses
    the database decision and renames the artifact so it can never be
    mistaken for one the owner actually classified.
    """

    def test_override_only_valid_for_public_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "gather_manifest") as gather_mock, \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main(
                    ["--profile", "full", "--policy-override", "none"])
        self.assertEqual(exit_code, 2)
        gather_mock.assert_not_called()

    def test_override_renames_the_artifact_and_bypasses_the_database_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            seen = {}

            def fake_gather(env, ollama_url, profile="full", citation_mode_override=None):
                seen["gathered_override"] = citation_mode_override
                return _fake_manifest(profile="public", schemas=["corpus", "citation"])

            def fake_public(env, gz_path, citation_mode_override=None):
                seen["dump_override"] = citation_mode_override
                gz_path.write_bytes(b"public")

            def fake_package(workdir, out_path):
                out_path.write_bytes(b"fake-tar-zst")

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_public", side_effect=fake_public), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package), \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main(
                    ["--profile", "public", "--policy-override", "topology-only"])
            written = sorted(p.name for p in (corpus_dir / "deploy").glob("*.tar.zst"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["gathered_override"], "topology-only")
        self.assertEqual(seen["dump_override"], "topology-only")
        self.assertEqual(len(written), 1)
        self.assertTrue(written[0].startswith("kb-public-override-"), written)
        # Never named as if it were an ordinary (owner-classified) build.
        self.assertFalse(written[0].startswith("kb-public-2"), written)


if __name__ == "__main__":
    unittest.main()
