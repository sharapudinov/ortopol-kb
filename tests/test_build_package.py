"""Unit tests for deploy/build_package.py's main(): no Docker,
no live Postgres, no real pg_dump/tar. Every collaborator main() orchestrates
(gather_manifest, bundle_runtime_files, dump_schemas, package, sha256_file)
has its own dedicated test module already -- this one covers main()'s own
control flow instead: argument/path defaults, the PostgresUnavailable ->
print+return 1 error path, and the happy-path sequencing that assembles
manifest['files']/manifest['dump'] before writing manifest.json.

The citation half of main() -- whose decision the mode was, how many times
the policy is read, and the --policy-override build -- is next door in
test_build_package_citation.py (kb/CLAUDE.md FILE_SIZE, split along that
seam); the manifest both drive main() with is _build_package_fixtures.py.
"""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import artifact_bundle
import build_package
from copy_rows import DumpedRows
import public_dump
from citation_profile import CitationUnclassified
from legal_profile import Unclassified
from _build_package_fixtures import fake_manifest
from pg_common import PostgresUnavailable


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

            def fake_dump_schemas(env, gz_path, *, citation_mode):
                gz_path.write_bytes(b"\x1f\x8b\x08\x00fake-gzip")
                # Both writers answer with what they carried, per citation
                # table -- main() stamps it into the manifest block.
                return DumpedRows(corpus={"documents": 70, "pages": 2462},
                                  citation={"work": 441, "cites": 2427})

            def fake_package(workdir, out_path):
                captured["manifest"] = json.loads((workdir / "manifest.json").read_text())
                out_path.write_bytes(b"fake-tar-zst")

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "full_profile_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "gather_manifest",
                                    return_value=fake_manifest()), \
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

            def fake_full(env, gz_path, *, citation_mode):
                seen["writer"] = "full"
                gz_path.write_bytes(b"full")
                return DumpedRows(corpus={"documents": 70, "pages": 2462},
                                  citation={"work": 441})

            def fake_public(env, gz_path, *, citation_mode):
                seen["writer"] = "public"
                seen["dump_citation_mode"] = citation_mode
                gz_path.write_bytes(b"public")
                return DumpedRows(corpus={"documents": 3, "pages": 4},
                                  citation={"work": 2, "cites": 1})

            def fake_package(workdir, out_path):
                seen["manifest"] = json.loads((workdir / "manifest.json").read_text())
                out_path.write_bytes(b"fake-tar-zst")

            def fake_gather(env, ollama_url, profile="full", *,
                            citation_mode, policy_source):
                seen["gathered_profile"] = profile
                seen["manifest_citation_mode"] = citation_mode
                seen["policy_source"] = policy_source
                return fake_manifest(
                    profile=profile_in_manifest,
                    schemas=["corpus"] if profile_in_manifest == "public"
                    else ["corpus", "measurements"],
                )

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "full_profile_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_schemas", side_effect=fake_full), \
                 mock.patch.object(build_package, "dump_public", side_effect=fake_public), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package):
                exit_code = build_package.main(argv)
            seen["written"] = sorted(p.name for p in (corpus_dir / "deploy").glob("*.tar.zst"))
        return exit_code, seen

    def test_the_manifest_declares_what_the_writer_says_it_carried(self):
        """manifest.citation.table_rows is stamped from the dump's own
        answer, not re-derived: the recipient's gate requires every table
        named there to be in the file with exactly that many rows, so the
        two must be one resolution of one cut.

        The two headline totals come from that same answer, for the same
        reason and one step further: counted against the live database
        before the dump existed, they were the same cut row sets counted
        twice, and the gate compares them against these very COPY blocks.
        """
        for argv, profile, expected in ((["--profile", "public"], "public",
                                         {"work": 2, "cites": 1}),
                                        ([], "full", {"work": 441})):
            with self.subTest(profile=profile):
                _code, seen = self._build(argv, profile)
                citation = seen["manifest"]["citation"]
                self.assertEqual(citation["table_rows"], expected)
                self.assertEqual(citation["work_count"], expected["work"])
                self.assertEqual(citation["cites_count"], expected.get("cites", 0))

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
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("topology-only", "owner")), \
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
        # Unlike an unclassified DOCUMENT, this is a whole-schema policy
        # (citation.public_policy has no row), and it is refused at the one
        # place the mode is resolved -- before the manifest is gathered.
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    side_effect=CitationUnclassified("citation.public_policy: no row")), \
                 mock.patch.object(build_package, "gather_manifest") as gather_mock, \
                 mock.patch.object(build_package, "dump_public") as public_mock, \
                 mock.patch.object(build_package, "package") as package_mock, \
                 mock.patch("sys.stderr") as stderr_mock:
                exit_code = build_package.main(["--profile", "public"])
            written = list((corpus_dir / "deploy").glob("*.tar.zst"))
        self.assertEqual(exit_code, 1)
        gather_mock.assert_not_called()
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
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "full_profile_mode",
                                    return_value=("full-skeleton", "not-applicable")), \
                 mock.patch.object(build_package, "gather_manifest", return_value=fake_manifest()), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_schemas",
                                    side_effect=lambda env, gz, citation_mode="none":
                                    gz.write_bytes(b"x")
                                    and DumpedRows(corpus={}, citation={})), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package):
                exit_code = build_package.main(["--pgenv", str(explicit_pgenv), "--out-dir", str(out_dir)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(seen_pgenv_paths, [explicit_pgenv])


class TheDumpSeamHasNoDefaultModeTests(unittest.TestCase):
    """dump_schemas() always required the resolved mode; its two siblings on
    the same seam defaulted to `none`, so a caller that said nothing got the
    cut that ships nothing -- a decision taken by omission on the one
    question only the owner answers.
    """

    def test_no_writer_on_the_seam_supplies_a_mode_of_its_own(self):
        for fn in (build_package.write_dump, public_dump.dump_public,
                   artifact_bundle.dump_schemas):
            with self.subTest(writer=fn.__name__):
                parameter = inspect.signature(fn).parameters["citation_mode"]
                self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_write_dump_refuses_a_call_that_names_no_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                build_package.write_dump("full", {}, Path(tmp) / "01_dump.sql.gz")


if __name__ == "__main__":
    unittest.main()
