"""build_package.py's citation half: whose decision the mode was, and how
many times it is read.

Split from test_build_package.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE), along the seam the packager itself keeps: that module tests
main()'s own control flow -- paths, the unreachable-Postgres exit, the order
manifest.json is assembled in -- and this one the ONE resolution of
citation.public_policy, the --policy-override build that must never look
publishable, and the pair (mode, provenance) that travels from it to both
the manifest and the dump.
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
from copy_rows import DumpedRows
import citation_profile
from citation_profile import CitationUnclassified
from _build_package_fixtures import fake_manifest


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

    def test_override_is_recorded_in_the_manifest_not_only_in_the_name(self):
        """The filename is not part of the package: a copy under another
        name loses it, and a recipient reading manifest.json never sees it.
        policy_source travels INSIDE, and profile_checks.py fails on it.
        """
        _exit_code, seen = self._override_build()
        self.assertEqual(seen["policy_source"], "override")

    def test_an_ordinary_public_build_claims_the_owner_as_its_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            seen = {}

            def fake_gather(env, ollama_url, profile="full", *,
                            citation_mode, policy_source):
                seen["policy_source"] = policy_source
                return fake_manifest(profile="public", schemas=["corpus", "citation"])

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("full-skeleton", "owner")), \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_public",
                                    side_effect=lambda e, p, citation_mode="none":
                                    p.write_bytes(b"x")
                                    and DumpedRows(corpus={}, citation={})), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package",
                                    side_effect=lambda w, o: o.write_bytes(b"z")):
                exit_code = build_package.main(["--profile", "public"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["policy_source"], "owner")

    def test_override_renames_the_artifact_and_bypasses_the_database_decision(self):
        exit_code, seen = self._override_build()
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["manifest_mode"], "topology-only")
        self.assertEqual(seen["dump_mode"], "topology-only")
        written = seen["written"]
        self.assertEqual(len(written), 1)
        self.assertTrue(written[0].startswith("kb-override-public-"), written)

    def test_the_override_name_is_outside_every_profile_namespace(self):
        """Not merely distinguishable from a classified build -- unreachable
        by any selection made on the profile's name. `kb-public-*` is how a
        publish step, a release script or a human picks "the public
        artifact", and a tag suffixed INSIDE that namespace answered to all
        three; the one anchored regex in smoke_stack.py was the only
        consumer that knew better (tests/test_smoke_stack.py pins that side).
        """
        _exit_code, seen = self._override_build()
        name = seen["written"][0]
        for profile in ("public", "full"):
            self.assertFalse(name.startswith(f"kb-{profile}-"), name)

    def _override_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            seen = {}

            def fake_gather(env, ollama_url, profile="full", *,
                            citation_mode, policy_source):
                seen["manifest_mode"] = citation_mode
                seen["policy_source"] = policy_source
                return fake_manifest(profile="public", schemas=["corpus", "citation"])

            def fake_public(env, gz_path, *, citation_mode):
                seen["dump_mode"] = citation_mode
                gz_path.write_bytes(b"public")
                return DumpedRows(corpus={"documents": 3, "pages": 4},
                                  citation={"work": 2, "cites": 1})

            def fake_package(workdir, out_path):
                out_path.write_bytes(b"fake-tar-zst")

            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "ortopol"}), \
                 mock.patch.object(build_package, "resolve_citation_mode",
                                    return_value=("topology-only", "override")) as resolve_mock, \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_public", side_effect=fake_public), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=fake_package), \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main(
                    ["--profile", "public", "--policy-override", "topology-only"])
            resolve_mock.assert_called_once()
            seen["written"] = sorted(p.name for p in (corpus_dir / "deploy").glob("*.tar.zst"))
        return exit_code, seen


class CitationModeResolvedOnceTests(unittest.TestCase):
    """The citation policy is data, and a build reads it exactly once: the
    manifest block and the dump must describe the same cut, and two
    independent resolutions of the same policy can disagree (a schema that
    exists for one reader and not for the other, an override honoured by one
    and not the other). main() resolves, both consumers receive.
    """

    def _run(self, argv, resolved=None, resolve_error=None, resolve=None):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            seen = {}

            def fake_gather(env, ollama_url, profile="full", *,
                            citation_mode, policy_source):
                seen["manifest_mode"] = citation_mode
                seen["policy_source"] = policy_source
                return fake_manifest(profile=profile, schemas=["corpus"])

            def fake_public(env, gz_path, *, citation_mode):
                seen["dump_mode"] = citation_mode
                gz_path.write_bytes(b"public")
                return DumpedRows(corpus={"documents": 3, "pages": 4},
                                  citation={"work": 2, "cites": 1})

            def fake_full(env, gz_path, *, citation_mode):
                seen["dump_mode"] = "<full writer>"
                gz_path.write_bytes(b"full")
                return DumpedRows(corpus={"documents": 70, "pages": 2462},
                                  citation={"work": 441})

            resolve = mock.Mock(
                side_effect=resolve_error or resolve,
                **({} if (resolve_error or resolve) else {"return_value": resolved}))
            with mock.patch.object(build_package, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(build_package, "load_pgenv", return_value={"PGUSER": "x"}), \
                 mock.patch.object(build_package, "resolve_citation_mode", resolve), \
                 mock.patch.object(build_package, "gather_manifest", side_effect=fake_gather), \
                 mock.patch.object(build_package, "bundle_runtime_files", return_value={}), \
                 mock.patch.object(build_package, "dump_public", side_effect=fake_public), \
                 mock.patch.object(build_package, "dump_schemas", side_effect=fake_full), \
                 mock.patch.object(build_package, "sha256_file", return_value="a" * 64), \
                 mock.patch.object(build_package, "package", side_effect=lambda w, o: o.write_bytes(b"z")), \
                 mock.patch("sys.stderr"):
                exit_code = build_package.main(argv)
            seen["resolve_calls"] = resolve.call_count
        return exit_code, seen

    def test_manifest_and_dump_receive_the_same_resolved_mode(self):
        exit_code, seen = self._run(["--profile", "public"],
                                    resolved=("topology-only", "owner"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["resolve_calls"], 1, "политика прочитана не один раз")
        self.assertEqual(seen["manifest_mode"], "topology-only")
        self.assertEqual(seen["dump_mode"], "topology-only")
        self.assertEqual(seen["policy_source"], "owner")

    def test_undecided_policy_is_a_refusal_message_not_a_traceback(self):
        exit_code, seen = self._run(
            ["--profile", "public"],
            resolve_error=CitationUnclassified("citation.public_policy has no row"),
        )
        self.assertEqual(exit_code, 1)
        self.assertNotIn("manifest_mode", seen, "манифест собирался после отказа")
        self.assertNotIn("dump_mode", seen, "дамп писался после отказа")

    def test_full_profile_also_resolves_exactly_once(self):
        exit_code, seen = self._run([], resolved=("full-skeleton", "not-applicable"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["resolve_calls"], 1)
        self.assertEqual(seen["manifest_mode"], "full-skeleton")
        self.assertEqual(seen["policy_source"], "not-applicable")

    def test_a_public_build_against_a_schemaless_database_stops(self):
        """Nothing was decided: no schema to read a policy from, no owner
        row, and an artifact carrying no citation graph would be the
        packager answering by omission a question that is the owner's.
        Refused where an unclassified document is refused, and nothing is
        written.
        """
        with mock.patch.object(citation_profile, "citation_schema_exists",
                                return_value=False):
            exit_code, seen = self._run(["--profile", "public"],
                                        resolve=citation_profile.resolve_citation_mode)
        self.assertEqual(exit_code, 1)
        self.assertNotIn("manifest_mode", seen, "манифест собирался после отказа")
        self.assertNotIn("dump_mode", seen, "дамп писался после отказа")

    def test_an_override_against_a_schemaless_database_stops_too(self):
        """The flag names a mode for a schema that exists, and it cannot
        conjure one -- so it does not turn the refusal off either.
        """
        with mock.patch.object(citation_profile, "citation_schema_exists",
                                return_value=False):
            exit_code, seen = self._run(
                ["--profile", "public", "--policy-override", "topology-only"],
                resolve=citation_profile.resolve_citation_mode)
        self.assertEqual(exit_code, 1)
        self.assertNotIn("manifest_mode", seen, "манифест собирался после отказа")

    def test_a_full_build_against_a_schemaless_database_carries_nothing(self):
        """full applies no policy and describes the database as it is: an
        instance with no citation schema simply has no citation block, and
        the provenance says nothing was decided.
        """
        with mock.patch.object(citation_profile, "citation_schema_exists",
                                return_value=False), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            exit_code, seen = self._run([], resolve=citation_profile.resolve_citation_mode)
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen["manifest_mode"], "none")
        self.assertEqual(seen["policy_source"], "not-applicable")
        require_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
