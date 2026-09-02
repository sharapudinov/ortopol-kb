"""The manifest-version gate on profile_checks.run_checks().

The recipient runs profile_checks.py standalone (deploy/AGENT_GUIDE.md) and
build_package.py names it as the authority an override build cannot be
certified by, so the version comparison cannot live only in the Docker path
(smoke_checks.py had the only one). Against a manifest of another version
every check below reads its keys as absent -- and an absent key is what
turns a certification into a row of trivially satisfied checks.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import profile_checks
from _artifact_fixtures import ArtifactBuilder
from manifest_contract import MANIFEST_SCHEMA_VERSION, Key

GATE = "версия манифеста = версия проверяльщика"


def _with_version(directory: Path, version) -> Path:
    """A well-formed artifact whose manifest then claims `version`."""
    ArtifactBuilder(directory).write()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if version is None:
        manifest.pop(Key.SCHEMA_VERSION)
    else:
        manifest[Key.SCHEMA_VERSION] = version
    path.write_text(json.dumps(manifest, ensure_ascii=False))
    return directory


class ManifestVersionGateTests(unittest.TestCase):
    def test_the_current_version_passes_and_the_pass_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = profile_checks.run_checks(ArtifactBuilder(Path(tmp)).write())
        self.assertEqual(results[0][0], GATE)
        self.assertTrue(results[0][1], results[0][2])
        self.assertGreater(len(results), 1)

    def test_an_older_manifest_fails_the_gate_and_stops_the_pass(self):
        """Not "fails among ten passes": the checks underneath a manifest
        this reader cannot read are not evidence about the package, and a
        list of green rows beneath a red one reads as a certification.
        """
        with tempfile.TemporaryDirectory() as tmp:
            results = profile_checks.run_checks(
                _with_version(Path(tmp), MANIFEST_SCHEMA_VERSION - 1))
        self.assertEqual(len(results), 1)
        name, ok, detail = results[0]
        self.assertEqual(name, GATE)
        self.assertFalse(ok)
        self.assertIn(str(MANIFEST_SCHEMA_VERSION), detail)

    def test_a_manifest_with_no_version_at_all_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = profile_checks.run_checks(_with_version(Path(tmp), None))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0][1])

    def test_a_newer_manifest_is_refused_too(self):
        """The gate is equality, not a floor: a field this reader has never
        heard of is exactly as unreadable as one that disappeared.
        """
        with tempfile.TemporaryDirectory() as tmp:
            results = profile_checks.run_checks(
                _with_version(Path(tmp), MANIFEST_SCHEMA_VERSION + 1))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0][1])

    def test_the_cli_exits_nonzero_on_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _with_version(Path(tmp), MANIFEST_SCHEMA_VERSION - 1)
            self.assertEqual(profile_checks.main(["--artifact-dir", str(directory)]), 1)


if __name__ == "__main__":
    unittest.main()
