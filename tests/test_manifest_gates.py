"""deploy/manifest_gates.py: the two fields profile_checks.py's own wiring
reads before any check can run -- the manifest version, and the dump block
whose path the pass opens.

Both are asked through run_checks() here rather than in isolation, because
what they have to produce is a ROW: a gate that raises instead leaves a
caller extending its own list with ours (smoke_test.py, no try/except) with
a traceback and no results at all, which is the failure they exist to
prevent. The artifact fixture is the same one every other static-verification
module uses (_artifact_fixtures.ArtifactBuilder).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import profile_checks
from manifest_keys import MANIFEST_SCHEMA_VERSION
from _artifact_fixtures import ArtifactBuilder


def _results(builder: ArtifactBuilder) -> dict[str, tuple[bool, str]]:
    directory = builder.write()
    return {name: (ok, detail) for name, ok, detail in profile_checks.run_checks(directory)}


class VersionGateTests(unittest.TestCase):
    def test_a_manifest_of_another_version_stops_the_pass(self):
        """Every check below the gate asks the manifest for a key, and a
        manifest of another version answers by omission -- an absent field
        read as a satisfied check is the certification this refuses.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["schema_version"] = MANIFEST_SCHEMA_VERSION - 1
            path.write_text(json.dumps(manifest, ensure_ascii=False))
            results = profile_checks.run_checks(directory)
        self.assertEqual(len(results), 1)
        name, ok, detail = results[0]
        self.assertFalse(ok, detail)
        self.assertIn("версия манифеста", name)


class DumpBlockGateTests(unittest.TestCase):
    def test_a_manifest_with_no_dump_block_is_a_row_not_a_traceback(self):
        """The twin of the citation gate one key over: run_checks() builds
        the path it opens out of manifest.dump, so a block that is absent
        or is not a mapping used to raise before a single result existed --
        and a caller extending its own list with ours (smoke_test.py, no
        try/except) then aborted with no results at all.
        """
        for label, prepare in (
            ("нет ключа", lambda b: setattr(b, "dump_key", False)),
            ("не словарь", lambda b: setattr(b, "dump", "01_dump.sql.gz")),
            ("нет имени", lambda b: setattr(b, "dump", {"bytes": 1, "sha256": "x"})),
        ):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as tmp:
                    builder = ArtifactBuilder(Path(tmp))
                    prepare(builder)
                    results = _results(builder)
                ok, detail = results["манифест несёт блок dump с существующим файлом"]
                self.assertFalse(ok, detail)

    def test_a_dump_the_package_does_not_carry_is_a_row_not_a_traceback(self):
        """A manifest naming a file the extraction does not contain -- the
        partial-extraction and hand-edit case -- ends in a verdict about
        the package rather than a FileNotFoundError out of gzip.open.
        """
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.dump = {"file": "нет-такого.sql.gz", "bytes": 1, "sha256": "x"}
            results = _results(builder)
        ok, detail = results["манифест несёт блок dump с существующим файлом"]
        self.assertFalse(ok)
        self.assertIn("нет-такого.sql.gz", detail)
        # The gate is the whole answer: no column of passes underneath it.
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
