"""Unit tests for deploy/bundled_files_check.check_bundled_files():
pure filesystem, no run_sql/Docker/network involved.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import bundled_files_check


class CheckBundledFilesTests(unittest.TestCase):
    """check_bundled_files() re-hashes manifest["files"] against an
    extracted directory -- pure filesystem, no run_sql/Docker involved.
    """

    def _manifest(self, files, dump_file="01_dump.sql.gz"):
        return {"files": files, "dump": {"file": dump_file}}

    def test_all_declared_files_present_and_matching_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            sha = hashlib.sha256(b"hello").hexdigest()
            manifest = self._manifest({"a.py": sha})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertTrue(ok)
        self.assertIn("missing=none", detail)

    def test_missing_declared_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            manifest = self._manifest({"a.py": "deadbeef" * 8})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertFalse(ok)
        self.assertIn("a.py", detail)

    def test_tampered_file_content_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("tampered")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertFalse(ok)
        self.assertIn("mismatch", detail)

    def test_undeclared_extra_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            (extract_dir / "sneaky.sh").write_text("#!/bin/sh\n")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertFalse(ok)
        self.assertIn("sneaky.sh", detail)

    def test_manifest_json_and_dump_file_are_not_treated_as_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            (extract_dir / "manifest.json").write_text("{}")
            (extract_dir / "01_dump.sql.gz").write_bytes(b"\x1f\x8b")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, _ = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertTrue(ok)

    def test_pycache_is_not_treated_as_extra(self):
        # The "extracted-and-run-in-place" mode: by the time this check
        # runs, importing smoke_test.py's own sibling modules has already
        # written bytecode caches into extract_dir (see check_bundled_files'
        # own comment) -- a build byproduct, not tampering.
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            pycache = extract_dir / "__pycache__"
            pycache.mkdir()
            (pycache / "a.cpython-313.pyc").write_bytes(b"\x00")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, _ = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertTrue(ok)


class CheckBundledFilesPristineModeTests(unittest.TestCase):
    """fcbc1fa1: AGENT_GUIDE.md's own documented in-place flow (`cp
    .pgenv.example .pgenv` in the extracted directory, THEN `python3
    smoke_test.py` from that same directory) used to fail its own
    self-check, because .pgenv is an unaccounted extra file to a scan with
    no notion of "operator byproduct". pristine=False narrows the scan with
    bundled_files_check.OPERATIONAL_ALLOWLIST instead of disabling it outright:
    default (pristine=True, matching every pre-existing call site/test
    above) keeps the strict pristine-extraction behaviour unchanged.
    """

    def _manifest(self, files, dump_file="01_dump.sql.gz"):
        return {"files": files, "dump": {"file": dump_file}}

    def test_pgenv_fails_by_default_pristine_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            (extract_dir / ".pgenv").write_text("PGPASSWORD=x\n")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest)
        self.assertFalse(ok)
        self.assertIn(".pgenv", detail)

    def test_pgenv_is_tolerated_in_non_pristine_mode(self):
        # The literal AGENT_GUIDE.md sequence: cp .pgenv.example .pgenv,
        # THEN python3 smoke_test.py from the same (in-place) directory.
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            (extract_dir / ".pgenv").write_text("PGPASSWORD=x\n")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest, pristine=False)
        self.assertTrue(ok, detail)

    def test_non_allowlisted_extra_file_still_fails_in_non_pristine_mode(self):
        # The allowlist narrows the scan, it does not disable it: a real
        # tampered/unexpected file must still fail even in-place.
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / "a.py").write_text("hello")
            (extract_dir / ".pgenv").write_text("PGPASSWORD=x\n")
            (extract_dir / "sneaky.sh").write_text("#!/bin/sh\n")
            manifest = self._manifest({"a.py": hashlib.sha256(b"hello").hexdigest()})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest, pristine=False)
        self.assertFalse(ok)
        self.assertIn("sneaky.sh", detail)
        self.assertNotIn(".pgenv", detail.split("extra=")[1])

    def test_missing_declared_file_still_fails_in_non_pristine_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            (extract_dir / ".pgenv").write_text("PGPASSWORD=x\n")
            manifest = self._manifest({"a.py": "deadbeef" * 8})
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest, pristine=False)
        self.assertFalse(ok)
        self.assertIn("a.py", detail)

    def test_agent_guide_documented_sequence_passes_end_to_end(self):
        # Literally reproduces fcbc1fa1: extract, `cp .pgenv.example
        # .pgenv`, then run the same check the standalone `python3
        # smoke_test.py` (in-place, pristine=False) run makes.
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp)
            pgenv_example = extract_dir / ".pgenv.example"
            pgenv_example.write_text("PGPASSWORD=changeme\n")
            (extract_dir / "manifest.json").write_text("{}")
            (extract_dir / "01_dump.sql.gz").write_bytes(b"\x1f\x8b")
            manifest = self._manifest(
                {".pgenv.example": hashlib.sha256(pgenv_example.read_bytes()).hexdigest()},
            )
            # cp .pgenv.example .pgenv
            (extract_dir / ".pgenv").write_text(pgenv_example.read_text())
            ok, detail = bundled_files_check.check_bundled_files(extract_dir, manifest, pristine=False)
        self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()
