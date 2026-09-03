"""Unit tests for deploy/dump_integrity.py: the shared hash, and the shared
comparison of a dump against the manifest that describes it.

One implementation, two readers -- smoke_test.py before it lets Postgres
restore anything, profile_checks.py as the last gate of the static pass a
recipient can run on the package alone. It used to be three inline
conditions in the Docker path only, so the standalone certifier read the
contents of whatever file sat at that path.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import dump_integrity
import profile_checks
from _artifact_fixtures import ArtifactBuilder
from manifest_keys import Key

GATE = "дамп — тот, что описан манифестом (размер, sha256)"


class Sha256FileTests(unittest.TestCase):
    def _write(self, data: bytes) -> Path:
        fd, name = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        return Path(name)

    def test_known_bytes_match_hashlib_oracle(self):
        data = b"the quick brown fox jumps over the lazy dog"
        path = self._write(data)
        self.assertEqual(dump_integrity.sha256_file(path), hashlib.sha256(data).hexdigest())

    def test_empty_file(self):
        path = self._write(b"")
        self.assertEqual(dump_integrity.sha256_file(path), hashlib.sha256(b"").hexdigest())

    def test_multi_chunk_file_with_uneven_remainder(self):
        # Exceeds the 1 MiB default CHUNK_SIZE by an amount that is NOT a
        # multiple of it, to pin the read loop against an off-by-one that
        # would silently drop the final partial chunk.
        data = bytes((i % 251) for i in range(dump_integrity.CHUNK_SIZE + 12_345))
        path = self._write(data)
        self.assertEqual(dump_integrity.sha256_file(path), hashlib.sha256(data).hexdigest())

    def test_custom_tiny_chunk_size_gives_the_same_digest(self):
        # A chunk_size far smaller than the data forces many loop
        # iterations; the digest must not depend on how the reads are
        # chunked.
        data = bytes((i % 251) for i in range(10_000))
        path = self._write(data)
        self.assertEqual(
            dump_integrity.sha256_file(path, chunk_size=7),
            hashlib.sha256(data).hexdigest(),
        )


class DumpMatchesManifestTests(unittest.TestCase):
    """Length and digest, against the block that names the file."""

    def _artifact(self) -> tuple[Path, dict]:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        dump = directory / "01_dump.sql.gz"
        dump.write_bytes(b"not really gzip, but bytes are bytes")
        manifest = {Key.DUMP: {Key.FILE: dump.name,
                               Key.BYTES: dump.stat().st_size,
                               Key.SHA256: dump_integrity.sha256_file(dump)}}
        return directory, manifest

    def test_the_declared_dump_passes(self):
        directory, manifest = self._artifact()
        ok, detail = dump_integrity.check_dump_matches_manifest(manifest, directory)
        self.assertTrue(ok, detail)
        self.assertIn("01_dump.sql.gz", detail)

    def test_one_flipped_byte_fails_on_the_digest(self):
        """The whole point of hashing a file whose length nothing changed:
        a package edited in place declares its own size correctly.
        """
        directory, manifest = self._artifact()
        dump = directory / manifest[Key.DUMP][Key.FILE]
        data = bytearray(dump.read_bytes())
        data[0] ^= 0x01
        dump.write_bytes(bytes(data))
        ok, detail = dump_integrity.check_dump_matches_manifest(manifest, directory)
        self.assertFalse(ok)
        self.assertIn("sha256", detail)

    def test_a_truncated_dump_fails_on_the_length_first(self):
        directory, manifest = self._artifact()
        dump = directory / manifest[Key.DUMP][Key.FILE]
        dump.write_bytes(dump.read_bytes()[:-5])
        ok, detail = dump_integrity.check_dump_matches_manifest(manifest, directory)
        self.assertFalse(ok)
        self.assertIn("bytes", detail)

    def test_a_missing_file_is_a_verdict_not_a_raise(self):
        directory, manifest = self._artifact()
        (directory / manifest[Key.DUMP][Key.FILE]).unlink()
        ok, detail = dump_integrity.check_dump_matches_manifest(manifest, directory)
        self.assertFalse(ok)
        self.assertIn("файла нет", detail)

    def test_a_block_that_names_nothing_is_a_verdict_too(self):
        directory, _manifest = self._artifact()
        for block in (None, "01_dump.sql.gz", {}, {Key.FILE: 7}):
            with self.subTest(block=block):
                ok, _detail = dump_integrity.check_dump_matches_manifest(
                    {Key.DUMP: block}, directory)
                self.assertFalse(ok)


class TheCertifierRefusesATamperedDumpTests(unittest.TestCase):
    """The gate in profile_checks.py, end to end on a built artifact: one
    byte of the dump changed, and the whole static certification stops at
    that row instead of reporting on contents nobody tied to the manifest.
    """

    def _results(self, directory: Path):
        return profile_checks.run_checks(directory)

    def test_a_clean_artifact_passes_the_gate_and_goes_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self._results(ArtifactBuilder(Path(tmp)).write())
        names = [name for name, _ok, _detail in results]
        self.assertIn(GATE, names)
        self.assertGreater(len(names), names.index(GATE) + 1)
        self.assertTrue(all(ok for _name, ok, _detail in results), results)

    def test_one_flipped_byte_stops_the_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = ArtifactBuilder(Path(tmp)).write()
            dump = directory / "01_dump.sql.gz"
            data = bytearray(dump.read_bytes())
            data[-1] ^= 0x01
            dump.write_bytes(bytes(data))
            results = self._results(directory)
        self.assertEqual([name for name, _ok, _detail in results][-1], GATE)
        self.assertFalse(results[-1][1])
        self.assertTrue(all(ok for _name, ok, _detail in results[:-1]))


if __name__ == "__main__":
    unittest.main()
