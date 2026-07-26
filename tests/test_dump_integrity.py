"""Unit tests for dump_integrity.sha256_file: the single shared hashing
implementation build_package.py and smoke_test.py both rely on to detect a
corrupted/truncated dump before restore. Previously had zero coverage.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import dump_integrity


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


if __name__ == "__main__":
    unittest.main()
