"""Streaming sha256 for the (potentially multi-gigabyte) dump artifact.

The dump embeds every source PDF/djvu blob in the corpus, so it is expected
to be hundreds of MB to several GB. Both sides of the deploy pipeline hash
it: build_package.py after writing it, smoke_test.py again after extracting
it and before letting Postgres restore it. Neither side may read the whole
file into memory to do so.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
