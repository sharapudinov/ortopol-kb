#!/usr/bin/env python3
"""The disk cache behind the three HTTP clients, as an object that can be
switched off the way the writers can.

DRY_RUN_WRITES_NOTHING (kb/CLAUDE.md) is kept by CONSTRUCTION for the graph
(citations/store.py) and for measurements (citations/spike_runs.py): the
mode picks a writer, and the promise holds because there is nothing in the
chosen object that writes. The response cache is the third channel the
invariant names -- paths.py argues at length that it is "data the tree
keeps, not a temp file" -- and it had no such seam at all: every client
mkdir'd its directory in __init__ and wrote every body it fetched,
identically in both modes.

Two implementations, chosen by one flag the caller already has:

  DiskCache      reads hits, writes misses, creates the directory. What a
                 real run wants: a wiped OpenAlex cache costs a day of
                 quota, and Math-Net starts timing out after a few dozen
                 rapid requests.
  ReadOnlyCache  serves the same hits and drops every write on the floor,
                 and does NOT create the directory -- a --dry-run against
                 a cache that does not exist must leave the tree exactly as
                 it found it, not leave an empty directory behind.

BOTH halves are the object's, and no client is handed a path. A seam that
answered "where would this live" and left the reading to the caller
encapsulated nothing that mattered: three clients each re-spelled the
is_file/read_text/count-the-hit sequence, and the next one could just as
easily write through the path it was given, past ReadOnlyCache. read() and
write() take a NAME (a url hash, a zbMATH id, a Math-Net id -- what is
cached is the client's business); the hit counter is the cache's, and each
client reports it as its own n_cache_hits.

cache_for(None) is None, which every client already understands as "no
cache".
"""
from __future__ import annotations

from pathlib import Path


class DiskCache:
    """Read-write: the ordinary run's cache."""

    read_only = False

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.hits = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def read(self, name: str, *, floor: int = 0) -> str | None:
        """The cached body for `name`, or None -- and a hit is counted here.

        `floor` is the Math-Net rule generalised: a body of `floor` bytes or
        fewer is not a hit. A truncated page must not stand in for the page,
        and at the default an empty file is a miss rather than something no
        client could parse anyway.
        """
        path = self.directory / name
        try:
            if not path.is_file() or path.stat().st_size <= floor:
                return None
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        self.hits += 1
        return text

    def write(self, name: str, text: str) -> None:
        (self.directory / name).write_text(text, encoding="utf-8")


class ReadOnlyCache(DiskCache):
    """Same hits, no writes, no directory creation."""

    read_only = True

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.hits = 0

    def write(self, name: str, text: str) -> None:
        return None


def cache_for(directory, *, read_only: bool = False) -> DiskCache | None:
    return None if directory is None else (
        ReadOnlyCache(directory) if read_only else DiskCache(directory))
