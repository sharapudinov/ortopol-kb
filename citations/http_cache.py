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

cache_for(None) is None, which every client already understands as "no
cache": the read-through variant is the only new behaviour here.
"""
from __future__ import annotations

from pathlib import Path


class DiskCache:
    """Read-write: the ordinary run's cache."""

    read_only = False

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """Where `name` lives. Each client spells its own name (a url hash,
        a zbMATH id, a Math-Net id): what is cached is the client's
        business, that it is written at all is this object's.
        """
        return self.directory / name

    def write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")


class ReadOnlyCache(DiskCache):
    """Same hits, no writes, no directory creation."""

    read_only = True

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def write(self, path: Path, text: str) -> None:
        return None


def cache_for(directory, *, read_only: bool = False) -> DiskCache | None:
    return None if directory is None else (
        ReadOnlyCache(directory) if read_only else DiskCache(directory))
