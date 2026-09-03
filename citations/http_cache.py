#!/usr/bin/env python3
"""The disk cache behind the HTTP clients, as an object that can be
switched off the way the writers can.

DRY_RUN_WRITES_NOTHING (kb/CLAUDE.md) is kept by CONSTRUCTION for the graph
(citations/store.py) and for measurements (citations/spike_runs.py): the
mode picks a writer, and the promise holds because there is nothing in the
chosen object that writes. The response cache is the third channel the
invariant names -- paths.py argues at length that it is "data the tree
keeps, not a temp file" -- and it had no such seam at all: every client
mkdir'd its directory in __init__ and wrote every body it fetched,
identically in both modes.

One contract, `Cache`, and two implementations of it:

  DiskCache      reads hits, writes misses, creates the directory. What a
                 real run wants: a wiped OpenAlex cache costs a day of
                 quota, and Math-Net starts timing out after a few dozen
                 rapid requests.
  ReadOnlyCache  serves the same hits and drops every write on the floor,
                 and does NOT create the directory -- a --dry-run against
                 a cache that does not exist must leave the tree exactly as
                 it found it, not leave an empty directory behind.

A Protocol and two independent classes, exactly as store.Writer is written
and for the same reason: ReadOnlyCache used to BE a DiskCache with write()
overridden, so "writes nothing" was inherited-by-default -- any write-capable
member added to the parent would be live on the read-only object until
somebody remembered to override it too. Now the read-only object has no
write in it to begin with, and the shared reading is two module-level
functions both classes call rather than a base class either of them extends.

BOTH halves are the object's, and no client is handed a path. A seam that
answered "where would this live" and left the reading to the caller
encapsulated nothing that mattered: three clients each re-spelled the
is_file/read_text/count-the-hit sequence, and the next one could just as
easily write through the path it was given, past ReadOnlyCache -- which is
precisely what happened to the batch sidecars (citations/hub_cache.py).

read() and write() take a NAME (a url hash, a zbMATH id, a Math-Net id --
what is cached is the client's business); the hit counter is the cache's,
and each client reports it as its own n_cache_hits. names() is for the
reader that has no name to ask for: what the cache HOLDS, which is a
question about the directory and therefore the cache's too. `limit` reads
only the head of an entry -- a cached OpenAlex batch page runs to tens of
megabytes and one classifying field lives in its first hundred bytes.

cache_for(None) is None, which every client already understands as "no
cache".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """What a cached-response reader may rely on, in both modes.

    read_only is the object saying which it is, for a caller that has to
    report the mode rather than choose behaviour by it: choosing is what the
    two implementations are for.
    """

    read_only: bool
    hits: int
    # WHERE the cache is -- for saying which one, never for reading through:
    # every read and every write is a method above, and a client that took
    # this path to open a file itself is the sidecar bug all over again. A
    # refusal that cannot name the directory it means ("there is no cache of
    # answers here") sends the operator looking for it.
    directory: Path

    def names(self) -> list[str]: ...

    def read(self, name: str, *, floor: int = 0,
             limit: int | None = None) -> str | None: ...

    def write(self, name: str, text: str) -> None: ...


def _names(directory: Path) -> list[str]:
    """Entry names in the cache directory; [] if there is no directory.

    A cache that was never written is empty, not an error: the read-only
    mode meets exactly that, and iterdir() on a missing directory raises.
    """
    try:
        return sorted(path.name for path in directory.iterdir() if path.is_file())
    except OSError:
        return []


def _read(directory: Path, name: str, floor: int, limit: int | None) -> str | None:
    """The cached body for `name`, or None -- the read both objects share.

    `floor` is the Math-Net rule generalised: a body of `floor` bytes or
    fewer is not a hit. A truncated page must not stand in for the page, and
    at the default an empty file is a miss rather than something no client
    could parse anyway.
    """
    path = directory / name
    try:
        if not path.is_file() or path.stat().st_size <= floor:
            return None
        with path.open(encoding="utf-8") as handle:
            return handle.read() if limit is None else handle.read(limit)
    except OSError:
        return None


class DiskCache:
    """Read-write: the ordinary run's cache."""

    read_only = False

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.hits = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return _names(self.directory)

    def read(self, name: str, *, floor: int = 0, limit: int | None = None) -> str | None:
        text = _read(self.directory, name, floor, limit)
        if text is not None:
            self.hits += 1
        return text

    def write(self, name: str, text: str) -> None:
        """The whole entry or none of it: a temporary neighbour, renamed.

        A cached batch page runs to tens of megabytes, so a process killed
        while one is being written dies in the MIDDLE of it, and a plain
        write leaves the stump behind under the entry's own name. The stump
        is not empty, so _read() serves it as a hit and every later run
        reads a body that stops mid-token. os.replace is atomic within one
        filesystem, which a neighbour in the same directory guarantees; the
        name is `.part`-suffixed rather than `.json`, so a stump this
        process cannot clean up (its own kill) is not a cache entry either.
        """
        temporary = self.directory / f".{name}.{os.getpid()}.part"
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.directory / name)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class ReadOnlyCache:
    """Same hits, no writes, no directory creation."""

    read_only = True

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.hits = 0

    def names(self) -> list[str]:
        return _names(self.directory)

    def read(self, name: str, *, floor: int = 0, limit: int | None = None) -> str | None:
        text = _read(self.directory, name, floor, limit)
        if text is not None:
            self.hits += 1
        return text

    def write(self, name: str, text: str) -> None:
        return None


def cache_for(directory, *, read_only: bool = False) -> Cache | None:
    return None if directory is None else (
        ReadOnlyCache(directory) if read_only else DiskCache(directory))
