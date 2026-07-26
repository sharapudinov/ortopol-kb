"""Streaming a Postgres client's stdout into an open file object, without
the two-pipe deadlock.

Extracted from artifact_bundle.dump_schemas when public_dump.py needed the
same thing for its COPY ... TO STDOUT selects: both stream potentially
multi-gigabyte output straight into one gzip file, and both talk to a child
that writes to stderr on its own schedule. Duplicating the thread-drained
stderr handling was the alternative -- exactly the kind of copy whose second
instance quietly loses the subtlety the first one was written for.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from typing import IO


class CommandFailed(RuntimeError):
    """Non-zero exit, carrying the child's stderr (which is otherwise lost:
    it was drained on a background thread, not left in the pipe).
    """


def stream_stdout(argv: list[str], env: dict, dst: IO[bytes]) -> None:
    """Runs argv and copies its stdout into dst, one pass, no intermediate
    file.

    stderr is drained on a background thread CONCURRENTLY with the stdout
    copy, not read afterward: pg_dump and psql both write NOTICE/WARNING
    lines to stderr on large jobs, and with both stdout and stderr as pipes,
    nothing reading stderr while the main thread blocks on stdout is a
    textbook deadlock -- once stderr fills the OS pipe buffer (~64KB on
    Linux) the child blocks writing to it and the whole transfer stalls
    forever, well before EOF on stdout.
    """
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()))
    stderr_thread.start()
    try:
        shutil.copyfileobj(proc.stdout, dst)
    finally:
        proc.stdout.close()
        stderr_thread.join()
        proc.stderr.close()
        returncode = proc.wait()
    if returncode != 0:
        stderr = b"".join(chunk for chunk in stderr_chunks if chunk)
        raise CommandFailed(
            f"{argv[0]} failed (exit {returncode}): {stderr.decode(errors='replace').strip()}"
        )
