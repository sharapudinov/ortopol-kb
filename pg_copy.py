"""Bulk loads into Postgres: many rows at once, through a temp CSV file.

Split off pg_common.py by responsibility (and by kb/CLAUDE.md FILE_SIZE):
that module is the psql invocation itself and the parsing of what comes
back, this one is the direction no statement fits -- thousands of rows,
quoted by the csv module and never interpolated into SQL, because the text
loaded here is third-party (page bodies, titles, abstracts).

Two forms, and the second is not a convenience over the first:

  copy_csv_into  takes CSV already built as a string. What the corpus
                 loaders hand over: a document's pages are built once and
                 are small beside the PDF they came from.
  copy_csv_rows  takes an ITERABLE of rows and writes them one at a time,
                 and can carry staging DDL and the upsert that consumes the
                 staged rows in the SAME script. What the crawl needs: its
                 batches are the largest in the repository, and its staging
                 relation has to be private to the one session.
"""
from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from pg_common import run_sql


def copy_csv_into(env: dict[str, str], table_columns: str, csv_text: str) -> subprocess.CompletedProcess:
    """COPY CSV text into `table_columns`, e.g. "documents (id, filename)"."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = Path(f.name)
    try:
        escaped_path = str(csv_path).replace("'", "''")
        return run_sql(
            env, f"\\copy {table_columns} FROM '{escaped_path}' WITH (FORMAT csv)\n")
    finally:
        csv_path.unlink(missing_ok=True)


class CopyResult(NamedTuple):
    """How many rows were streamed out, and what the script printed.

    `rows` is what the caller handed over; `stdout` is what its epilogue
    said about them -- which is not the same number whenever the upsert
    filters (a self-edge) or skips (an edge already known).
    """

    rows: int
    stdout: str

    def accepted(self) -> int:
        """The count an epilogue ending in `SELECT count(*) FROM ...`
        printed: rows the DATABASE took, not rows submitted.

        The last non-empty line, that statement being the last in the
        script. Read here rather than by the caller because the script and
        its output are one contract, and this output is a number of the
        caller's own making -- unlike a data read, which is parsed with
        split_records() precisely because a value can contain a newline.
        """
        lines = [line for line in self.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(
                "the copy script printed no count -- did its epilogue lose "
                "the closing SELECT?")
        return int(lines[-1])


def copy_csv_rows(
    env: dict[str, str],
    table_columns: str,
    rows,
    *,
    preamble: str = "",
    epilogue: str = "",
) -> CopyResult:
    """Stream `rows` into `table_columns` as CSV, optionally staging and
    consuming them in the same breath.

    Two things copy_csv_into() cannot do, both paid for by the crawl:

    - the rows are written to the file ONE AT A TIME, so peak memory follows
      one row rather than the level. Building the whole CSV as a string and
      handing it over held the batch twice over, and the batches here are
      the biggest in the repository (a depth-2 journal is ~100k rows; a work
      row carries every raw source record it came from).
    - `preamble` and `epilogue` travel in the SAME script as the \\copy, so
      staging DDL, the load and the upsert that reads it are one psql
      session and -- under --single-transaction -- one transaction. That is
      what lets the staging relation be a TEMP table: private to this
      session, gone at COMMIT, and impossible for a concurrent writer to
      observe, clobber or drop halfway through.

    The quoting is the csv module's, identical to what copy_csv_into()
    receives from its callers, because it is the same writer.
    """
    written = 0
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8", newline="",
    ) as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in rows:
            writer.writerow(row)
            written += 1
        csv_path = Path(f.name)
    try:
        escaped_path = str(csv_path).replace("'", "''")
        script = (
            f"{preamble}"
            f"\\copy {table_columns} FROM '{escaped_path}' WITH (FORMAT csv)\n"
            f"{epilogue}"
        )
        result = run_sql(env, script,
                         extra_args=["-t", "-A", "--single-transaction"])
    finally:
        csv_path.unlink(missing_ok=True)
    return CopyResult(written, result.stdout)
