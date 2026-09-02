"""Shared helpers for talking to the corpus Postgres instance via the psql CLI.

No psycopg driver is installed in this environment (checked, not assumed).
Rather than add a new pip dependency silently to an installer other people
are meant to run on their own machines, every Postgres interaction in this
repository goes through the psql command-line client, which any Postgres
installation already provides.

Three mechanisms, all avoiding shell string interpolation of untrusted data:

- Query parameters go through psql script variables (`-v name=value`,
  referenced in SQL as `:'name'`), which only work when the SQL is read as
  a *script* (`-f file`), not via `-c`. run_sql() therefore always writes
  the SQL to a temp file first.
- Bulk loads go through `\\copy ... FROM 'tempfile' WITH (FORMAT csv)`,
  with the CSV built by Python's csv module (correct quoting of embedded
  commas/quotes/newlines) and written to its own temp file. That direction
  has its own module, pg_copy.py, and runs through run_sql() like
  everything else.
- Where neither fits -- a LIST of values inside one statement, for which
  psql has no binding form at all -- sql_literal() builds the quoted
  literal. One implementation, so no call site invents an f-string.

Reading psql's answer is the same kind of shared plumbing, so the row and
field separators (FIELD_SEP/RECORD_SEP/ROW_ARGS/split_records) live here
too, and every parser in the repository imports them rather than spelling
the bytes again.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class PostgresUnavailable(RuntimeError):
    pass


# How a multi-row psql result is delimited, and the flags that produce it.
# Here because it is the same kind of plumbing run_sql() is -- every reader
# in this repository parses psql's output, and a title, a reason or a page
# of PDF text can contain a comma, a tab and a newline, so the separators
# are the ASCII unit/record ones no source text carries. \x1e is
# deliberately NOT run through str.splitlines() -- that treats \x1c-\x1e and
# \x85 as line boundaries too, which is how a multi-line title used to
# arrive as several rows. One declaration and one splitter: a second copy
# drifts the moment one caller's separator changes.
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
ROW_ARGS = ["-t", "-A", "-F", FIELD_SEP, "-R", RECORD_SEP]


def split_records(stdout: str) -> list[str]:
    return [r.strip("\n") for r in stdout.split(RECORD_SEP) if r.strip("\n")]


def load_pgenv(pgenv_path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .pgenv file into a psql-compatible environment.

    Callers must not print or log the returned dict; it carries PGPASSWORD.
    """
    if not pgenv_path.is_file():
        raise PostgresUnavailable(f"credentials file not found: {pgenv_path}")
    env = os.environ.copy()
    for line in pgenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _run(env: dict[str, str], args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["psql", "-v", "ON_ERROR_STOP=1", "--quiet", *args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"psql failed ({' '.join(args)}): {result.stderr.strip()}")
    return result


def run_sql_file(
    env: dict[str, str],
    path: Path,
    variables: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a .sql file as a script; `variables` are available as :'name'."""
    args = list(extra_args or [])
    for key, value in (variables or {}).items():
        args += ["-v", f"{key}={value}"]
    args += ["-f", str(path)]
    return _run(env, args)


def run_sql(
    env: dict[str, str],
    sql: str,
    variables: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run an inline SQL string as a script (supports :'name' interpolation)."""
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False, encoding="utf-8") as f:
        f.write(sql)
        script_path = Path(f.name)
    try:
        return run_sql_file(env, script_path, variables, extra_args)
    finally:
        script_path.unlink(missing_ok=True)


def sql_literal(value) -> str:
    """One value as a SQL string literal, safe to splice into a statement.

    For the cases psql's own `-v name=value` / :'name' binding cannot reach:
    it substitutes ONE value, so a list of keys inside a single statement
    (an IN (...) list, a Cypher WHERE ... IN [...]) has no bound form. The
    alternative every such call site would otherwise reach for is an
    f-string, and this repository's data is third-party text.

    E'' with backslash escaping rather than plain quote-doubling: E'' means
    the same thing whatever standard_conforming_strings is set to, while a
    plain '...' literal changes meaning with it. Backslashes are doubled
    FIRST -- a backslash inserted while escaping a quote must not be doubled
    afterwards, the same ordering citation.cypher_literal() documents.

    A NUL byte cannot appear in a Postgres text value at all, and letting it
    through would truncate the statement rather than fail it.
    """
    text = str(value)
    if "\x00" in text:
        raise ValueError("NUL byte cannot appear in a SQL string literal")
    return "E'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def scalar(
    env: dict[str, str],
    sql: str,
    variables: dict[str, str] | None = None,
) -> str:
    """Run a SQL script expected to return exactly one scalar value.

    Previously reimplemented identically (`run_sql(...).stdout.strip()` with
    `-t -A`) in three separate deploy modules -- smoke_checks.py,
    blob_integrity_checks.py and manifest_probe.py -- each a candidate to
    drift from the other two the moment the `-t -A` convention or error
    handling needed to change. This is the one copy all three now import.
    """
    return run_sql(env, sql, variables=variables, extra_args=["-t", "-A"]).stdout.strip()


def scalar_row(
    env: dict[str, str],
    sql: str,
    variables: dict[str, str] | None = None,
    field_sep: str = FIELD_SEP,
    expected_columns: int | None = None,
) -> list[str]:
    """Run a SQL script expected to return exactly one row and split its
    columns on field_sep. Saves callers that need more than one scalar
    value (e.g. `model, dims`) from either a second round trip or building
    the concatenation themselves in SQL.

    expected_columns, when given, turns a missing/malformed row into a
    RuntimeError naming the query and its variables, instead of leaving the
    caller to unpack too few values and hit a bare, contextless ValueError.
    """
    out = run_sql(
        env, sql, variables=variables, extra_args=["-t", "-A", "-F", field_sep]
    ).stdout.strip()
    row = out.split(field_sep) if out else []
    if expected_columns is not None and len(row) != expected_columns:
        where = f"variables={variables!r}" if variables else "no variables"
        raise RuntimeError(
            f"expected {expected_columns} column(s), got {len(row)} ({row!r}); "
            f"query={sql.strip()[:200]!r}, {where}"
        )
    return row


def row_or_none(
    env: dict[str, str],
    sql: str,
    variables: dict[str, str] | None,
    expected_columns: int,
    what: str,
) -> list[str] | None:
    """scalar_row(), but tolerant of the query legitimately matching zero
    rows -- a real, expected outcome for e.g. "the model/dims row hasn't
    been populated yet" or "this page has no embedding", not a corruption.
    expected_columns therefore cannot be handed to scalar_row directly: it
    would turn "no such row" into the same RuntimeError as "wrong number of
    columns", collapsing two call sites that need different handling (a
    missing row is often a normal fallback path; a malformed one is always
    a bug) into one.

    None on a genuinely empty result; a shape-checked row otherwise, so a
    future column added to/removed from the caller's SQL text fails loudly
    here instead of as a bare, contextless ValueError at the unpacking call
    site.
    """
    row = scalar_row(env, sql, variables=variables)
    if not row:
        return None
    if len(row) != expected_columns:
        raise RuntimeError(
            f"{what}: expected {expected_columns} column(s), got {len(row)} ({row!r})"
        )
    return row


def check_postgres_available(env: dict[str, str]) -> bool:
    try:
        run_sql(env, "SELECT 1;")
        return True
    except Exception:
        return False
