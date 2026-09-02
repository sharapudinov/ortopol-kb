"""What schema citation actually holds, read from the catalog -- and the
refusal when the code has nothing to say about part of it.

Split off citation_dump.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that module knows HOW a table is written into the dump, this
one answers WHICH tables there are, which of their columns travel, which
columns own a sequence and what order a restore needs. Every one of those
is a question about the database, and every one of them used to have at
least a partial answer hardcoded beside the dump -- which is how a table
added to the schema later could ship its DDL with no COPY block, and how a
new BIGSERIAL id could ship with its sequence left at 1.

Nothing here classifies anything: the maps and the refusal live with the
dump (citation_dump.classified_tables, citation_columns), and the answers
below are only what the database says. That is the same polarity
table_columns() always had, now applied to the two facts that were still
declared by hand.
"""
from __future__ import annotations

from pg_common import FIELD_SEP, run_sql


# Every ordinary table the schema actually holds. 'r' and 'p' only:
# views, sequences and indexes are recreated by the DDL and carry no rows
# of their own to copy. Ordered by oid, i.e. by the order the schema files
# created them in: restore_order() only has to move a table that a foreign
# key puts out of place, and everything else keeps an order a human wrote.
_TABLES_SQL = """
SELECT c.relname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'citation' AND c.relkind IN ('r', 'p')
ORDER BY c.oid;
"""

# Which table must be restored before which. Only keys INSIDE schema
# citation: citation.work references corpus.documents, and that slice is
# written by public_dump.py before this one is called at all.
_FOREIGN_KEYS_SQL = """
SELECT c.relname, f.relname
FROM pg_constraint co
JOIN pg_class c ON c.oid = co.conrelid
JOIN pg_class f ON f.oid = co.confrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_namespace nf ON nf.oid = f.relnamespace
WHERE co.contype = 'f' AND n.nspname = 'citation' AND nf.nspname = 'citation'
  AND c.relname <> f.relname
ORDER BY c.relname, f.relname;
"""

# Which columns own a sequence, and therefore need a setval() after their
# COPY block. Asked of the catalog rather than kept as a list of table
# names: a table added later with a BIGSERIAL id passes every
# classification check there is, and a forgotten sequence is not a failed
# restore but a successful one that hands the next crawl an id already
# taken.
#
# Both column questions are asked of the WHOLE schema at once, and the
# answer carries (relname, attname). pg_attribute holds every table's
# columns in one relation, so a per-table WHERE turned one catalog read
# into one psql process, one temp script and one connection PER TABLE, on
# a loop that grows with every table the schema gains -- the cost
# pg_graph_common.py's own docstring prices. The per-table helpers below
# stay, as lookups over the map.
_SERIAL_COLUMNS_SQL = """
SELECT c.relname, a.attname
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'citation' AND c.relkind IN ('r', 'p')
  AND a.attnum > 0 AND NOT a.attisdropped
  AND pg_get_serial_sequence('citation.' || c.relname, a.attname) IS NOT NULL
ORDER BY c.relname, a.attnum;
"""

_COLUMNS_SQL = """
SELECT c.relname, a.attname
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'citation' AND c.relkind IN ('r', 'p')
  AND a.attnum > 0 AND NOT a.attisdropped AND a.attgenerated = ''
ORDER BY c.relname, a.attnum;
"""


def restore_order(present: list[str], edges) -> list[str]:
    """`present` reordered so a table follows every table it references.

    The restore replays COPY blocks against live foreign keys, so the order
    is the keys' answer and pg_constraint is where it is written down. A
    declared tuple answered the same question for the schema as it stood,
    and answered it silently wrong for the next table added.

    Tables no key relates keep the order they arrived in (the catalog's, so
    the order the schema files declare them in). A cycle is refused rather
    than guessed at: there is no order that restores it, and pretending
    otherwise produces an artifact that fails at the recipient's end.
    """
    waiting_for = {name: {parent for child, parent in edges
                          if child == name and parent in present}
                   for name in present}
    ordered: list[str] = []
    remaining = list(present)
    while remaining:
        # One at a time and always the earliest ready one, so a table the
        # keys say nothing about does not overtake its neighbours.
        nxt = next((name for name in remaining
                    if not waiting_for[name] - set(ordered)), None)
        if nxt is None:
            raise RuntimeError(
                "foreign keys inside schema citation form a cycle over "
                + ", ".join(sorted(remaining)) + " -- no restore order exists")
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def foreign_key_edges(env: dict) -> list[tuple[str, str]]:
    """(child, parent) for every foreign key inside schema citation."""
    rows = run_sql(env, _FOREIGN_KEYS_SQL, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    seen = []
    for line in rows.splitlines():
        if line.strip():
            child, parent = line.split(FIELD_SEP)
            if (child, parent) not in seen:
                seen.append((child, parent))
    return seen


def _by_table(env: dict, sql: str) -> dict[str, list[str]]:
    """{table: [column, ...]} out of one (relname, attname) catalog read."""
    rows = run_sql(env, sql, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    grouped: dict[str, list[str]] = {}
    for line in rows.splitlines():
        if line.strip():
            table, column = line.split(FIELD_SEP)
            grouped.setdefault(table.strip(), []).append(column.strip())
    return grouped


def schema_columns(env: dict) -> dict[str, list[str]]:
    """Every dumpable column of every table in schema citation, in one read."""
    return _by_table(env, _COLUMNS_SQL)


def schema_serial_columns(env: dict) -> dict[str, list[str]]:
    """Every sequence-owning column of the schema, in one read."""
    return _by_table(env, _SERIAL_COLUMNS_SQL)


def columns_of(columns: dict[str, list[str]], table: str) -> list[str]:
    """`table`'s dumpable columns out of a schema-wide read, or a refusal.

    The guard is here rather than at the read so that both the one-table
    and the whole-schema caller get it: an empty list means the name does
    not match anything the catalog holds, and a COPY block with no columns
    is not something to write and find out about at the recipient's end.
    """
    found = columns.get(table) or []
    if not found:
        raise RuntimeError(f"citation.{table} has no dumpable columns -- wrong table name?")
    return found


def serial_columns(env: dict, table: str) -> list[str]:
    """The columns of `table` that own a sequence, per the catalog."""
    return schema_serial_columns(env).get(table, [])


def present_tables(env: dict) -> list[str]:
    """Every ordinary table schema citation holds, in creation order."""
    rows = run_sql(env, _TABLES_SQL, extra_args=["-t", "-A"]).stdout
    present = [line.strip() for line in rows.splitlines() if line.strip()]
    if not present:
        raise RuntimeError("schema citation carries no tables -- wrong database?")
    return present


def table_columns(env: dict, table: str) -> list[str]:
    return columns_of(schema_columns(env), table)
