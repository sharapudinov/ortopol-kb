"""The PUBLIC profile's citation-schema slice of the dump: the same pg_dump
--schema-only + hand-rolled COPY pattern public_dump.py uses for schema
corpus (see that module's own docstring for why), applied to schema citation
and governed by citation_profile.py's mode instead of a per-row legal class.

Split into its own module purely for size (kb/CLAUDE.md FILE_SIZE), not for
a different design: public_dump.dump_public() decides WHETHER and under
which mode to call dump_citation(); this module only knows HOW to write one
mode's worth of citation.* DDL+COPY. Like public_dump.py itself, this is
build-time-only and deliberately NOT bundled into the artifact (see
artifact_bundle.DEPLOY_FILES's own comment on legal_profile.py/
public_dump.py) -- it reads and cuts the live database, which is the
builder's job, not the recipient's.

What the legal cut removes (citation_profile.shipped_work_sql /
shipped_crawl_step_sql, applied in _SOURCE below): a work row whose
document_id names a document the public artifact does not carry leaves with
that document -- its row, every edge touching it, and every journal row that
names it. Not a second policy invented here: it is corpus.documents.
public_distribution, the same column public_dump.py cuts by, reaching the
rows that reference it across a foreign key.

WHICH tables travel is the catalog's answer, not a tuple's: pg_dump
--schema-only emits DDL for every table in schema citation, so a table
added later and forgotten here would ship as a correctly-created, silently
EMPTY one. citation_tables() reads pg_class (schema_catalog.py, the one
engine both dumps ask) and refuses to build when a relation is not in the
classification -- the same polarity, and the same refusal, that
citation_columns.py applies per column.

id is preserved (never re-sequenced the way corpus.pages.id is excluded and
left to the restore-side sequence): citation.cites references
citation.work.id BY VALUE, and a column-list COPY has no id-remapping
mechanism, so both the id and the BIGSERIAL sequence position must survive
the round trip. WHICH columns need that fix-up is the catalog's answer too
(schema_serial_columns: pg_get_serial_sequence per column), and so is the ORDER
the blocks are written in (restore_order over pg_constraint) -- both were
hand-kept lists beside a classification guard that did not check them, so a
table added later would have shipped in an order nothing verified and with
its sequence left at 1.
"""
from __future__ import annotations

from typing import IO, NamedTuple

from citation_columns import (
    CENSUS_COLUMN,
    CENSUS_TABLE,
    CITATION_COLUMN_CLASS,
    blanked_value,
)
from block_census import FieldTally
from copy_plan import CopyBlock
from copy_writer import write_copy_block
from schema_catalog import (
    classified_tables,
    columns_of,
    foreign_key_edges,
    present_tables,
    restore_order,
    schema_columns,
    schema_serial_columns,
)
from citation_profile import crawl_step_cut_ctes, shipped_crawl_step_sql, shipped_work_sql
from manifest_contract import ships_citation, strips_content
from pg_stream import stream_stdout

SCHEMA = "citation"

# One alias per dumped table (the same discipline corpus_cut.TABLE_ALIASES
# follows): an unlisted table raises KeyError instead of quietly producing
# SQL with the wrong alias.
TABLE_ALIASES = {"work": "w", "cites": "c", "crawl_step": "s", "public_policy": "p",
                 "schema_backfill": "b"}

# Every row that names a document is cut by that document's own legal class
# as well as by the schema-wide mode -- citation_profile.py's predicates say
# why and how. This module ships only the public profile (public_dump.py is
# its only caller; the full profile goes through pg_dump and applies no cut
# at all), so the cut is unconditional here rather than a mode.
_SOURCE = {
    "work": "FROM citation.work w WHERE {work} ORDER BY w.id",
    "cites": ("FROM citation.cites c "
              "JOIN citation.work wa ON wa.id = c.citing "
              "JOIN citation.work wb ON wb.id = c.cited "
              "WHERE {citing} AND {cited} ORDER BY c.citing, c.cited, c.source"),
    "crawl_step": "FROM citation.crawl_step s WHERE {step} ORDER BY s.id",
    # Our own decision record about the crawl as a whole, naming no document
    # and no third party: it ships whole under any shipping mode.
    "public_policy": "FROM citation.public_policy p ORDER BY p.id",
    # Which one-time parses this schema has already run. Ships for the same
    # reason it exists: a restored database that does not carry the record
    # would re-scan the whole journal the first time the schema is applied
    # to it, looking for prose the parse can no longer find.
    "schema_backfill": "FROM citation.schema_backfill b ORDER BY b.name",
}

# A table is classified here only if all three maps know it: what may
# leave (citation_columns), which alias its projection uses, and which rows
# it contributes. Any one of them missing is the same silence.
CLASSIFIED = (set(CITATION_COLUMN_CLASS) & set(TABLE_ALIASES) & set(_SOURCE))

_UNCLASSIFIED_HINT = ("дополните CITATION_COLUMN_CLASS "
                      "(deploy/citation_columns.py), TABLE_ALIASES и _SOURCE "
                      "(deploy/citation_dump.py)")


def citation_tables(env: dict) -> list[str]:
    """The tables to dump: the catalog's list, held to the classification
    and put in the order a restore needs."""
    present = classified_tables(present_tables(env, SCHEMA), CLASSIFIED,
                                SCHEMA, _UNCLASSIFIED_HINT)
    return restore_order(present, foreign_key_edges(env, SCHEMA), SCHEMA)


def _select_expression(table: str, column: str, mode: str) -> str:
    """One column's projection under `mode`.

    Every column is classified, in every mode -- citation_columns.
    blanked_value() raises on one that is not, so a column added to the
    schema and forgotten here stops the build instead of shipping by
    default. The catalog is what says a column exists (schema_columns), the
    classification is what says whether it may leave; both are consulted
    for every column, which is the point.

    The MODE is read with the same polarity: content survives only under a
    mode manifest_contract declares full-content (strips_content), so a
    mode this build has never heard of blanks rather than ships. Keyed on
    `== TOPOLOGY_ONLY` it was the other way round, and the declared
    authority was never consulted at all.
    """
    withheld = blanked_value(table, column)
    if withheld and strips_content(mode):
        return f"{withheld} AS {column}"
    return f"{TABLE_ALIASES[table]}.{column}"


def _source_clause(table: str) -> str:
    return _SOURCE[table].format(
        work=shipped_work_sql("w"),
        citing=shipped_work_sql("wa"),
        cited=shipped_work_sql("wb"),
        step=shipped_crawl_step_sql("s"),
    )


# The journal cut is membership in two CTEs, so the statement carries them
# in front of its SELECT (citation_profile.crawl_step_cut_ctes: derived once
# per statement, not once per row). Keyed by table, so a table with no
# prefix cannot silently acquire one.
_QUERY_PREFIX = {"crawl_step": crawl_step_cut_ctes}


def copy_select(table: str, columns: list[str], mode: str) -> str:
    projection = ",\n       ".join(_select_expression(table, c, mode) for c in columns)
    prefix = _QUERY_PREFIX[table]() if table in _QUERY_PREFIX else ""
    return f"COPY ({prefix}SELECT {projection}\n{_source_clause(table)}) TO STDOUT"


class CitationPlan(NamedTuple):
    """What this schema contributes to a dump, decided before it is opened.

    `ships` is manifest_contract.ships_citation() over the build's mode, and
    it is part of the plan rather than re-asked at write time: a schema that
    does not travel has no blocks AND no DDL, and those are one decision.
    """

    ships: bool
    blocks: tuple[CopyBlock, ...]


def plan_citation(env: dict, mode: str) -> CitationPlan:
    """Every catalog and classification question the citation dump asks,
    answered up front.

    Resolved BEFORE the caller opens its output, because the answers can be
    a refusal: an unclassified table raises TableUnclassified
    (schema_catalog) and an unclassified column ColumnUnclassified
    (citation_columns), and this schema is written LAST, after the whole
    corpus DDL and every corpus COPY block. Asked from inside the open file,
    those refusals fired past public_dump.py's `except CommandFailed` -- a
    different class entirely -- and left a truncated 01_dump.sql.gz on disk
    under a module whose docstring promises it "refuses to write anything at
    all".
    """
    if not ships_citation(mode):
        return CitationPlan(False, ())
    tables = citation_tables(env)
    columns = schema_columns(env, SCHEMA)
    serials = schema_serial_columns(env, SCHEMA)
    blocks = []
    for table in tables:
        table_columns = columns_of(columns, table, SCHEMA)
        blocks.append(CopyBlock(SCHEMA, table, table_columns,
                                tuple(serials.get(table, ())),
                                copy_select(table, table_columns, mode)))
    return CitationPlan(True, tuple(blocks))


def dump_ddl(env: dict, dst: IO[bytes]) -> None:
    stream_stdout(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges", "--no-tablespaces",
         "--schema=citation", "--exclude-schema=citation_graph", "--exclude-schema=ag_catalog"],
        env, dst,
    )


class CitationWritten(NamedTuple):
    """What dump_citation() put in the file: the rows per table, and the
    census of the one column the manifest describes by value rather than by
    total (citation_columns.CENSUS_TABLE/CENSUS_COLUMN).
    """

    tables: dict[str, int]
    work_by_kind: dict[str, int]


def dump_citation(env: dict, dst: IO[bytes], plan: CitationPlan) -> CitationWritten:
    """Writes the citation schema's DDL + every table's COPY block, or
    nothing at all under a plan that does not ship it, and returns what it
    wrote: {table: rows} and the kind census.

    The census comes off the same stream as the counts and for the same
    reason (MANIFEST_DESCRIBES_ARTIFACT): read from the live database
    beside the dump, it described whatever the instance held at that
    moment, and the crawl writes ~100k journal rows per pass against it.

    Takes the PLAN rather than the mode: every question whose answer could
    be a refusal is plan_citation()'s, asked before the caller opened `dst`.
    Whether the schema ships is manifest_contract.ships_citation(), the same
    predicate schemas_for() reads to decide whether manifest.json declares
    it -- one authority, so the bytes and their description cannot disagree.
    Keyed on `== NONE` it was a denylist beside an allowlist:
    CitationMode.ALL grows with the database's own vocabulary (it is
    inherited, deliberately), SHIPPED is hand-written, and a fourth mode
    would have been omitted from the manifest and written into the dump
    anyway -- third-party titles in a package whose manifest denies
    carrying them.
    """
    if not plan.ships:
        return CitationWritten({}, {})
    dump_ddl(env, dst)
    dst.write(b"\n")
    census = FieldTally(CENSUS_COLUMN)
    tables = {block.table: write_copy_block(
        env, dst, block, census if block.table == CENSUS_TABLE else None)
        for block in plan.blocks}
    return CitationWritten(tables, dict(census.counts))
