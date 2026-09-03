"""The PUBLIC profile's dump: same schema, legally filtered content.

Why not pg_dump: pg_dump has no row- or cell-level filter, so the only
pg_dump-shaped options were (a) build temporary tables/schema in the live
database and dump those -- which writes to the live instance and renames the
schema the whole toolchain reads (corpus -> something else), or (b) dump
everything and post-process a multi-gigabyte SQL stream with sed. Both are
worse than the third option, which this module takes:

    pg_dump --schema-only --schema=corpus   (real DDL, unmodified)
  + COPY (SELECT ...) TO STDOUT             (the rows we may ship)

The result restores into schema `corpus` exactly as the full artifact does
-- same table names, same generated columns, same indexes -- so every
consumer (pg_search.py, smoke_checks.py, AGENT_GUIDE.md's own recipes) works
against it unchanged. measurements is simply not dumped: our own research
records ship as their own artifact, not folded into a corpus package.

What the filter does, per corpus.documents.public_distribution (see
legal_profile.py -- this module contains no legal knowledge and no id list):

  full-text / internal -- every column, blob included; pages keep body
                          (and therefore tsv, which is GENERATED from it)
  metadata-only        -- the documents row WITHOUT source_blob; pages keep
                          page_number + embedding, and body is written as
                          the empty string
  excluded             -- no row is written at all, in either table: the
                          COPY selects filter the document out (SHIPPED_SQL)
                          instead of blanking its columns

The difference between the last two is the whole reason `excluded` exists.
Blanking columns still ships a row saying "this work is in our corpus, here
is its bibliography"; for a document whose legal regime the owner has not
established, that sentence is itself the decision the packager must not
make. Filtering happens in the SELECT, so nothing about the document reaches
the dump file even in a form a reader could count.

body='' rather than "no page row" is deliberate and load-bearing: the pages
row still exists, so the page's vector is still searchable (semantic search
finds the DOCUMENT, the reader then goes to the publisher's site for the
text), the page count still matches the source, and corpus.pages.body's NOT
NULL survives. Empty body means to_tsvector('russian','') -- an empty tsv,
i.e. no fulltext content, which profile_checks.py verifies rather than
assumes.

Column lists are read from the catalog, not typed here: when the legal
classification columns were added to corpus.documents, a hardcoded list
would have silently dropped legal_class/public_distribution/legal_note from
the public artifact -- the one artifact whose whole point is carrying that
information. Only generated columns (recomputed on restore) and pages.id
(the sequence reassigns it, see PAGES_EXCLUDED) are left out.

The table list is read the same way, through the same module the citation
dump reads (schema_catalog.py). It was a hand-typed map and a hand-ordered
sequence of calls, which agreed with pg_schema.sql by nothing but
attention: a fourth corpus table would have shipped its DDL with no COPY
block, and profile_checks.check_schemas compares SCHEMAS, not tables, so
nothing downstream contradicts an empty one. What stays hand-written is
the per-schema knowledge -- which alias a table's projection uses and
which rows it contributes -- and an unclassified table is a refusal.

So is the third catalog question, WHICH columns own a sequence: every one
of them gets a setval() after its COPY block. The corpus half used to emit
none, and was correct only because the single BIGSERIAL it has is the one
PAGES_EXCLUDED happens to drop from the COPY -- hand-kept knowledge, and
knowledge nothing checks, since a table only has to appear in TABLE_ALIASES
and _SOURCE to build. A sequence left at 1 restores cleanly and collides on
the recipient's first insert.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import citation_dump  # noqa: E402
import schema_catalog  # noqa: E402
from artifact_bundle import DUMP_COMPRESSLEVEL  # noqa: E402
from manifest_contract import Profile, base_schemas_for  # noqa: E402
from legal_profile import FULL_CONTENT_SQL, SHIPPED_SQL, require_classified  # noqa: E402
from pg_stream import CommandFailed, stream_stdout  # noqa: E402

SCHEMA = "corpus"

# What _dump_ddl() asks pg_dump for: the public profile's schemas before
# the citation mode is applied, because citation_dump.dump_citation()
# appends that schema's DDL itself, under the mode. Asked of
# manifest_contract as THAT question rather than derived by naming a mode
# that ships nothing: a policy value is not a mechanism, and the two sides
# of the invariant were aligned only by accident -- pg_dump emitting the
# citation DDL here as well would put duplicated CREATE statements into one
# file, which aborts the restore and which no manifest check catches
# (profile_checks compares schema NAMES, not statements).
PUBLIC_SCHEMAS = base_schemas_for(Profile.PUBLIC)

# pages.id is a BIGSERIAL nothing references (probes address a page by
# document_id + page_number). Omitting it from the COPY lets the sequence
# assign ids on restore, in the deterministic order of the ORDER BY below.
# The sequence is then repositioned explicitly like every other one in the
# dump (write_copy_block's serials), so nothing about the restore rests on
# this exclusion: a corpus table whose id DOES travel gets the same
# treatment, which is what the exclusion used to be standing in for.
PAGES_EXCLUDED = ("id",)

# One alias per dumped table, so _select_expression() never has to guess.
TABLE_ALIASES = {"documents": "d", "pages": "p", "embedding_model": "m"}

# Which rows each table contributes, keyed exactly as citation_dump._SOURCE
# is and for the same reason: there is no else branch to fall into. A table
# reached through an `else` shipped `FROM corpus.<table> ORDER BY id` -- every
# row, no legal predicate, no join to corpus.documents -- so the next corpus
# table carrying per-document rows was made to build by one line in
# TABLE_ALIASES and then shipped rows belonging to excluded documents. The
# refusal is the only answer a packager may give to a table nobody classified.
_SOURCE = {
    "documents": f"FROM corpus.documents d WHERE {SHIPPED_SQL} ORDER BY d.id",
    # The join supplies public_distribution twice over: for the body CASE in
    # _select_expression() and for the WHERE here. The class lives on the
    # document, both the cut and the omission apply to its pages -- a page of
    # an excluded document is dropped with it, so the artifact cannot carry
    # an orphan vector for text it does not ship.
    "pages": ("FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id "
              f"WHERE {SHIPPED_SQL} "
              "ORDER BY p.document_id, p.page_number"),
    # Which model every vector above was computed with: one row, naming no
    # document and carrying no third-party text.
    "embedding_model": "FROM corpus.embedding_model m ORDER BY m.id",
}

# A table is classified only if BOTH maps know it: which alias its projection
# uses and which rows it contributes. Either one missing is the same silence.
CLASSIFIED = set(TABLE_ALIASES) & set(_SOURCE)

_UNCLASSIFIED_HINT = ("дополните TABLE_ALIASES и _SOURCE "
                      "(deploy/public_dump.py)")

# Columns the dump leaves to the restore side, per table.
EXCLUDED_COLUMNS = {"pages": PAGES_EXCLUDED}


def corpus_tables(env: dict) -> list[str]:
    """The tables to dump: the catalog's list, held to the classification
    and put in the order a restore needs (corpus.pages references
    corpus.documents, and pg_constraint is where that is written down).
    """
    present = schema_catalog.classified_tables(
        schema_catalog.present_tables(env, SCHEMA), CLASSIFIED,
        SCHEMA, _UNCLASSIFIED_HINT)
    return schema_catalog.restore_order(
        present, schema_catalog.foreign_key_edges(env, SCHEMA), SCHEMA)


def _select_expression(table: str, column: str) -> str:
    """The expression written into the COPY select for one column: the
    column itself, except where the legal filter replaces its value.
    """
    alias = TABLE_ALIASES[table]
    if table == "documents" and column == "source_blob":
        return f"CASE WHEN {FULL_CONTENT_SQL} THEN {alias}.source_blob END AS source_blob"
    if table == "pages" and column == "body":
        return f"CASE WHEN {FULL_CONTENT_SQL} THEN {alias}.body ELSE '' END AS body"
    return f"{alias}.{column}"


def _copy_select(table: str, columns: list[str]) -> str:
    projection = ",\n       ".join(_select_expression(table, c) for c in columns)
    return f"COPY (SELECT {projection}\n{_SOURCE[table]}) TO STDOUT"


def write_copy_block(env: dict, dst: IO[bytes], table: str, columns: list[str],
                     serials=()) -> None:
    """Writes one `COPY corpus.<table> (cols) FROM stdin;` block, streaming
    the server-side COPY TO STDOUT output between the header and the `\\.`
    terminator -- the same shape pg_dump emits, so the result is an ordinary
    psql-restorable dump with no special-case loader.

    `serials` are the table's sequence-owning columns, and each gets the
    setval() the citation half has always written (schema_catalog.
    setval_sql). Asked of the catalog rather than reasoned about per table:
    a sequence left at 1 restores without complaint and hands the
    recipient's first insert an id already taken, which is a failure at
    their end rather than at ours.
    """
    header = f"COPY corpus.{table} ({', '.join(columns)}) FROM stdin;\n"
    dst.write(header.encode())
    argv = ["psql", "-v", "ON_ERROR_STOP=1", "--quiet", "--no-psqlrc",
            "-c", _copy_select(table, columns)]
    stream_stdout(argv, env, dst)
    dst.write(b"\\.\n")
    for column in serials:
        dst.write(schema_catalog.setval_sql(SCHEMA, table, column))
    dst.write(b"\n")


def _dump_ddl(env: dict, dst: IO[bytes]) -> None:
    schema_args = [f"--schema={name}" for name in PUBLIC_SCHEMAS]
    stream_stdout(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges", "--no-tablespaces",
         *schema_args, "--exclude-schema=citation_graph", "--exclude-schema=ag_catalog"],
        env, dst,
    )


PREAMBLE = (
    "--\n"
    "-- ortopol knowledge base, PUBLIC profile.\n"
    "--\n"
    "-- Schema: real pg_dump --schema-only output for schema corpus (measurements\n"
    "-- is deliberately absent). Data: COPY blocks filtered by\n"
    "-- corpus.documents.public_distribution -- documents classified\n"
    "-- 'metadata-only' carry no source_blob and their pages carry an empty body\n"
    "-- (and therefore an empty tsv), only page_number and embedding; documents\n"
    "-- classified 'excluded' have no row here at all, in either table. See\n"
    "-- manifest.json's `legal` block for the classification this build used\n"
    "-- (documents_by_distribution names every document, shipped_distributions\n"
    "-- says which of those lists this file carries) and deploy/public_dump.py\n"
    "-- for how it was applied. Schema `citation`, when present, is cut per\n"
    "-- manifest.json's `citation` block instead (deploy/citation_profile.py /\n"
    "-- deploy/citation_dump.py) -- a separate policy, decided once for the\n"
    "-- whole crawl record rather than per document. The per-document cut\n"
    "-- still reaches it: a citation.work row naming a document absent from\n"
    "-- this file is absent too, and so are its edges and its journal rows.\n"
    "--\n\n"
)


def dump_public(env: dict, gz_path: Path, *, citation_mode: str) -> dict[str, int]:
    """Writes the filtered dump to gz_path, gzip-streamed in one pass, and
    returns {table: rows} for every citation table it carried.

    The counts go back to the caller rather than being re-derived for the
    manifest: they are read from the plan this call dumped by, so the
    numbers manifest.json declares and the COPY blocks the file holds are
    one resolution of one cut (MANIFEST_DESCRIBES_ARTIFACT), and the
    recipient's check is what compares them against the shipped bytes.

    Refuses to write anything at all while any document lacks a usable
    classification (legal_profile.require_classified) -- that must stop the
    build, not be quietly assigned a default.

    citation_mode is the value build_package.main() resolved once
    (citation_profile.resolve_citation_mode) and handed to the manifest as
    well; this module does not re-read the policy, so the dump and the
    manifest cannot describe two different cuts of the same schema. It has
    no default for the same reason gather_manifest()'s has none: an omitted
    mode would be a cut nobody chose, and CitationMode.NONE happens to be
    the cut that ships nothing.
    """
    require_classified(env)
    tables = corpus_tables(env)
    # Both catalog questions are asked once for the whole schema, as the
    # citation half asks them: one read answers for every table, and a
    # per-table read costs a psql process each time round the loop.
    columns = schema_catalog.schema_columns(env, SCHEMA)
    serials = schema_catalog.schema_serial_columns(env, SCHEMA)
    # The citation half is resolved here too, BEFORE the file is opened.
    # Its refusals -- TableUnclassified, ColumnUnclassified -- are neither
    # CommandFailed, so asked from inside the gzip context they went past
    # the handler below and left a truncated dump on disk, after the whole
    # corpus DDL and every corpus COPY block had already been written. The
    # promise in this docstring is that nothing is written at all, and a
    # refusal keeps that promise only while it is still cheap.
    citation_plan = citation_dump.plan_citation(env, citation_mode)
    # Counted here for the same reason the plan is resolved here: before the
    # file exists, so a failing read is a build that wrote nothing.
    citation_rows = citation_dump.plan_row_counts(env, citation_plan)
    try:
        with gzip.open(gz_path, "wb", compresslevel=DUMP_COMPRESSLEVEL) as dst:
            dst.write(PREAMBLE.encode())
            _dump_ddl(env, dst)
            dst.write(b"\n")
            for table in tables:
                write_copy_block(env, dst, table, schema_catalog.columns_of(
                    columns, table, SCHEMA, exclude=EXCLUDED_COLUMNS.get(table, ())),
                    serials=serials.get(table, ()))
            dst.write(b"\n")
            citation_dump.dump_citation(env, dst, citation_plan)
    except CommandFailed as exc:
        gz_path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    return citation_rows
