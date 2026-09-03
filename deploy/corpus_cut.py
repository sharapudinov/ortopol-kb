"""Which rows of schema corpus travel, and how each column is projected.

The corpus counterpart of the pair the citation half already has: what the
cut IS lives apart from the file that applies it (citation_profile.py's
predicates and citation_dump.py's maps against public_dump.py). Three maps
and two projections, all of them per-schema knowledge, split out of
public_dump.py by responsibility and by kb/CLAUDE.md FILE_SIZE -- that
module assembles the file; this one answers what belongs in it.

The legal knowledge itself is elsewhere again and stays there: WHICH
documents may travel is legal_profile.py (LEGAL_IS_DATA -- no id list, no
filename pattern, no year), and WHICH columns are content is
corpus_columns.py. Here the two meet the catalog.

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

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import corpus_columns  # noqa: E402
import schema_catalog  # noqa: E402
from copy_plan import CopyBlock  # noqa: E402
from legal_profile import FULL_CONTENT_SQL, SHIPPED_SQL  # noqa: E402

SCHEMA = "corpus"

# pages.id is a BIGSERIAL nothing references (probes address a page by
# document_id + page_number). Omitting it from the COPY lets the sequence
# assign ids on restore, in the deterministic order of the ORDER BY below.
# The sequence is then repositioned explicitly like every other one in the
# dump (public_dump.write_copy_block's serials), so nothing about the
# restore rests on
# this exclusion: a corpus table whose id DOES travel gets the same
# treatment, which is what the exclusion used to be standing in for.
PAGES_EXCLUDED = ("id",)

# One alias per dumped table, so select_expression() never has to guess.
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
    # select_expression() and for the WHERE here. The class lives on the
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

# A table is classified only if ALL THREE maps know it: what of it may
# leave (corpus_columns), which alias its projection uses, and which rows it
# contributes. Any one missing is the same silence -- the discipline the
# citation half already followed (citation_dump.CLASSIFIED), applied here
# now that this schema has a column classification of its own.
CLASSIFIED = set(corpus_columns.CORPUS_COLUMN_CLASS) & set(TABLE_ALIASES) & set(_SOURCE)

_UNCLASSIFIED_HINT = ("дополните CORPUS_COLUMN_CLASS "
                      "(deploy/corpus_columns.py), TABLE_ALIASES и _SOURCE "
                      "(deploy/corpus_cut.py)")

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


def select_expression(table: str, column: str) -> str:
    """The expression written into the COPY select for one column: the
    column itself, except where the legal filter replaces its value.

    WHICH columns those are is corpus_columns.py, the same map the
    artifact-side checker holds the shipped bytes to -- not a pair of
    hardcoded names here. Two hardcoded branches over a catalog-driven
    column list ended in a fall-through, i.e. a denylist: a column added to
    corpus.documents and forgotten SHIPPED, whatever it carried. Every
    column is classified now, and an unclassified one raises
    ColumnUnclassified and stops the build (UNCLASSIFIED_FAILS_BUILD).
    """
    alias = TABLE_ALIASES[table]
    withheld = corpus_columns.withheld_value(table, column)
    if withheld is None:
        return f"{alias}.{column}"
    return (f"CASE WHEN {FULL_CONTENT_SQL} THEN {alias}.{column} "
            f"ELSE {withheld} END AS {column}")


def copy_select(table: str, columns: list[str]) -> str:
    projection = ",\n       ".join(select_expression(table, c) for c in columns)
    return f"COPY (SELECT {projection}\n{_SOURCE[table]}) TO STDOUT"


def plan_corpus(env: dict) -> tuple[CopyBlock, ...]:
    """Every catalog and classification question this schema's blocks ask,
    answered before the caller opens its output.

    The citation half's plan_citation() with the same reason and the same
    shape (copy_plan.CopyBlock): an unclassified table raises
    TableUnclassified and an unclassified column ColumnUnclassified, neither
    of them a CommandFailed, so asked from inside the open gzip they fly
    past the handler that unlinks the partial file. This schema is written
    FIRST, so what such a refusal leaves behind is the preamble, the whole
    DDL and however many COPY blocks came before it -- a truncated dump on
    disk under a docstring promising nothing was written at all.

    The table list is asked FIRST because it is the answer that can be the
    refusal; the two column reads are for the whole schema at once, since
    one read answers for every table where a per-table read costs a psql
    process each time round the loop.
    """
    tables = corpus_tables(env)
    columns = schema_catalog.schema_columns(env, SCHEMA)
    serials = schema_catalog.schema_serial_columns(env, SCHEMA)
    blocks = []
    for table in tables:
        table_columns = schema_catalog.columns_of(
            columns, table, SCHEMA, exclude=EXCLUDED_COLUMNS.get(table, ()))
        blocks.append(CopyBlock(table, table_columns,
                                tuple(serials.get(table, ())),
                                copy_select(table, table_columns)))
    return tuple(blocks)
