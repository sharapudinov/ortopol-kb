r"""The one writer of a resolved COPY block, for either schema.

`COPY <schema>.<table> (cols) FROM stdin;`, the server-side COPY TO STDOUT
output streamed between that header and the `\.` terminator, then a
setval() per sequence-owning column -- the same shape pg_dump emits, so the
result is an ordinary psql-restorable dump with no special-case loader.

One writer rather than one per schema, and the schema comes IN with the
block (copy_plan.CopyBlock). Written twice -- once in public_dump.py with
`corpus.` in the header, once in citation_dump.py with `citation.` -- the
two bodies were identical but for that literal and the optional tally, and
kb/CLAUDE.md says what happens to the twin nobody edits: every other axis
of this dump (schema_catalog's catalog reads, restore_order and setval_sql,
column_classes, copy_rows, table_rows_check) already takes the schema as a
parameter precisely so that the two halves cannot answer differently.

The block arrives fully resolved: every question whose answer could be a
REFUSAL was asked by corpus_cut.plan_corpus() / citation_dump.
plan_citation() before `dst` was opened, because TableUnclassified and
ColumnUnclassified are not CommandFailed and would fly past the handler
that unlinks the partial file.

Rows are counted at the seam they pass through (copy_rows.RowCounter), not
asked of the database beside it: the manifest's numbers and the file's COPY
blocks have to be one fact, and two reads of a live instance are two (see
copy_rows.py). `tally` is that same seam asked for one column's census
instead of a total, and it is bound to THIS block's column list.
"""
from __future__ import annotations

from typing import IO

from block_census import FieldTally
from copy_plan import CopyBlock
from copy_rows import RowCounter
from pg_stream import stream_stdout
from schema_catalog import setval_sql

PSQL_ARGV = ["psql", "-v", "ON_ERROR_STOP=1", "--quiet", "--no-psqlrc", "-c"]


def write_copy_block(env: dict, dst: IO[bytes], block: CopyBlock,
                     tally: FieldTally | None = None) -> int:
    """Writes one block and returns how many rows went past.

    `block.serials` are the table's sequence-owning columns, and each gets
    the setval() schema_catalog builds for the block's own schema. Asked of
    the catalog rather than reasoned about per table: a sequence left at 1
    restores without complaint and hands the recipient's first insert an id
    already taken, which is a failure at their end rather than at ours.
    """
    dst.write(f"COPY {block.schema}.{block.table} ({', '.join(block.columns)}) "
              "FROM stdin;\n".encode())
    if tally is not None:
        tally.start(block.columns)
    counter = RowCounter(dst, tally)
    stream_stdout([*PSQL_ARGV, block.statement], env, counter)
    dst.write(b"\\.\n")
    for column in block.serials:
        dst.write(setval_sql(block.schema, block.table, column))
    dst.write(b"\n")
    return counter.rows
