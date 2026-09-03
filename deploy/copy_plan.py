"""One table's COPY block, fully resolved: everything that can REFUSE is
answered before the output file is opened.

Both halves of the public dump ask the same two kinds of question before
they write anything -- what the catalog holds (schema_catalog.py) and what
the classification says about it (column_classes.py) -- and the second kind
can answer with TableUnclassified or ColumnUnclassified. Neither is a
CommandFailed, so asked from inside `with gzip.open(...)` they fly straight
past public_dump.py's handler, the one that unlinks the partial file, and
leave a truncated 01_dump.sql.gz on disk under a docstring promising the
build "refuses to write anything at all".

The shape lives here rather than in either dump because both build it:
corpus_cut.plan_corpus() and citation_dump.plan_citation() answer for their
own schema, and public_dump.py opens the file only once both have. One
declaration, so the newer and stricter half cannot acquire the polarity
without the structure that makes the polarity safe -- which is exactly how
the corpus half came to have the classification and not the plan.
"""
from __future__ import annotations

from typing import NamedTuple


class CopyBlock(NamedTuple):
    """What write_copy_block() needs and nothing it has to look up.

    `columns` is the catalog's list for the table (generated columns and
    whatever the dump leaves to the restore side already removed),
    `serials` the sequence-owning columns whose setval() follows the block,
    and `statement` the finished `COPY (SELECT ...) TO STDOUT` -- the one
    that carries the classification's verdict on every column.
    """

    table: str
    columns: list[str]
    serials: tuple[str, ...]
    statement: str
