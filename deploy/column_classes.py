"""What a dumped column IS -- topology, or content a cut must withhold --
asked the same way for every schema the public dump writes.

One engine, two declarations: citation_columns.py names every column of
schema citation, corpus_columns.py every column of schema corpus. Both
travel INSIDE the artifact beside the checker that re-answers "does this
dump match the cut it declares" without a database or this repository, which
is why the classification is a module rather than a section of the dump
writer: a second, hand-written copy on the recipient's side could only agree
with the producer by accident, and catching exactly that disagreement is
what the checker exists for.

The polarity is the whole design. schema_catalog.schema_columns() reads the
column list from pg_attribute, so no column can silently vanish from the
artifact -- but a cut written as a hardcoded denylist SHIPS anything nobody
named in it. citation.work.embedding walked straight through that gap, and
corpus.documents.source_blob/corpus.pages.body were the same two hardcoded
names one schema over. Here every column is named: an unclassified one
raises ColumnUnclassified and fails the build, the same answer
legal_profile.require_classified gives an unclassified document
(kb/CLAUDE.md UNCLASSIFIED_FAILS_BUILD). Neither shipping nor withholding
by default is available, because both are decisions.

What REPLACES a withheld value is per schema and lives with the
declaration, not here: schema citation blanks a content column to a typed
NULL for the whole dump, schema corpus keeps the row and substitutes per
document (a NULL blob, an empty body) under the legal predicate. The engine
only insists that a content column HAS a replacement -- one without is as
unbuildable as one without a class.
"""
from __future__ import annotations

TOPOLOGY = "topology"
CONTENT = "content"


class ColumnUnclassified(RuntimeError):
    """A dumped column nobody has said topology or content about -- or a
    content column with nothing declared to stand in for its value."""


class ColumnClasses:
    """One schema's column classification, and the questions asked of it.

    `classes` is {table: {column: TOPOLOGY | CONTENT}} and `withheld` is
    {(table, column): SQL expression}, both spelled by the schema's own
    module. `hint` and `withheld_hint` are what a refusal tells the person
    who has to fix it -- which map, in which file.
    """

    def __init__(self, schema: str, classes: dict[str, dict[str, str]],
                 withheld: dict[tuple[str, str], str], *, hint: str,
                 withheld_hint: str):
        self.schema = schema
        self.classes = classes
        self.withheld = withheld
        self.hint = hint
        self.withheld_hint = withheld_hint

    @property
    def tables(self) -> set[str]:
        return set(self.classes)

    def class_of(self, table: str, column: str) -> str:
        try:
            return self.classes[table][column]
        except KeyError:
            raise ColumnUnclassified(
                f"колонка {self.schema}.{table}.{column} не классифицирована "
                f"(topology | content) -- {self.hint}; сборка отказывается "
                f"угадывать, уезжает ли новая колонка в public-артефакт"
            ) from None

    def content_columns(self, table: str) -> tuple[str, ...]:
        """The columns a cut withholds in `table`, in declaration order."""
        return tuple(column for column, kind in self.classes[table].items()
                     if kind == CONTENT)

    def withheld_value(self, table: str, column: str) -> str | None:
        """The SQL that stands in for this column's value, or None to ship
        it as it is.

        Raises for a column nobody classified, and for a content column with
        no replacement declared -- either way the build stops instead of
        guessing.
        """
        if self.class_of(table, column) == TOPOLOGY:
            return None
        try:
            return self.withheld[(table, column)]
        except KeyError:
            raise ColumnUnclassified(
                f"{self.schema}.{table}.{column} — content, но без замены "
                f"значения в {self.withheld_hint}"
            ) from None
