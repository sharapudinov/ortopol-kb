"""Every table and column of a classified schema is named, asked of the
live catalog before either dump writes a byte.

The public profile gets this for free: it builds its own COPY selects, so a
table outside the classification raises TableUnclassified
(corpus_cut.corpus_tables / citation_dump.citation_tables) and a column
outside it ColumnUnclassified (ColumnClasses.class_of), both before the
file is opened. The full profile is a bare pg_dump of whole schemas and
consulted no classification at all -- yet the checker that travels INSIDE
the artifact is profile-blind: column_class_checks.check_columns_are_
classified runs against every COPY block of schemas corpus and citation,
whichever profile produced it.

So a table or column added to either schema built cleanly under --profile
full, reported success, and then failed the package's own certification
afterwards -- the "an artifact that fails its own bundled certification,
after a build that reported success" failure MANIFEST_DESCRIBES_ARTIFACT
names, reintroduced on the profile that skipped the gate. The full profile
ships everything regardless of the verdict; "the map knows every table and
column" is not a decision about what travels, it is the precondition the
recipient will enforce, and it belongs on both paths at build time.

Which schemas are classified is column_class_checks.CLASSIFIED_SCHEMAS --
imported, not restated: the producer and the recipient have to be asking
about the same set, and that module is where the artifact side declares it.
A schema outside the set (measurements, and pg_dump's own public-schema
statements) is not this gate's business, exactly as it is not that check's.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from column_class_checks import CLASSIFIED_SCHEMAS  # noqa: E402
from schema_catalog import (  # noqa: E402
    classified_tables,
    columns_of,
    present_tables,
    schema_columns,
)


def require_classified_schemas(env: dict, schemas) -> None:
    """Raises TableUnclassified / ColumnUnclassified for the first thing in
    `schemas` the classification has no entry for.

    Reads the catalog once per classified schema, the way both plans do:
    one table list and one column list answer for the whole schema, where a
    per-table read costs a psql process each time round the loop.
    """
    for schema in schemas:
        classes = CLASSIFIED_SCHEMAS.get(schema)
        if classes is None:
            continue
        tables = classified_tables(present_tables(env, schema), classes.tables,
                                   schema, classes.hint)
        columns = schema_columns(env, schema)
        for table in tables:
            for column in columns_of(columns, table, schema):
                classes.class_of(table, column)
