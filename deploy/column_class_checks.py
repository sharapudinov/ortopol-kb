"""Every column the dump actually ships is one the classification names.

The polarity column_classes.py is built on -- a cut written as a denylist
ships whatever nobody named, so every column is named and an unclassified
one is a refusal -- was implemented on the PRODUCER side only
(citation_dump._select_expression, public_dump._select_expression, both
raising ColumnUnclassified). This module is the same question asked of the
finished file, from inside the artifact.

That is not redundancy. manifest.json is unsigned and profile_checks.py
travels in the package precisely so a recipient can certify it without
trusting its producer (kb/CLAUDE.md ARTIFACT_SIDE_FAILS_CLOSED). A dump
carrying a corpus or citation column that appears in no classification map
-- a hand-edited file, an older or divergent builder, a column added to the
schema between the build and this reading -- was certified [OK] on every
row, because every bundled check iterated only the columns the maps already
knew (content_columns()), and the one column-shape assertion on the corpus
half was a single hardcoded name: the two-name denylist the invariant names
as the original defect, relocated from the dumper into the verifier.

Both schemas inherit the predicate from the one engine
(ColumnClasses.unknown_columns), so the corpus half and the citation half
cannot answer this differently. A table absent from its schema's map counts
as unclassified WHOLE: it has no visitor on the scan either, so nothing else
in the package looks at it at all.

Schemas nobody classified are not this module's business: the full profile
carries measurements (and pg_dump's own public-schema statements), and
which schemas may travel at all is check_schemas() one line above.
"""
from __future__ import annotations

from citation_columns import CITATION
from corpus_columns import CORPUS

# The schemas whose columns are classified, keyed by the name a COPY header
# spells. Both maps travel in the artifact beside this module.
CLASSIFIED_SCHEMAS = {CITATION.schema: CITATION, CORPUS.schema: CORPUS}


def check_columns_are_classified(scans: dict) -> tuple[bool, str]:
    """No COPY block of a classified schema carries an unnamed column.

    `scans` is dump_scan.DumpContents.tables -- the COPY column list the
    file itself declares, per table.
    """
    unknown_tables: list[str] = []
    unknown_columns: list[str] = []
    checked = 0
    for key, scan in sorted(scans.items()):
        classes = CLASSIFIED_SCHEMAS.get(scan.schema)
        if classes is None:
            continue
        checked += 1
        if not classes.knows(scan.table):
            unknown_tables.append(key)
            continue
        unknown_columns += [f"{key}.{column}"
                            for column in classes.unknown_columns(scan.table, scan.columns)]
    ok = not unknown_tables and not unknown_columns
    return ok, (
        f"{checked} COPY-блок(ов) схем {sorted(CLASSIFIED_SCHEMAS)}; "
        f"таблиц вне классификации: {unknown_tables or 'нет'}, "
        f"колонок вне классификации: {unknown_columns or 'нет'}"
    )
