"""Every table a manifest declares is in the dump with exactly that many
rows -- asked of BOTH classified schemas, from one engine.

A check that finds no COPY block for a table has no way to tell "the cut
removed it correctly" from "it never shipped", and reports a green nought
either way. manifest.<schema>.table_rows is what closes that: the packager
names every table it wrote and how many rows each got (manifest_rows.py,
counted off the writing), and this asks the file both directions -- nothing
declared is missing, nothing shipped is undeclared.

Split out of citation_cut_checks.py because the question is the schema's,
not either schema's: it landed on citation first (the journal, the most
delicately cut table in the package, was the one nobody could ask about),
and the corpus half then had the same hole one level up -- documents and
pages are described by two headline counts and everything else by nothing.
Parameterised rather than duplicated: two copies of a both-directions
comparison agree only by accident, and the polarity ("an empty declaration
under a shipping schema is a refusal") is exactly the kind that decays in
the copy nobody edits.

WHICH schemas are asked is column_class_checks.CLASSIFIED_SCHEMAS through
the two maps that own their own name -- the same set whose columns the
recipient holds to the classification, since a schema whose columns are
named is a schema whose tables can be. WHETHER the artifact carries a
schema at all is manifest_contract.required_schemas(), the same rule the
producer resolved by, not the manifest's own list of schemas: on this side
of an unsigned manifest a self-declaration cannot excuse a check.
"""
from __future__ import annotations

from citation_columns import CITATION
from column_class_checks import CLASSIFIED_SCHEMAS
from corpus_columns import CORPUS
from manifest_keys import Key
from manifest_contract import required_schemas

# Which manifest block each schema's {table: rows} declaration sits in.
# One declaration, because the producer stamps it (manifest_rows.py) and
# the recipient reads it, and a manifest key spelled on both sides of the
# artifact boundary agrees with itself only while someone remembers both.
#
# WHICH schemas get one is not a second declaration: the set is
# column_class_checks.CLASSIFIED_SCHEMAS, the same one whose columns the
# recipient holds to the classification, and the mapping is derived
# through it. So a schema classified there and forgotten here is a
# KeyError at import -- a package that cannot be certified at all --
# rather than a table count nobody asks for, which is the shape this
# module exists to refuse.
_BLOCK = {CORPUS.schema: Key.CORPUS, CITATION.schema: Key.CITATION}
DECLARATION_BLOCK = {schema: _BLOCK[schema] for schema in CLASSIFIED_SCHEMAS}
# The schemas this check runs for, in a fixed order so a report reads the
# same way twice.
DECLARED_SCHEMAS = tuple(sorted(DECLARATION_BLOCK))


def declared_rows(manifest: dict, schema: str):
    """manifest.<schema>.table_rows, raw -- absent, empty and malformed all
    reach the check as themselves, because each of the three is a different
    verdict there and none of them is a default this reader may supply.
    """
    block = manifest.get(DECLARATION_BLOCK[schema])
    return block.get(Key.TABLE_ROWS) if isinstance(block, dict) else None


def check_every_declared_table_shipped(manifest: dict, scans: dict,
                                       schema: str) -> tuple[bool, str]:
    """Every table manifest.<schema>.table_rows names is in the dump with
    exactly that many COPY rows -- and no table of that schema is in the
    dump the manifest does not name.

    Both directions, and an empty declaration under a schema this artifact
    carries is a failure rather than a quiet pass: a manifest that names no
    table is one this reader cannot hold to anything. A schema the artifact
    does not carry must declare nothing and ship nothing, which is the same
    sentence with both sides empty.
    """
    declared = declared_rows(manifest, schema)
    shipped = {name for name in scans if name.startswith(f"{schema}.")}
    required, problem = required_schemas(manifest)
    if required is None:
        return False, (f"по манифесту нельзя вывести, везёт ли пакет схему {schema}: "
                       f"{problem}")
    if schema not in required:
        ok = not declared and not shipped
        return ok, (f"схема {schema} этим пакетом не везётся; "
                    f"declared {sorted(declared or [])}, in dump {sorted(shipped)}")
    if not isinstance(declared, dict) or not declared:
        return False, (
            f"manifest.{DECLARATION_BLOCK[schema]}.{Key.TABLE_ROWS} пуст ({declared!r}) "
            f"при схеме, которую пакет везёт: держать дамп не к чему -- пересоберите "
            "пакет текущим сборщиком"
        )
    problems = []
    for table, want in sorted(declared.items()):
        qualified = f"{schema}.{table}"
        scan = scans.get(qualified)
        if scan is None:
            problems.append(f"{qualified}: блока COPY нет, а манифест обещает {want}")
        elif scan.rows != want:
            problems.append(f"{qualified}: {scan.rows} строк против {want}")
    for name in sorted(shipped - {f"{schema}.{table}" for table in declared}):
        problems.append(f"{name}: в дампе есть, в манифесте не назван")
    return not problems, (
        f"{len(declared)} declared table(s): " + ("; ".join(problems) or "все на месте, "
        + ", ".join(f"{table}={rows}" for table, rows in sorted(declared.items())))
    )
