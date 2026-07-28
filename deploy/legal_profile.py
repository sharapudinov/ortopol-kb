"""Which documents may leave the machine, read from the database.

The legal boundary is DATA, not code: every corpus.documents row carries
legal_class, public_distribution and legal_note (the basis, one sentence),
set by the corpus owner from corpus/legal-regimes-dossier.md. This module
is the only place that knows what those values MEAN for packaging, and it
knows nothing about which document falls where -- no id lists, no filename
patterns. An earlier draft of the public profile hardcoded both (four
monograph ids, `LIKE '____\\_isu%'`, a year cutoff); that made the packager
the authority on a legal question it cannot answer, and made re-classifying
one document a code change.

Two profiles:

- FULL: every schema, every byte, including source blobs and full page
  text. NEVER published (see build_package.py's own docstring and README) --
  it exists so the corpus can be moved between the owner's own machines.
- PUBLIC: content is cut per public_distribution:
    full-text      -- everything (body, tsv, embedding, source_blob)
    metadata-only  -- the documents row (legal_class/legal_note included)
                      with NO source_blob, and pages carrying only
                      page_number + embedding, no body and therefore no tsv
    internal       -- our own curated metadata documents (INDEX/THEMES),
                      shipped whole: nobody else's copyright is involved
    excluded       -- nothing at all: no documents row, no pages, no
                      vectors. The three cuts above all publish SOMETHING
                      about a work (at minimum its bibliography and the
                      fact that we hold it); this value exists for the
                      documents whose regime the owner has not established,
                      where publishing even that much would be the packager
                      deciding an open legal question by default

Anything else -- NULL, or a value added to the database that this module
has not been taught -- FAILS the build (unclassified_documents(), called by
build_package.py before anything is written). Defaulting an unknown class
to "strip it" would silently ship less than intended; defaulting to "ship
it" would silently ship someone else's copyright. Refusing is the only
answer that cannot be wrong.
"""
from __future__ import annotations

import json

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from manifest_contract import Distribution, Key  # noqa: E402
from pg_common import run_sql, scalar  # noqa: E402

# The vocabulary itself lives in manifest_contract.Distribution, shared with
# the static verifier (profile_checks.py, which must not import anything
# Postgres-shaped). Its MEANING is this module's docstring; an unknown value
# never reaches a content decision at all -- require_classified() stops the
# build first.
KNOWN_DISTRIBUTIONS = Distribution.ALL
FULL_CONTENT_DISTRIBUTIONS = Distribution.FULL_CONTENT
SHIPPED_DISTRIBUTIONS = Distribution.SHIPPED

FIELD_SEP = "\x1f"

# Multi-valued fields (a class's distinct legal_note values) come back as
# JSON, not as a separator-joined string. The obvious separator choices are
# traps: psql output is split with str.splitlines(), which treats \x1c, \x1d,
# \x1e and \x85 as line boundaries too -- a chr(30)-joined string_agg was
# silently shredding one row into several here. json_agg escapes anything a
# note can contain, so this cannot recur.


def _sql_literals(values: tuple[str, ...]) -> str:
    """Module-owned constants only -- never caller input. Keeps the SQL
    predicates below derived from KNOWN_DISTRIBUTIONS/FULL_CONTENT_
    DISTRIBUTIONS/SHIPPED_DISTRIBUTIONS instead of restating them as a
    second literal list that can drift.
    """
    return ", ".join(f"'{value}'" for value in values)


# SQL booleans over corpus.documents, used unqualified inside queries that
# have exactly one documents row in scope (public_dump.py's COPY selects):
# SHIPPED_SQL decides whether a row exists in the public artifact at all,
# FULL_CONTENT_SQL decides how much of it that row carries.
FULL_CONTENT_SQL = f"public_distribution IN ({_sql_literals(FULL_CONTENT_DISTRIBUTIONS)})"
SHIPPED_SQL = f"public_distribution IN ({_sql_literals(SHIPPED_DISTRIBUTIONS)})"

# The manifest's verify_query for the classification: "is every document
# classified, with a value the packager understands". Recorded in the
# manifest verbatim so a reader re-asks the question against the data
# instead of trusting the number beside it.
UNCLASSIFIED_SQL = (
    "SELECT count(*) FROM corpus.documents WHERE legal_class IS NULL "
    "OR public_distribution IS NULL OR legal_note IS NULL "
    f"OR public_distribution NOT IN ({_sql_literals(KNOWN_DISTRIBUTIONS)});"
)


class Unclassified(RuntimeError):
    """A document the packager is not allowed to guess about."""


def unclassified_documents(env: dict) -> list[tuple[str, str, str]]:
    """Every document whose legal classification is missing or unknown, as
    (id, legal_class, public_distribution). Empty list is the only state in
    which a public artifact may be built.
    """
    rows = run_sql(
        env,
        "SELECT id, coalesce(legal_class, ''), coalesce(public_distribution, '') "
        "FROM corpus.documents "
        "WHERE legal_class IS NULL OR public_distribution IS NULL OR legal_note IS NULL "
        f"OR public_distribution NOT IN ({_sql_literals(KNOWN_DISTRIBUTIONS)}) ORDER BY id;",
        extra_args=["-t", "-A", "-F", FIELD_SEP],
    ).stdout
    offenders = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        doc_id, legal_class, distribution = line.split(FIELD_SEP)
        offenders.append((doc_id, legal_class, distribution))
    return offenders


def require_classified(env: dict) -> None:
    offenders = unclassified_documents(env)
    if offenders:
        listed = ", ".join(f"{i} (class={c!r}, distribution={d!r})" for i, c, d in offenders)
        raise Unclassified(
            f"{len(offenders)} document(s) with no usable legal classification: {listed} "
            "-- classify them in corpus.documents (see EXTENDING.md, principle 6) "
            "before building a public artifact; the packager must not guess"
        )


def documents_by_distribution(env: dict) -> dict[str, list[str]]:
    """{public_distribution: [document_id, ...]} over the whole corpus, one
    key per KNOWN_DISTRIBUTIONS value (empty list when unused). Ordered ids,
    so the manifest is stable across builds of unchanged data.
    """
    rows = run_sql(
        env,
        "SELECT coalesce(public_distribution, ''), id FROM corpus.documents ORDER BY 1, 2;",
        extra_args=["-t", "-A", "-F", FIELD_SEP],
    ).stdout
    grouped: dict[str, list[str]] = {name: [] for name in KNOWN_DISTRIBUTIONS}
    for line in rows.splitlines():
        if not line.strip():
            continue
        distribution, doc_id = line.split(FIELD_SEP)
        grouped.setdefault(distribution, []).append(doc_id)
    return grouped


def class_counts(env: dict) -> list[dict]:
    """Counts by (legal_class, public_distribution) plus that class's
    distinct legal_note values -- the "legal basis, one sentence per class"
    the artifact must carry, taken from the data rather than restated here.
    """
    rows = run_sql(
        env,
        "SELECT coalesce(legal_class, ''), coalesce(public_distribution, ''), count(*), "
        "json_agg(DISTINCT coalesce(legal_note, ''))::text "
        "FROM corpus.documents GROUP BY 1, 2 ORDER BY 1, 2;",
        extra_args=["-t", "-A", "-F", FIELD_SEP],
    ).stdout
    counts = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        legal_class, distribution, count, notes_json = line.split(FIELD_SEP)
        counts.append({
            "legal_class": legal_class,
            "public_distribution": distribution,
            "documents": int(count),
            "legal_basis": sorted(note for note in json.loads(notes_json) if note),
        })
    return counts


def shipped_ids(summary: dict) -> set[str]:
    """The ids a public artifact built from this summary actually carries.

    Pure, over legal_summary()'s own output rather than over the database:
    the callers that need it (manifest_probe, deciding whether a probe names
    a document the package contains) already hold the summary, and asking
    Postgres a second time could answer about a different moment.
    """
    grouped = summary.get(Key.DOCUMENTS_BY_DISTRIBUTION, {})
    return {
        doc_id
        for name in summary.get(Key.SHIPPED_DISTRIBUTIONS, ())
        for doc_id in grouped.get(name, [])
    }


def legal_summary(env: dict) -> dict:
    """The manifest's `legal` block: the aggregate, the id lists per
    distribution, and the verify query with its answer against THIS build's
    data.

    documents_by_distribution covers the whole corpus, excluded documents
    included -- the artifact must be able to say WHICH works it deliberately
    leaves out, not merely be silent about them. shipped_distributions is
    what separates the lists it carries from the lists it does not, and
    profile_checks.py reads exactly that to decide which ids it may find in
    the dump.
    """
    return {
        Key.VERIFY_QUERY: UNCLASSIFIED_SQL,
        Key.UNCLASSIFIED_DOCUMENTS: int(scalar(env, UNCLASSIFIED_SQL)),
        Key.CLASS_COUNTS: class_counts(env),
        Key.DOCUMENTS_BY_DISTRIBUTION: documents_by_distribution(env),
        Key.FULL_CONTENT_DISTRIBUTIONS: list(FULL_CONTENT_DISTRIBUTIONS),
        Key.SHIPPED_DISTRIBUTIONS: list(SHIPPED_DISTRIBUTIONS),
    }
