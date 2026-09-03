"""Static verification of the citation-schema slice of a dump, split out of
profile_checks.py purely for size (kb/CLAUDE.md FILE_SIZE) -- same
DUMP-BYTES-ONLY discipline as that module's own docstring: what
manifest.legal is to corpus.documents, manifest.citation is to
citation.work/cites, and this module is the reader that holds the dump to
that claim.

Checks:

  content is stripped     no row of any dumped table carries a non-empty
                           value in a column citation_columns classifies as
                           content, under every mode outside
                           CitationMode.FULL_CONTENT, an unrecognised one
                           included. The column list is IMPORTED from
                           citation_columns, never restated here: a second
                           copy of the classification could only agree with
                           the producer by accident, and the one column it
                           forgot was the whole finding that put this module
                           and the dump on one map

The counts, and the three checks that hold the citation cut to the DOCUMENT
cut, read the containers this module fills but ask a different question of
them, and live next door in citation_cut_checks.py (module size).

profile_checks.py owns the single streaming pass over the dump (dump_scan.
scan reads the whole file once); attach_visitors() only adds this module's
row callbacks to that same pass instead of asking for a second one.

The provenance question -- WHOSE decision the mode was -- reads no dump
byte and lives in citation_policy_check.py.
"""
from __future__ import annotations

from typing import NamedTuple

from citation_columns import (
    CENSUS_COLUMN,
    CITATION_COLUMN_CLASS,
    JOURNAL_KEY_COLUMNS,
    content_columns,
)
from copy_row import NULL_FIELD
from manifest_keys import Key
from manifest_contract import strips_content

# How many offending rows the verdict quotes. The count is exact; the list
# is a sample, because the scan that fills it runs over citation.crawl_step
# too -- ~100k rows per depth-2 crawl, every one of them a candidate for a
# formatted string held in memory and then interpolated into a single
# message. Twenty names locate the breach; the total says how big it is.
LEAK_SAMPLE = 20

WORK_TABLE = "citation.work"
CITES_TABLE = "citation.cites"
JOURNAL_TABLE = "citation.crawl_step"
# Which columns a journal row NAMES something in is citation_columns.
# JOURNAL_KEY_COLUMNS -- imported, never restated: the producer's cut and
# this collection are two readings of one declaration, and re-typed here the
# checker would agree with the SQL that built the artifact only by accident.
ID_COLUMN = "id"
KEY_COLUMN = "key"
# The census column is citation_columns.CENSUS_COLUMN, imported for the
# reason JOURNAL_KEY_COLUMNS is: the producer tallies it off the bytes it
# writes and this module re-tallies it off the shipped file, and a name
# re-typed on one side of the boundary agrees with the other by accident.
DOCUMENT_ID_COLUMN = "document_id"
CITING_COLUMN = "citing"
CITED_COLUMN = "cited"


class LeakSample:
    """How many content values leaked, and the first few of them by name.

    An unbounded list was a fact about the dump held entirely in memory and
    then rendered entirely into one line: an artifact that failed to strip
    crawl_step.reason would report the breach as a hundred-thousand-element
    string nobody can read.
    """

    def __init__(self, limit: int = LEAK_SAMPLE):
        self.limit = limit
        self.total = 0
        self.sample: list[str] = []

    def add(self, item: str) -> None:
        self.total += 1
        if len(self.sample) < self.limit:
            self.sample.append(item)


class CitationFacts(NamedTuple):
    """What the citation visitors collected on profile_checks.py's one pass.

    A record with named fields for the reason corpus_content_checks.
    CorpusFacts is one: read out of a dict with a default, every fact here
    resolves to an empty container when it is absent, and an empty
    container is what makes "absent from the dump: none" and "leaked 0
    row(s)" true. A visitor that never fired because the COPY header spelled
    the table differently, a key renamed on one side of the module split, a
    checker built from an older bundle -- all three used to certify a green
    row. Now the read itself raises (ARTIFACT_SIDE_FAILS_CLOSED).
    """

    leaked: "LeakSample"
    work_by_kind: dict[str, int]
    work_documents: dict[str, str]
    work_ids: set[str]
    work_keys: set[str]
    edge_endpoints: set[str]
    journal_keys: set[str]


def attach_visitors(row_visitors: dict, mode: str | None, *,
                    cut_applies: bool) -> CitationFacts:
    """Registers this module's row callbacks into `row_visitors` (mutated in
    place) and returns the CitationFacts record they fill -- read it back
    after dump_scan.scan() has actually run. profile_checks.py carries it
    beside the corpus half's record rather than merging the two.

    `mode` is the manifest's citation mode, and how much gets registered is
    manifest_contract.strips_content(mode) -- the predicate the dump itself
    projects by (citation_dump._select_expression), so no mode is exempt
    from the hunt for content that same mode was allowed to ship. The hunt
    is skipped only under a mode DECLARED full-content.

    Skipping is worth doing because a visitor is not free: it is a call and
    a Row per row of a block that has one, and citation.crawl_step grows by
    ~100k rows per depth-2 crawl. Under a full-content mode those tables get
    no visitor at all; a dump carrying no citation table pays for the
    registration and finds nothing to visit.

    work and cites keep their visitors either way: work_documents /
    work_ids / edge_endpoints feed the checks that hold the citation cut to
    the document cut, and those run under every shipping mode.

    The journal's KEY columns are the third such fact and the expensive one
    -- ~100k rows per depth-2 crawl, a visit each -- so they are
    collected exactly when their check can decide something: `cut_applies`
    is manifest_classes.cut_applies(), the one declaration of "this artifact
    classifies some document out". Where nothing is cut (the full profile by
    construction) the only consumer, citation_cut_checks.
    check_journal_names_nothing_cut, compares the keys with an empty set and
    is green whatever the journal holds -- so the accumulation buys a
    verdict nobody can fail. Registered only `if stripping`, on the other
    hand, the largest and most delicately cut table in the schema was the
    one table the recipient could learn nothing about; the content hunt over
    it still runs by mode, below.
    """
    leaked = LeakSample()
    work_by_kind: dict[str, int] = {}
    work_documents: dict[str, str] = {}
    work_ids: set[str] = set()
    work_keys: set[str] = set()
    edge_endpoints: set[str] = set()
    journal_keys: set[str] = set()

    def content_visitor(table: str, name_of):
        """Reports every non-empty content column of `table`, whatever the
        classification currently says they are -- so a column promoted to
        content in citation_columns.py is checked here without this module
        being edited at all.
        """
        columns = content_columns(table.split(".", 1)[1])

        def visit(row) -> None:
            for column in columns:
                if not row.is_blank(column):
                    leaked.add(f"{table}.{column}:{name_of(row)}")
        return visit

    def no_content_hunt(_row) -> None:
        return None

    stripping = strips_content(mode)
    leaked_work = content_visitor(
        WORK_TABLE, lambda row: row.get("key", "?")) if stripping else no_content_hunt
    leaked_cites = content_visitor(
        CITES_TABLE,
        lambda row: f"{row.get('citing', '?')}->{row.get('cited', '?')}",
    ) if stripping else no_content_hunt
    leaked_journal = content_visitor(
        JOURNAL_TABLE, lambda row: row.get("id", "?")) if stripping else no_content_hunt

    def on_work(row) -> None:
        leaked_work(row)
        # Every work row is counted, under its kind or -- if the dump
        # carries no kind column at all -- under the wire format's own NULL,
        # which no manifest census can equal. A row silently left out would
        # make the census agree by shrinking (ARTIFACT_SIDE_FAILS_CLOSED).
        kind = row.get(CENSUS_COLUMN, NULL_FIELD)
        work_by_kind[kind] = work_by_kind.get(kind, 0) + 1
        if ID_COLUMN in row:
            work_ids.add(row[ID_COLUMN])
        if KEY_COLUMN in row:
            work_keys.add(row[KEY_COLUMN])
        if not row.is_blank(DOCUMENT_ID_COLUMN):
            work_documents.setdefault(row[DOCUMENT_ID_COLUMN], row.get("key", "?"))

    def on_cites(row) -> None:
        leaked_cites(row)
        for column in (CITING_COLUMN, CITED_COLUMN):
            if column in row:
                edge_endpoints.add(row[column])

    def on_journal(row) -> None:
        leaked_journal(row)
        for column in JOURNAL_KEY_COLUMNS:
            if not row.is_blank(column):
                journal_keys.add(row[column])

    row_visitors[WORK_TABLE] = on_work
    row_visitors[CITES_TABLE] = on_cites
    if cut_applies:
        row_visitors[JOURNAL_TABLE] = on_journal
    # The remaining tables have no facts of their own to collect but still
    # carry content columns, and a table whose visitor nobody registered is
    # a table the scan never opens: crawl_step.reason shipped unchecked for
    # exactly that reason before the classification became one map. They
    # exist for the content hunt and nothing else, so they are registered
    # only when there is a hunt (whatever row_visitors already holds is
    # skipped here -- the journal included, when its keys are collected).
    if stripping:
        for table in CITATION_COLUMN_CLASS:
            qualified = f"citation.{table}"
            if qualified not in row_visitors and content_columns(table):
                row_visitors[qualified] = content_visitor(
                    qualified, lambda row: row.get("id", "?"))
    return CitationFacts(leaked=leaked, work_by_kind=work_by_kind,
                         work_documents=work_documents,
                         work_ids=work_ids, work_keys=work_keys,
                         edge_endpoints=edge_endpoints, journal_keys=journal_keys)


def check_content_is_stripped(manifest: dict, facts) -> tuple[bool, str]:
    """No content column survives a mode outside CitationMode.FULL_CONTENT
    -- an unrecognised one included, which is why the question is
    strips_content(mode) rather than `== topology-only`."""
    citation = manifest.get(Key.CITATION, {})
    mode = citation.get(Key.CITATION_MODE)
    if not strips_content(mode):
        return True, f"mode={mode!r} carries content by declaration -- nothing to strip"
    leaked = facts.citation.leaked
    if not leaked.total:
        return True, "leaked 0 row(s)"
    return False, (f"leaked {leaked.total} row(s), first {len(leaked.sample)}: "
                   f"{leaked.sample}")
