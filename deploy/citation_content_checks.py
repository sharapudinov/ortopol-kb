"""Static verification of the citation-schema slice of a dump, split out of
profile_checks.py purely for size (kb/CLAUDE.md FILE_SIZE) -- same
DUMP-BYTES-ONLY discipline as that module's own docstring: what
manifest.legal is to corpus.documents, manifest.citation is to
citation.work/cites, and this module is the reader that holds the dump to
that claim.

Checks:

  policy is the owner's  manifest.citation.policy_source says the mode came
                           from citation.public_policy and not from
                           --policy-override. The one check here that reads
                           no dump byte: it is about the PROVENANCE of the
                           cut, not about what the cut produced, and an
                           override artifact is otherwise indistinguishable
                           from a classified one
  schema/counts agree     the dump carries (or, under CitationMode.NONE,
                           carries NEITHER of) citation.work/citation.cites,
                           and their row counts equal
                           manifest.citation.work_count/cites_count
  topology-only stripped  no row of any dumped table carries a non-empty
                           value in a column citation_columns classifies as
                           content, whenever manifest.citation.mode is
                           topology-only (a no-op check, trivially true,
                           under the other two modes). The column list is
                           IMPORTED from citation_columns, never restated
                           here: a second copy of the classification could
                           only agree with the producer by accident, and
                           the one column it forgot was the whole finding
                           that put this module and the dump on one map
  work -> document holds  every citation.work.document_id in the dump names
                           a corpus.documents row the SAME dump carries.
                           citation.work.document_id is a foreign key across
                           the boundary between two independent policy cuts
                           (citation.public_policy and corpus.documents.
                           public_distribution), so this is both a restore
                           question and a legal one: a surviving row would
                           publish the title of a document classified out
  edge -> work holds      every citation.cites endpoint names a
                           citation.work row the dump carries -- the same
                           question one level down, for the rows the work
                           cut takes with it

profile_checks.py owns the single streaming pass over the dump (dump_scan.
scan reads the whole file once); attach_visitors() only adds this module's
row callbacks to that same pass instead of asking for a second one.
"""
from __future__ import annotations

import dump_scan
from citation_columns import CITATION_COLUMN_CLASS, content_columns
from manifest_contract import CitationMode, Key, PolicySource, Profile

WORK_TABLE = "citation.work"
CITES_TABLE = "citation.cites"
ID_COLUMN = "id"
DOCUMENT_ID_COLUMN = "document_id"
CITING_COLUMN = "citing"
CITED_COLUMN = "cited"


def attach_visitors(row_visitors: dict) -> dict:
    """Registers this module's row callbacks into `row_visitors` (mutated in
    place) and returns the fact containers they fill -- read them back after
    dump_scan.scan() has actually run. profile_checks.py merges the returned
    dict into its own facts, so the keys are namespaced.
    """
    leaked: list[str] = []
    work_documents: dict[str, str] = {}
    work_ids: set[str] = set()
    edge_endpoints: set[str] = set()

    def content_visitor(table: str, name_of):
        """Reports every non-empty content column of `table`, whatever the
        classification currently says they are -- so a column promoted to
        content in citation_columns.py is checked here without this module
        being edited at all.
        """
        columns = content_columns(table.split(".", 1)[1])

        def visit(row: dict) -> None:
            for column in columns:
                if row.get(column, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
                    leaked.append(f"{table}.{column}:{name_of(row)}")
        return visit

    leaked_work = content_visitor(WORK_TABLE, lambda row: row.get("key", "?"))
    leaked_cites = content_visitor(
        CITES_TABLE, lambda row: f"{row.get('citing', '?')}->{row.get('cited', '?')}")

    def on_work(row: dict) -> None:
        leaked_work(row)
        if ID_COLUMN in row:
            work_ids.add(row[ID_COLUMN])
        document_id = row.get(DOCUMENT_ID_COLUMN, dump_scan.NULL_FIELD)
        if document_id not in (dump_scan.NULL_FIELD, ""):
            work_documents.setdefault(document_id, row.get("key", "?"))

    def on_cites(row: dict) -> None:
        leaked_cites(row)
        for column in (CITING_COLUMN, CITED_COLUMN):
            if column in row:
                edge_endpoints.add(row[column])

    row_visitors[WORK_TABLE] = on_work
    row_visitors[CITES_TABLE] = on_cites
    # The tables with no facts of their own to collect still carry content
    # columns, and a table whose visitor nobody registered is a table the
    # scan never opens: crawl_step.reason shipped unchecked for exactly that
    # reason before the classification became one map.
    for table in CITATION_COLUMN_CLASS:
        qualified = f"citation.{table}"
        if qualified not in row_visitors and content_columns(table):
            row_visitors[qualified] = content_visitor(
                qualified, lambda row: row.get("id", "?"))
    return {
        "citation_leaked": leaked,
        "citation_work_documents": work_documents,
        "citation_work_ids": work_ids,
        "citation_edge_endpoints": edge_endpoints,
    }


def _ships_citation(manifest: dict) -> bool:
    mode = manifest.get(Key.CITATION, {}).get(Key.CITATION_MODE)
    return mode is not None and mode != CitationMode.NONE


def check_policy_is_the_owners(manifest: dict) -> tuple[bool, str]:
    """The citation mode this artifact applied must be the owner's decision.

    --policy-override forces a mode without reading citation.public_policy,
    which is legitimate for exercising the pipeline and never legitimate for
    an artifact anybody publishes (CITATION_POLICY_IS_DATA,
    PUBLIC_APPROVED_BY_OWNER). The build records which it was; this refuses
    the one it must.

    An override is refused BEFORE the mode is looked at: `--policy-override
    none` produces an artifact carrying no citation schema at all, and the
    refusal is about the provenance of the decision, not about how much it
    let through.

    WHICH source is required depends on the profile, because only the
    public one applies a policy: public must name the owner, and anything
    else must say "not-applicable" -- a full artifact claiming an owner
    decision names one nobody made, since the packager never reads
    citation.public_policy for that profile. The two are refused in each
    other's place, not merely accepted loosely.

    A shipping artifact whose manifest names no source is refused too --
    that is a manifest written before the field existed, and reading it
    with a default is exactly how an override build would come to be
    certified as owner-classified. (The version gate in smoke_checks names
    that case more precisely; this is the static backstop, which runs
    without a database.) An artifact that ships no citation schema has no
    policy to source and nothing to certify, so it passes.
    """
    citation = manifest.get(Key.CITATION, {})
    mode = citation.get(Key.CITATION_MODE)
    source = citation.get(Key.CITATION_POLICY_SOURCE)
    public = manifest.get(Key.PROFILE) == Profile.PUBLIC
    if source == PolicySource.OVERRIDE:
        return False, (
            "артефакт собран с --policy-override, не по решению владельца; "
            f"публиковать нельзя (mode={mode!r} задан командной строкой, а не "
            "citation.public_policy)"
        )
    if not _ships_citation(manifest):
        return True, f"mode={mode!r} — граф не уезжает, политике неоткуда взяться"
    wanted = PolicySource.OWNER if public else PolicySource.NOT_APPLICABLE
    if source != wanted:
        return False, (
            f"citation.policy_source={source!r} при профиле "
            f"{manifest.get(Key.PROFILE)!r} — ожидалось {wanted!r} "
            f"(значения: {PolicySource.ALL}); пересоберите артефакт "
            "текущим сборщиком"
        )
    if not public:
        return True, (f"policy_source={source!r}, mode={mode!r} — профиль "
                      "политики не применяет, решать было нечего")
    return True, f"policy_source={source!r}, mode={mode!r} — решение владельца"


def check_work_documents_are_in_the_dump(manifest: dict, facts: dict) -> tuple[bool, str]:
    """No citation.work row may name a document this dump does not carry.

    The FK (citation.work.document_id REFERENCES corpus.documents) would
    abort the restore, and the row itself would publish the bibliography of
    a document the owner classified out of the public artifact -- two
    independent reasons, either sufficient.
    """
    if not _ships_citation(manifest):
        return True, "citation schema not in this profile -- nothing to check"
    named = facts.get("citation_work_documents", {})
    dangling = sorted(set(named) - facts.get("documents", set()))
    ok = not dangling
    return ok, (
        f"{len(named)} document(s) named by citation.work; "
        f"absent from the dump: {[f'{d} ({named[d]})' for d in dangling] or 'none'}"
    )


def check_edges_reference_shipped_works(manifest: dict, facts: dict) -> tuple[bool, str]:
    """No citation.cites endpoint may name a work row the dump dropped --
    the same FK question one level down, for the edges the work cut takes
    with it.
    """
    if not _ships_citation(manifest):
        return True, "citation schema not in this profile -- nothing to check"
    endpoints = facts.get("citation_edge_endpoints", set())
    dangling = sorted(endpoints - facts.get("citation_work_ids", set()))
    ok = not dangling
    return ok, (
        f"{len(endpoints)} distinct endpoint(s) in citation.cites; "
        f"without a work row: {dangling or 'none'}"
    )


def check_citation_schema_matches_mode(manifest: dict, scans: dict) -> tuple[bool, str]:
    citation = manifest.get(Key.CITATION, {})
    mode = citation.get(Key.CITATION_MODE)
    present = {WORK_TABLE, CITES_TABLE} & set(scans)
    if mode == CitationMode.NONE or mode is None:
        ok = not present
        return ok, f"mode={mode!r}, citation table(s) in dump: {sorted(present) or 'none'}"
    work_rows = scans[WORK_TABLE].rows if WORK_TABLE in scans else 0
    cites_rows = scans[CITES_TABLE].rows if CITES_TABLE in scans else 0
    want_work = citation.get(Key.WORK_COUNT)
    want_cites = citation.get(Key.CITES_COUNT)
    ok = present == {WORK_TABLE, CITES_TABLE} and work_rows == want_work and cites_rows == want_cites
    return ok, (f"mode={mode!r}, work rows={work_rows} (manifest {want_work}), "
                f"cites rows={cites_rows} (manifest {want_cites})")


def check_topology_only_strips_content(manifest: dict, facts: dict) -> tuple[bool, str]:
    citation = manifest.get(Key.CITATION, {})
    if citation.get(Key.CITATION_MODE) != CitationMode.TOPOLOGY_ONLY:
        return True, "mode is not topology-only -- nothing to strip"
    leaked = facts.get("citation_leaked", [])
    ok = not leaked
    return ok, f"leaked column(s)/row(s): {leaked or 'none'}"
