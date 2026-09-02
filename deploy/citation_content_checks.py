"""Static verification of the citation-schema slice of a dump, split out of
profile_checks.py purely for size (kb/CLAUDE.md FILE_SIZE) -- same
DUMP-BYTES-ONLY discipline as that module's own docstring: what
manifest.legal is to corpus.documents, manifest.citation is to
citation.work/cites, and this module is the reader that holds the dump to
that claim.

Checks:

  schema/counts agree     the dump carries (or, under CitationMode.NONE,
                           carries NEITHER of) citation.work/citation.cites,
                           and their row counts equal
                           manifest.citation.work_count/cites_count
  topology-only stripped  no citation.work row carries a non-empty abstract
                           or evidence, and no citation.cites row carries
                           evidence, whenever manifest.citation.mode is
                           topology-only (a no-op check, trivially true,
                           under the other two modes)
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
from manifest_contract import CitationMode, Key

WORK_TABLE = "citation.work"
CITES_TABLE = "citation.cites"
ABSTRACT_COLUMN = "abstract"
EVIDENCE_COLUMN = "evidence"
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

    def on_work(row: dict) -> None:
        if row.get(ABSTRACT_COLUMN, "") not in ("", dump_scan.NULL_FIELD):
            leaked.append(f"{WORK_TABLE}.{ABSTRACT_COLUMN}:{row.get('key', '?')}")
        if row.get(EVIDENCE_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            leaked.append(f"{WORK_TABLE}.{EVIDENCE_COLUMN}:{row.get('key', '?')}")
        if ID_COLUMN in row:
            work_ids.add(row[ID_COLUMN])
        document_id = row.get(DOCUMENT_ID_COLUMN, dump_scan.NULL_FIELD)
        if document_id not in (dump_scan.NULL_FIELD, ""):
            work_documents.setdefault(document_id, row.get("key", "?"))

    def on_cites(row: dict) -> None:
        if row.get(EVIDENCE_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            leaked.append(
                f"{CITES_TABLE}.{EVIDENCE_COLUMN}:{row.get('citing', '?')}->{row.get('cited', '?')}"
            )
        for column in (CITING_COLUMN, CITED_COLUMN):
            if column in row:
                edge_endpoints.add(row[column])

    row_visitors[WORK_TABLE] = on_work
    row_visitors[CITES_TABLE] = on_cites
    return {
        "citation_leaked": leaked,
        "citation_work_documents": work_documents,
        "citation_work_ids": work_ids,
        "citation_edge_endpoints": edge_endpoints,
    }


def _ships_citation(manifest: dict) -> bool:
    mode = manifest.get(Key.CITATION, {}).get(Key.CITATION_MODE)
    return mode is not None and mode != CitationMode.NONE


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


def check_topology_only_strips_abstract_and_evidence(manifest: dict, facts: dict) -> tuple[bool, str]:
    citation = manifest.get(Key.CITATION, {})
    if citation.get(Key.CITATION_MODE) != CitationMode.TOPOLOGY_ONLY:
        return True, "mode is not topology-only -- nothing to strip"
    leaked = facts.get("citation_leaked", [])
    ok = not leaked
    return ok, f"leaked column(s)/row(s): {leaked or 'none'}"
