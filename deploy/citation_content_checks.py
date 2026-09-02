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


def attach_visitors(row_visitors: dict) -> list[str]:
    """Registers this module's row callbacks into `row_visitors` (mutated in
    place) and returns the list they append leaked column/row descriptions
    to -- read it back after dump_scan.scan() has actually run.
    """
    leaked: list[str] = []

    def on_work(row: dict) -> None:
        if row.get(ABSTRACT_COLUMN, "") not in ("", dump_scan.NULL_FIELD):
            leaked.append(f"{WORK_TABLE}.{ABSTRACT_COLUMN}:{row.get('key', '?')}")
        if row.get(EVIDENCE_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            leaked.append(f"{WORK_TABLE}.{EVIDENCE_COLUMN}:{row.get('key', '?')}")

    def on_cites(row: dict) -> None:
        if row.get(EVIDENCE_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            leaked.append(
                f"{CITES_TABLE}.{EVIDENCE_COLUMN}:{row.get('citing', '?')}->{row.get('cited', '?')}"
            )

    row_visitors[WORK_TABLE] = on_work
    row_visitors[CITES_TABLE] = on_cites
    return leaked


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
