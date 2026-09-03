"""The citation cut held to the DOCUMENT cut, from the dump's own bytes.

Split out of citation_content_checks.py for module size (kb/CLAUDE.md
FILE_SIZE) along the question, not the table: that module collects the
facts (it owns the row visitors profile_checks.py's single pass carries)
and hunts for content a mode was not allowed to ship; everything here asks
whether the rows that DID ship name only things this artifact carries.

Checks:

  schema/counts agree     the dump carries (or, under a mode that ships
                           nothing, carries NEITHER of) citation.work and
                           citation.cites, and their row counts equal
                           manifest.citation.work_count/cites_count
  every table is named    and, one level wider, EVERY citation table the
                           manifest declares in table_rows is in the dump
                           with exactly that many rows, and no citation
                           table is in the dump the manifest does not
                           declare -- so the checks below cannot pass
                           because their subject never shipped
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
  journal names nothing   no citation.crawl_step row names, in any of its
   the cut removed         three key columns, a document the manifest
                           classifies out of this artifact

"the SELECT filtered it out" is a claim about the packager; each of these is
a claim about the file, which is the whole reason profile_checks.py travels
inside the artifact (ARTIFACT_SIDE_FAILS_CLOSED).

`facts` is the pair profile_checks._visit() returns, and the three checks
below are the ones that read BOTH halves of it: the citation rows under
.citation, the document ids they must name under .corpus. Read by name, so
a half that never arrived raises here rather than passing as an empty set
-- these checks are exactly the ones an empty set certifies.
"""
from __future__ import annotations

from citation_content_checks import CITES_TABLE, WORK_TABLE
from manifest_classes import expected_ids
from manifest_keys import Key
from manifest_contract import ships_citation

# The schema whose tables this module reasons about. Spelled once: the
# manifest names a table bare, the dump names it qualified, and this is the
# only place the two spellings are put together.
SCHEMA = "citation"


def _ships_citation(manifest: dict) -> bool:
    """manifest_contract.ships_citation() over this manifest's block -- the
    same allowlist the dump was written by, so a mode neither side has heard
    of leaves the checks demanding an absent schema rather than exempting
    themselves."""
    return ships_citation(manifest.get(Key.CITATION, {}).get(Key.CITATION_MODE))


def check_work_documents_are_in_the_dump(manifest: dict, facts) -> tuple[bool, str]:
    """No citation.work row may name a document this dump does not carry.

    The FK (citation.work.document_id REFERENCES corpus.documents) would
    abort the restore, and the row itself would publish the bibliography of
    a document the owner classified out of the public artifact -- two
    independent reasons, either sufficient.
    """
    if not _ships_citation(manifest):
        return True, "citation schema not in this profile -- nothing to check"
    named = facts.citation.work_documents
    dangling = sorted(set(named) - facts.corpus.documents)
    ok = not dangling
    return ok, (
        f"{len(named)} document(s) named by citation.work; "
        f"absent from the dump: {[f'{d} ({named[d]})' for d in dangling] or 'none'}"
    )


def check_edges_reference_shipped_works(manifest: dict, facts) -> tuple[bool, str]:
    """No citation.cites endpoint may name a work row the dump dropped --
    the same FK question one level down, for the edges the work cut takes
    with it.
    """
    if not _ships_citation(manifest):
        return True, "citation schema not in this profile -- nothing to check"
    endpoints = facts.citation.edge_endpoints
    dangling = sorted(endpoints - facts.citation.work_ids)
    ok = not dangling
    return ok, (
        f"{len(endpoints)} distinct endpoint(s) in citation.cites; "
        f"without a work row: {dangling or 'none'}"
    )


def check_citation_schema_matches_mode(manifest: dict, scans: dict) -> tuple[bool, str]:
    citation = manifest.get(Key.CITATION, {})
    mode = citation.get(Key.CITATION_MODE)
    present = {WORK_TABLE, CITES_TABLE} & set(scans)
    if not ships_citation(mode):
        ok = not present
        return ok, f"mode={mode!r}, citation table(s) in dump: {sorted(present) or 'none'}"
    work_rows = scans[WORK_TABLE].rows if WORK_TABLE in scans else 0
    cites_rows = scans[CITES_TABLE].rows if CITES_TABLE in scans else 0
    want_work = citation.get(Key.WORK_COUNT)
    want_cites = citation.get(Key.CITES_COUNT)
    ok = present == {WORK_TABLE, CITES_TABLE} and work_rows == want_work and cites_rows == want_cites
    return ok, (f"mode={mode!r}, work rows={work_rows} (manifest {want_work}), "
                f"cites rows={cites_rows} (manifest {want_cites})")


def check_every_declared_table_shipped(manifest: dict, scans: dict) -> tuple[bool, str]:
    """Every citation table manifest.citation.table_rows names is in the
    dump, with exactly that many COPY rows -- and no citation table is in
    the dump that the manifest does not name.

    The two counts next door describe work and cites; crawl_step,
    public_policy and schema_backfill were described by nothing at all, so
    a dump that carried no journal satisfied every check about the journal.
    check_journal_names_nothing_cut() in particular reported a green "0
    distinct name(s)" -- an artifact missing the most delicately cut table
    in the schema certified as one whose cut holds.

    Both directions, and an empty declaration under a shipping mode is a
    failure rather than a quiet pass: a manifest that names no table is one
    this reader cannot hold to anything.
    """
    citation = manifest.get(Key.CITATION, {})
    declared = citation.get(Key.TABLE_ROWS)
    shipped = {name for name in scans if name.startswith(f"{SCHEMA}.")}
    if not _ships_citation(manifest):
        ok = not declared and not shipped
        return ok, (f"mode={citation.get(Key.CITATION_MODE)!r}; "
                    f"declared {sorted(declared or [])}, in dump {sorted(shipped)}")
    if not isinstance(declared, dict) or not declared:
        return False, (
            f"manifest.citation.{Key.TABLE_ROWS} пуст ({declared!r}) при режиме, "
            "который везёт схему: держать дамп не к чему -- пересоберите пакет "
            "текущим сборщиком"
        )
    problems = []
    for table, want in sorted(declared.items()):
        qualified = f"{SCHEMA}.{table}"
        scan = scans.get(qualified)
        if scan is None:
            problems.append(f"{qualified}: блока COPY нет, а манифест обещает {want}")
        elif scan.rows != want:
            problems.append(f"{qualified}: {scan.rows} строк против {want}")
    for name in sorted(shipped - {f"{SCHEMA}.{table}" for table in declared}):
        problems.append(f"{name}: в дампе есть, в манифесте не назван")
    return not problems, (
        f"{len(declared)} declared table(s): " + ("; ".join(problems) or "все на месте, "
        + ", ".join(f"{table}={rows}" for table, rows in sorted(declared.items())))
    )


def check_journal_names_nothing_cut(manifest: dict, facts) -> tuple[bool, str]:
    """No citation.crawl_step row may name a document this artifact does not
    carry.

    The journal cut is the package's most intricate policy SQL -- three key
    columns matched against two vocabularies (documents and work keys) in a
    three-branch UNION -- and it was the one cut asserted only by the
    packager's WHERE clause. This is the claim about the FILE: every name
    the journal spells, in any of the three columns, tested against the ids
    the manifest classifies out of this artifact.

    The DOCUMENT half is decidable here and is what this check makes; the
    other half of the cut -- a work key whose document was classified out --
    is not, and the count below says so rather than leaving the gap
    unspoken. A journal key naming no shipped work is the NORMAL shape of a
    `drop` row: a candidate that failed tau was journalled and never written
    to citation.work, so "every journal key is a work in the dump" would
    fail on a correct package (8 such rows in the live journal at the time
    of writing). Distinguishing a dropped candidate from a cut work would
    take the cut work keys into the manifest -- i.e. publishing OpenAlex
    identifiers for documents whose owner has not established a right to say
    even that much from this package (Distribution.SHIPPED's own note), so
    it is deliberately not done. What still holds that half is
    check_work_documents_are_in_the_dump: a work row that survived the cut
    while its document did not fails there.
    """
    if not _ships_citation(manifest):
        return True, "citation schema not in this profile -- nothing to check"
    keys = facts.citation.journal_keys
    _expected, absent = expected_ids(manifest)
    leaked = sorted(keys & absent)
    unresolved = keys - absent - facts.corpus.documents - facts.citation.work_keys
    ok = not leaked
    return ok, (
        f"{len(keys)} distinct name(s) in citation.crawl_step; "
        f"naming a document this artifact drops: {leaked or 'none'}; "
        f"{len(unresolved)} naming neither a shipped document nor a shipped work "
        "(dropped candidates -- undecidable from the dump alone)"
    )
