"""Static verification of the corpus-schema slice of a dump, split out of
profile_checks.py for size (kb/CLAUDE.md FILE_SIZE) along the seam
citation_content_checks.py already marks out: one module per schema whose
cut the artifact has to be held to, each contributing row visitors to the
single streaming pass profile_checks.py owns, and each answering only from
the dump's own bytes.

Checks:

  classification is complete   manifest.legal.unclassified_documents == 0
                               and every document row in the dump appears in
                               exactly one of the class lists the manifest
                               declares this artifact to carry
  excluded left no trace       a document classified outside
                               legal.shipped_distributions has no documents
                               row and no page row -- an omission asserted
                               from the shipped bytes, not from the
                               packager's WHERE clause
  metadata-only is stripped    no content column of corpus.documents and
                               none of corpus.pages carries a value for ANY
                               metadata-only document -- WHICH columns those
                               are is corpus_columns.py, the map the
                               packager cut by, imported rather than
                               restated
  full-text is intact          those same columns all carry one for EVERY
                               full-content document
  vectors survive both classes every page row carries an embedding
  no tsv anywhere              tsv is GENERATED from body; a dump that
                               declared it would restore stale text content

Which ids the manifest says are carried, and in which shape, is
manifest_classes.py -- a manifest-only reading with no dump byte in it.
Everything here is the other half: the dump held against one of those
answers.
"""
from __future__ import annotations

import corpus_columns
import dump_scan
from manifest_classes import classes, content_expectation, expected_ids
from manifest_keys import Key

# Which columns carry CONTENT is imported, never restated: corpus_columns.py
# is the map the packager cuts by, and a second copy of the two names here
# could only agree with it by accident -- on the one question this module
# exists to answer. The names below are the other kind: how a row says which
# document it is (id, document_id), the vector that ships in both classes,
# and the generated column no COPY list may carry. Those are the dump's
# SHAPE, not the classification.
EMBEDDING_COLUMN = "embedding"
DOCUMENT_ID_COLUMN = "document_id"
ID_COLUMN = "id"
TSV_COLUMN = "tsv"

DOCUMENTS_TABLE = "corpus.documents"
PAGES_TABLE = "corpus.pages"
DOCUMENT_CONTENT = corpus_columns.content_columns("documents")
PAGE_CONTENT = corpus_columns.content_columns("pages")


def _carries_content(row: dict, columns: tuple[str, ...]) -> bool:
    r"""Does this row carry a value in ANY column the map calls content?

    Empty and \N are both "no content": a metadata-only page ships with an
    empty body on purpose (corpus_columns.py says why), and an absent blob
    is NULL.
    """
    return any(row.get(column, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, "")
               for column in columns)


def attach_visitors(row_visitors: dict) -> dict:
    """Registers this module's row callbacks into `row_visitors` (mutated in
    place) and returns the fact containers they fill -- read them back after
    dump_scan.scan() has actually run, the same contract
    citation_content_checks.attach_visitors() follows.

    Per-document facts the content checks need: which ids the dump carries
    at all, which carry a blob, which carry page text, and whether every
    page row has an embedding.
    """
    documents: set[str] = set()
    page_documents: set[str] = set()
    with_blob: set[str] = set()
    with_body: set[str] = set()
    pages_seen = {"rows": 0, "no_embedding": 0}

    def on_document(row: dict) -> None:
        documents.add(row[ID_COLUMN])
        if _carries_content(row, DOCUMENT_CONTENT):
            with_blob.add(row[ID_COLUMN])

    def on_page(row: dict) -> None:
        pages_seen["rows"] += 1
        page_documents.add(row[DOCUMENT_ID_COLUMN])
        if row.get(EMBEDDING_COLUMN, dump_scan.NULL_FIELD) in (dump_scan.NULL_FIELD, ""):
            pages_seen["no_embedding"] += 1
        if _carries_content(row, PAGE_CONTENT):
            with_body.add(row[DOCUMENT_ID_COLUMN])

    row_visitors[DOCUMENTS_TABLE] = on_document
    row_visitors[PAGES_TABLE] = on_page
    return {
        "documents": documents, "page_documents": page_documents,
        "with_blob": with_blob, "with_body": with_body, "pages": pages_seen,
    }


def check_classification_complete(manifest: dict, scans: dict) -> tuple[bool, str]:
    legal = manifest.get(Key.LEGAL, {})
    if Key.SHIPPED_DISTRIBUTIONS not in legal:
        return False, (
            f"manifest.legal declares no {Key.SHIPPED_DISTRIBUTIONS} -- which classes this "
            "artifact carries cannot be inferred; rebuild it with the current packager"
        )
    unclassified = legal.get(Key.UNCLASSIFIED_DOCUMENTS)
    by_distribution, _full_content, _shipped = classes(manifest)
    all_listed = [doc_id for ids in by_distribution.values() for doc_id in ids]
    duplicated = sorted({i for i in all_listed if all_listed.count(i) > 1})
    expected, _absent = expected_ids(manifest)
    documents = scans.get(DOCUMENTS_TABLE)
    dumped = documents.rows if documents else 0
    ok = unclassified == 0 and not duplicated and dumped == len(expected)
    return ok, (
        f"unclassified={unclassified}, {len(all_listed)} id(s) across "
        f"{len(by_distribution)} class(es), {len(expected)} of them shipped by this profile, "
        f"vs {dumped} document row(s) in the dump"
        + (f", listed twice: {duplicated}" if duplicated else "")
    )


def check_excluded_absent(manifest: dict, facts: dict) -> tuple[bool, str]:
    """A document the manifest classifies outside shipped_distributions must
    have no documents row and no page row. Asserted against the dump's own
    bytes: "the SELECT filtered it out" is a claim about the packager, this
    is a claim about the file.
    """
    _expected, absent = expected_ids(manifest)
    leaked_documents = sorted(absent & facts["documents"])
    leaked_pages = sorted(absent & facts["page_documents"])
    ok = not leaked_documents and not leaked_pages
    return ok, (
        f"{len(absent)} excluded document(s); rows present for "
        f"{leaked_documents or 'none'}, pages present for {leaked_pages or 'none'}"
    )


def check_metadata_only_stripped(manifest: dict, facts: dict) -> tuple[bool, str]:
    _full_ids, stripped = content_expectation(manifest)
    leaked_blob = sorted(stripped & facts["with_blob"])
    leaked_body = sorted(stripped & facts["with_body"])
    ok = not leaked_blob and not leaked_body
    return ok, (
        f"{len(stripped)} metadata-only document(s); "
        f"blobs present for {leaked_blob or 'none'}, page text present for {leaked_body or 'none'}"
    )


def check_full_content_intact(manifest: dict, facts: dict) -> tuple[bool, str]:
    full_ids, _stripped = content_expectation(manifest)
    missing_blob = sorted(full_ids - facts["with_blob"])
    missing_body = sorted(full_ids - facts["with_body"])
    ok = not missing_blob and not missing_body
    return ok, (
        f"{len(full_ids)} full-content document(s); "
        f"missing blob: {missing_blob or 'none'}, missing page text: {missing_body or 'none'}"
    )


def check_pages_embedded(manifest: dict, scans: dict, facts: dict) -> tuple[bool, str]:
    pages = facts["pages"]
    want = manifest.get(Key.PAGES_COUNT)
    ok = pages["rows"] == want and pages["no_embedding"] == 0
    return ok, (
        f"{pages['rows']} page row(s) (manifest {want}), "
        f"{pages['no_embedding']} without an embedding"
    )


def check_no_generated_columns(scans: dict) -> tuple[bool, str]:
    """tsv (and source_path) are GENERATED: a dump that declared them in a
    COPY column list would either fail to restore or, worse, carry text
    content that no longer matches body -- the exact leak the public profile
    exists to prevent.
    """
    pages = scans.get(PAGES_TABLE)
    columns = pages.columns if pages else []
    ok = TSV_COLUMN not in columns
    return ok, f"corpus.pages COPY columns: {columns}"
