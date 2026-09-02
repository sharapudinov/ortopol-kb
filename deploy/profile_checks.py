#!/usr/bin/env python3
"""Static verification that an artifact's CONTENT matches the profile and
the legal classification its own manifest declares.

Static means: no Docker, no Postgres, no network -- the dump's bytes are
read directly (dump_scan.py) and compared against manifest.json. That is
deliberately the whole point. The question "did the public artifact ship a
paper it may not ship" must be answerable before the artifact is restored
anywhere, by anyone who has only the file, and it must be answered from the
shipped bytes rather than from the packager's intentions.

Checks:

  profile/manifest agreement   the declared profile's schemas are the ones
                               the dump actually contains (public: no
                               measurements schema at all, not merely no
                               measurements rows)
  classification is complete   manifest.legal.unclassified_documents == 0
                               and every document row in the dump appears in
                               exactly one of the class lists the manifest
                               declares this artifact to carry
  excluded left no trace       a document classified outside
                               legal.shipped_distributions has no documents
                               row and no page row -- an omission asserted
                               from the shipped bytes, not from the
                               packager's WHERE clause
  metadata-only is stripped    no source_blob and no page body for ANY
                               metadata-only document
  full-text is intact          a source blob and a non-empty body for EVERY
                               full-content document
  vectors survive both classes every page row carries an embedding
  no tsv anywhere              tsv is GENERATED from body; a dump that
                               declared it would restore stale text content
  citation policy is owner's   manifest.citation.policy_source == "owner":
                               an artifact whose citation mode was forced
                               with --policy-override fails here rather
                               than being certified as publishable
  legal vocabulary             which ids the manifest says are carried, and
                               in which shape, is manifest_classes.py
                               (module size): a manifest-only reading, with
                               no dump byte in it
  citation slice holds         the citation-schema checks live in
                               citation_content_checks.py (module size) and
                               run in this same pass -- including the two
                               that hold the citation cut to the DOCUMENT
                               cut: no work row names a document this dump
                               does not carry, no edge names a work it does
                               not carry

Same (ok, detail) contract as smoke_checks.py, so smoke_test.py can list
these beside its live checks; runnable standalone as well:

    python3 profile_checks.py                     # artifact beside this file
    python3 profile_checks.py --artifact-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import citation_content_checks
import dump_scan
from manifest_classes import classes, content_expectation, expected_ids
from manifest_contract import Key

# Column names the checks reason about, from corpus.documents/corpus.pages.
BLOB_COLUMN = "source_blob"
BODY_COLUMN = "body"
EMBEDDING_COLUMN = "embedding"
DOCUMENT_ID_COLUMN = "document_id"
ID_COLUMN = "id"
TSV_COLUMN = "tsv"

DOCUMENTS_TABLE = "corpus.documents"
PAGES_TABLE = "corpus.pages"


def check_schemas(dump_path: Path, manifest: dict) -> tuple[bool, str]:
    declared = set(manifest.get(Key.SCHEMAS, []))
    present = dump_scan.schema_names(dump_path)
    ok = present == declared
    return ok, f"dump carries {sorted(present)}, manifest declares {sorted(declared)}"


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


def _visit(dump_path: Path, manifest: dict) -> tuple[dict, dict]:
    """One streaming pass over the dump, collecting per-document facts the
    content checks need: which ids carry a blob, which carry page text, and
    whether every page row has an embedding.
    """
    documents: set[str] = set()
    page_documents: set[str] = set()
    with_blob: set[str] = set()
    with_body: set[str] = set()
    pages_seen = {"rows": 0, "no_embedding": 0}

    def on_document(row: dict) -> None:
        documents.add(row[ID_COLUMN])
        if row.get(BLOB_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            with_blob.add(row[ID_COLUMN])

    def on_page(row: dict) -> None:
        pages_seen["rows"] += 1
        page_documents.add(row[DOCUMENT_ID_COLUMN])
        if row.get(EMBEDDING_COLUMN, dump_scan.NULL_FIELD) in (dump_scan.NULL_FIELD, ""):
            pages_seen["no_embedding"] += 1
        if row.get(BODY_COLUMN, "") not in ("", dump_scan.NULL_FIELD):
            with_body.add(row[DOCUMENT_ID_COLUMN])

    row_visitors = {DOCUMENTS_TABLE: on_document, PAGES_TABLE: on_page}
    citation_facts = citation_content_checks.attach_visitors(
        row_visitors, manifest.get(Key.CITATION, {}).get(Key.CITATION_MODE))

    scans = dump_scan.scan(dump_path, row_visitors)
    facts = {
        "documents": documents, "page_documents": page_documents,
        "with_blob": with_blob, "with_body": with_body, "pages": pages_seen,
        **citation_facts,
    }
    return scans, facts


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


def run_checks(artifact_dir: Path) -> list[tuple[str, bool, str]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    dump_path = artifact_dir / manifest[Key.DUMP][Key.FILE]
    scans, facts = _visit(dump_path, manifest)
    profile = manifest.get(Key.PROFILE)
    return [
        (f"профиль {profile!r}: схемы дампа = манифест", *check_schemas(dump_path, manifest)),
        ("правовая классификация полна", *check_classification_complete(manifest, scans)),
        ("excluded: ни строки документа, ни страниц", *check_excluded_absent(manifest, facts)),
        ("metadata-only: ни блоба, ни текста", *check_metadata_only_stripped(manifest, facts)),
        ("full-text: блоб и текст на месте", *check_full_content_intact(manifest, facts)),
        ("векторы у всех страниц", *check_pages_embedded(manifest, scans, facts)),
        ("нет generated-колонок в дампе", *check_no_generated_columns(scans)),
        ("citation: режим — решение владельца, не --policy-override",
         *citation_content_checks.check_policy_is_the_owners(manifest)),
        ("citation: схема/счётчики совпадают с манифестом",
         *citation_content_checks.check_citation_schema_matches_mode(manifest, scans)),
        ("citation topology-only: content-колонки вырезаны",
         *citation_content_checks.check_topology_only_strips_content(manifest, facts)),
        ("citation.work ссылается только на документы этого пакета",
         *citation_content_checks.check_work_documents_are_in_the_dump(manifest, facts)),
        ("citation.cites ссылается только на узлы этого пакета",
         *citation_content_checks.check_edges_reference_shipped_works(manifest, facts)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="extracted artifact directory (default: this script's own)")
    args = parser.parse_args(argv)

    manifest_path = args.artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest.json under {args.artifact_dir}", file=sys.stderr)
        return 2

    all_ok = True
    for name, ok, detail in run_checks(args.artifact_dir):
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
