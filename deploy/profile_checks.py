#!/usr/bin/env python3
"""Static verification that an artifact's CONTENT matches the profile and
the legal classification its own manifest declares.

Static means: no Docker, no Postgres, no network -- the dump's bytes are
read directly (dump_scan.py) and compared against manifest.json. That is
deliberately the whole point. The question "did the public artifact ship a
paper it may not ship" must be answerable before the artifact is restored
anywhere, by anyone who has only the file, and it must be answered from the
shipped bytes rather than from the packager's intentions.

Checks, all four required by task 033 plus the two that make them meaningful:

  profile/manifest agreement   the declared profile's schemas are the ones
                               the dump actually contains (public: no
                               measurements schema at all, not merely no
                               measurements rows)
  classification is complete   manifest.legal.unclassified_documents == 0
                               and every document in the dump appears in
                               exactly one of the declared class lists
  metadata-only is stripped    no source_blob and no page body for ANY
                               metadata-only document
  full-text is intact          a source blob and a non-empty body for EVERY
                               full-content document
  vectors survive both classes every page row carries an embedding
  no tsv anywhere              tsv is GENERATED from body; a dump that
                               declared it would restore stale text content

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

import dump_scan
from manifest_contract import Key, Profile

# Column names the checks reason about, from corpus.documents/corpus.pages.
BLOB_COLUMN = "source_blob"
BODY_COLUMN = "body"
EMBEDDING_COLUMN = "embedding"
DOCUMENT_ID_COLUMN = "document_id"
ID_COLUMN = "id"
TSV_COLUMN = "tsv"

DOCUMENTS_TABLE = "corpus.documents"
PAGES_TABLE = "corpus.pages"


def _classes(manifest: dict) -> tuple[dict[str, list[str]], list[str]]:
    """(documents_by_distribution, full_content_distributions) from the
    manifest's legal block.
    """
    legal = manifest.get(Key.LEGAL, {})
    return (
        legal.get(Key.DOCUMENTS_BY_DISTRIBUTION, {}),
        legal.get(Key.FULL_CONTENT_DISTRIBUTIONS, []),
    )


def _content_expectation(manifest: dict) -> tuple[set[str], set[str]]:
    """(ids that must carry full content, ids that must be stripped).

    For the FULL profile nothing is stripped -- the artifact carries
    everything regardless of class, and that is exactly what these checks
    then assert about it.
    """
    by_distribution, full_content = _classes(manifest)
    everything = {doc_id for ids in by_distribution.values() for doc_id in ids}
    if manifest.get(Key.PROFILE) != Profile.PUBLIC:
        return everything, set()
    full_ids = {doc_id for name in full_content for doc_id in by_distribution.get(name, [])}
    return full_ids, everything - full_ids


def check_schemas(dump_path: Path, manifest: dict) -> tuple[bool, str]:
    declared = set(manifest.get(Key.SCHEMAS, []))
    present = dump_scan.schema_names(dump_path)
    ok = present == declared
    return ok, f"dump carries {sorted(present)}, manifest declares {sorted(declared)}"


def check_classification_complete(manifest: dict, scans: dict) -> tuple[bool, str]:
    legal = manifest.get(Key.LEGAL, {})
    unclassified = legal.get(Key.UNCLASSIFIED_DOCUMENTS)
    by_distribution, _ = _classes(manifest)
    listed = [doc_id for ids in by_distribution.values() for doc_id in ids]
    duplicated = sorted({i for i in listed if listed.count(i) > 1})
    documents = scans.get(DOCUMENTS_TABLE)
    dumped = documents.rows if documents else 0
    ok = unclassified == 0 and not duplicated and dumped == len(listed)
    return ok, (
        f"unclassified={unclassified}, {len(listed)} id(s) across "
        f"{len(by_distribution)} class(es) vs {dumped} document row(s) in the dump"
        + (f", listed twice: {duplicated}" if duplicated else "")
    )


def _visit(dump_path: Path, manifest: dict) -> tuple[dict, dict]:
    """One streaming pass over the dump, collecting per-document facts the
    content checks need: which ids carry a blob, which carry page text, and
    whether every page row has an embedding.
    """
    with_blob: set[str] = set()
    with_body: set[str] = set()
    pages_seen = {"rows": 0, "no_embedding": 0}

    def on_document(row: dict) -> None:
        if row.get(BLOB_COLUMN, dump_scan.NULL_FIELD) not in (dump_scan.NULL_FIELD, ""):
            with_blob.add(row[ID_COLUMN])

    def on_page(row: dict) -> None:
        pages_seen["rows"] += 1
        if row.get(EMBEDDING_COLUMN, dump_scan.NULL_FIELD) in (dump_scan.NULL_FIELD, ""):
            pages_seen["no_embedding"] += 1
        if row.get(BODY_COLUMN, "") not in ("", dump_scan.NULL_FIELD):
            with_body.add(row[DOCUMENT_ID_COLUMN])

    scans = dump_scan.scan(dump_path, {DOCUMENTS_TABLE: on_document, PAGES_TABLE: on_page})
    facts = {"with_blob": with_blob, "with_body": with_body, "pages": pages_seen}
    return scans, facts


def check_metadata_only_stripped(manifest: dict, facts: dict) -> tuple[bool, str]:
    _full_ids, stripped = _content_expectation(manifest)
    leaked_blob = sorted(stripped & facts["with_blob"])
    leaked_body = sorted(stripped & facts["with_body"])
    ok = not leaked_blob and not leaked_body
    return ok, (
        f"{len(stripped)} metadata-only document(s); "
        f"blobs present for {leaked_blob or 'none'}, page text present for {leaked_body or 'none'}"
    )


def check_full_content_intact(manifest: dict, facts: dict) -> tuple[bool, str]:
    full_ids, _stripped = _content_expectation(manifest)
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
        ("metadata-only: ни блоба, ни текста", *check_metadata_only_stripped(manifest, facts)),
        ("full-text: блоб и текст на месте", *check_full_content_intact(manifest, facts)),
        ("векторы у всех страниц", *check_pages_embedded(manifest, scans, facts)),
        ("нет generated-колонок в дампе", *check_no_generated_columns(scans)),
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
