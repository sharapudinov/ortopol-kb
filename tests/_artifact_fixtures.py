"""The artifact fixture the static-verification tests are written against:
a real gzipped dump (COPY blocks in Postgres' text format) beside a real
manifest.json, in a temp directory.

Beside _pathfix.py rather than inside one test module: that is the same
input a recipient has, and more than one module now asks profile_checks.py
about it (the checks themselves, and the manifest-version gate that decides
whether they run at all).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

from manifest_contract import MANIFEST_SCHEMA_VERSION

DOCUMENT_COLUMNS = ["id", "filename", "legal_class", "public_distribution",
                    "legal_note", "source_blob", "source_sha256"]
PAGE_COLUMNS = ["document_id", "page_number", "body", "embedding"]

FULL_DOC = "2009_isu34"
META_DOC = "1997_sm280"
INTERNAL_DOC = "INDEX"
EXCLUDED_DOC = "2016_vmj598"


def _document_row(doc_id, distribution, blob):
    return [doc_id, f"{doc_id}.pdf", "cc-by-4.0", distribution, "основание",
            blob, "a" * 64]


def _page_row(doc_id, page, body, embedding="[0.1,0.2]"):
    return [doc_id, str(page), body, embedding]


def _copy_block(table, columns, rows):
    lines = [f"COPY corpus.{table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


def _citation_copy_block(table, columns, rows):
    lines = [f"COPY citation.{table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


class ArtifactBuilder:
    """Writes a manifest + gzipped dump pair into a directory, defaulting to
    a well-formed public artifact: one full-text document with content, one
    metadata-only document stripped, one internal document shipped whole,
    and one excluded document the manifest names but the dump does not
    contain in any form.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.profile = "public"
        self.schemas = ["corpus"]
        self.documents = [
            _document_row(FULL_DOC, "full-text", "\\x2550"),
            _document_row(META_DOC, "metadata-only", "\\N"),
            _document_row(INTERNAL_DOC, "internal", "\\x2551"),
        ]
        self.pages = [
            _page_row(FULL_DOC, 1, "текст статьи"),
            _page_row(META_DOC, 1, ""),
            _page_row(META_DOC, 2, ""),
            _page_row(INTERNAL_DOC, 1, "наш индекс"),
        ]
        self.page_columns = list(PAGE_COLUMNS)
        self.by_distribution = {
            "full-text": [FULL_DOC], "metadata-only": [META_DOC], "internal": [INTERNAL_DOC],
            "excluded": [EXCLUDED_DOC],
        }
        self.shipped = ["full-text", "metadata-only", "internal"]
        self.unclassified = 0
        self.extra_sql = ""
        # None = this build shipped no citation schema; the manifest still
        # carries the block saying so, because "the graph does not travel"
        # is a decision the artifact has to name (citation_policy_check).
        # citation_block=False drops the key entirely -- a hole, not a
        # decision, and the certification must refuse it.
        self.citation: dict | None = None
        self.citation_block = True

    def write(self) -> Path:
        dump_text = (
            "CREATE SCHEMA corpus;\n"
            "CREATE TABLE corpus.documents (id text);\n"
            + self.extra_sql
            + _copy_block("documents", DOCUMENT_COLUMNS, self.documents)
            + _copy_block("pages", self.page_columns, self.pages)
        )
        if self.citation is not None and self.citation["mode"] != "none":
            dump_text += _citation_copy_block(
                "work", self.citation.get("work_columns", []), self.citation.get("work", []))
            dump_text += _citation_copy_block(
                "cites", self.citation.get("cites_columns", []), self.citation.get("cites", []))
        dump_path = self.directory / "01_dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        legal = {
            "verify_query": "SELECT count(*) ...",
            "unclassified_documents": self.unclassified,
            "class_counts": [],
            "documents_by_distribution": self.by_distribution,
            "full_content_distributions": ["full-text", "internal"],
        }
        if self.shipped is not None:
            legal["shipped_distributions"] = self.shipped
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": self.profile,
            "schemas": self.schemas,
            "pages_count": len(self.pages),
            "dump": {"file": dump_path.name, "bytes": dump_path.stat().st_size, "sha256": "x"},
            "legal": legal,
        }
        source = "owner" if self.profile == "public" else "not-applicable"
        if self.citation_block:
            citation = self.citation or {"mode": "none", "work_count": 0, "cites_count": 0}
            manifest["citation"] = {
                "mode": citation["mode"],
                "policy_source": citation.get("policy_source", source),
                "work_count": citation["work_count"],
                "cites_count": citation["cites_count"],
                "work_by_kind": citation.get("work_by_kind", {}),
            }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        return self.directory
