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
import tempfile
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_content_checks
import corpus_content_checks
import dump_scan
import profile_checks
from citation_columns import CENSUS_COLUMN
from dump_integrity import sha256_file
from manifest_contract import CitationMode
from manifest_keys import MANIFEST_SCHEMA_VERSION


def dump_facts(citation=None, documents=()):
    """The record pair profile_checks._visit() returns, for a test that
    scans only one half of a dump.

    Built through the same attach_visitors() the pass builds it with, so a
    test cannot hand a check a shape the production pass never produces --
    which is the whole reason the facts stopped being a dict.
    """
    corpus = corpus_content_checks.attach_visitors({})
    corpus.documents.update(documents)
    if citation is None:
        citation = citation_content_checks.attach_visitors(
            {}, CitationMode.TOPOLOGY_ONLY, cut_applies=True)
    return profile_checks.DumpFacts(corpus=corpus, citation=citation)


def citation_copy_block(table: str, columns: list[str], rows: list[list[str]]) -> str:
    """One COPY block as a dump carries it, for a table named in full."""
    lines = [f"COPY {table} ({', '.join(columns)}) FROM stdin;"]
    lines += ["\t".join(row) for row in rows]
    lines += ["\\.", ""]
    return "\n".join(lines)


def citation_scan(dump_text: str, mode: str = CitationMode.TOPOLOGY_ONLY, *,
                  cut_applies: bool = True) -> tuple[dict, "object"]:
    """A REAL dump_scan.scan() pass over `dump_text` under `mode`, with the
    citation visitors attached: (scans, facts).

    The facts every citation check reads are collected by
    citation_content_checks.attach_visitors() on that same pass, so a check
    written against a hand-made fact container would be a check about a
    shape this repository never builds. Topology-only by default: the mode
    the content hunt exists for. cut_applies defaults to True, the case the
    journal's key columns are collected in -- an artifact that classifies
    some document out.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        row_visitors: dict = {}
        facts = citation_content_checks.attach_visitors(
            row_visitors, mode, cut_applies=cut_applies)
        scans = dump_scan.scan(dump_path, row_visitors).tables
    return scans, dump_facts(facts)


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
        self.full_content = ["full-text", "internal"]
        self.unclassified = 0
        self.extra_sql = ""
        # None = declare exactly the corpus blocks write() puts in the dump.
        self.corpus_table_rows: dict | None = None
        # dump{}: None = describe the file write() actually produced; a
        # value replaces the block wholesale. dump_key=False drops the key
        # entirely -- a manifest that names no dump at all, which is what
        # the pass builds the path it opens out of.
        self.dump: object | None = None
        self.dump_key = True
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
        # {table: rows} as this fixture actually wrote them, so a package
        # built here declares what it carries the way the packager does
        # (manifest.citation.table_rows). A test about a MISMATCH sets
        # citation["table_rows"] itself and the declaration below stands
        # instead of this one.
        shipped_tables: dict[str, int] = {}
        census: dict[str, int] = {}
        if self.citation is not None and self.citation["mode"] != "none":
            for table, columns_key in (("work", "work_columns"),
                                       ("cites", "cites_columns")):
                columns = self.citation.get(columns_key, [])
                rows = self.citation.get(table, [])
                dump_text += _citation_copy_block(table, columns, rows)
                shipped_tables[table] = len(rows)
                # The kind census the manifest declares, taken from the rows
                # this fixture actually writes -- the same polarity as
                # table_rows above, and the same one the packager has (the
                # census comes off the COPY stream, copy_rows.FieldTally).
                # A test about a MISMATCH sets citation["work_by_kind"]
                # itself and this stands aside.
                if table == "work" and CENSUS_COLUMN in columns:
                    at = columns.index(CENSUS_COLUMN)
                    for row in rows:
                        census[row[at]] = census.get(row[at], 0) + 1
            # The journal ships under every mode that ships the schema, and
            # its cut is the one this fixture is asked about most: a row
            # naming EXCLUDED_DOC in any of the three key columns is the
            # leak deploy/citation_cut_checks.py exists to catch.
            if self.citation.get("crawl_step_columns"):
                rows = self.citation.get("crawl_step", [])
                dump_text += _citation_copy_block(
                    "crawl_step", self.citation["crawl_step_columns"], rows)
                shipped_tables["crawl_step"] = len(rows)
        dump_path = self.directory / "01_dump.sql.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(dump_text)
        legal = {
            "verify_query": "SELECT count(*) ...",
            "unclassified_documents": self.unclassified,
            "class_counts": [],
            "documents_by_distribution": self.by_distribution,
            "full_content_distributions": self.full_content,
        }
        if self.shipped is not None:
            legal["shipped_distributions"] = self.shipped
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": self.profile,
            "schemas": self.schemas,
            "pages_count": len(self.pages),
            # The corpus half of the per-table declaration, taken from the
            # blocks this fixture actually writes -- the same polarity the
            # citation block below has, and the same one the packager has
            # (manifest_rows.py stamps both from one tally). A test about a
            # MISMATCH sets corpus_table_rows itself.
            "corpus": {"table_rows": self.corpus_table_rows if self.corpus_table_rows
                       is not None else {"documents": len(self.documents),
                                         "pages": len(self.pages)}},
            "legal": legal,
        }
        if self.dump_key:
            # Length and digest of the file this fixture just wrote: the
            # certifier holds the dump to them before reading a byte of its
            # contents (dump_integrity.check_dump_matches_manifest), so a
            # package built here has to be internally honest the way the
            # packager's is. A test about TAMPERING edits one of them (or
            # the file) itself.
            manifest["dump"] = self.dump if self.dump is not None else {
                "file": dump_path.name, "bytes": dump_path.stat().st_size,
                "sha256": sha256_file(dump_path)}
        source = "owner" if self.profile == "public" else "not-applicable"
        if self.citation_block:
            citation = self.citation or {"mode": "none", "work_count": 0, "cites_count": 0}
            manifest["citation"] = {
                "mode": citation["mode"],
                "policy_source": citation.get("policy_source", source),
                "work_count": citation["work_count"],
                "cites_count": citation["cites_count"],
                "work_by_kind": citation.get("work_by_kind", census),
                "table_rows": citation.get("table_rows", shipped_tables),
            }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        return self.directory
