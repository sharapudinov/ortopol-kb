"""The PUBLIC profile's dump: same schema, legally filtered content.

Why not pg_dump: pg_dump has no row- or cell-level filter, so the only
pg_dump-shaped options were (a) build temporary tables/schema in the live
database and dump those -- which writes to the live instance and renames the
schema the whole toolchain reads (corpus -> something else), or (b) dump
everything and post-process a multi-gigabyte SQL stream with sed. Both are
worse than the third option, which this module takes:

    pg_dump --schema-only --schema=corpus   (real DDL, unmodified)
  + COPY (SELECT ...) TO STDOUT             (the rows we may ship)

The result restores into schema `corpus` exactly as the full artifact does
-- same table names, same generated columns, same indexes -- so every
consumer (pg_search.py, smoke_checks.py, AGENT_GUIDE.md's own recipes) works
against it unchanged. measurements is simply not dumped: our own research
records ship as their own artifact, not folded into a corpus package.

What the filter does, per corpus.documents.public_distribution (see
legal_profile.py -- this module contains no legal knowledge and no id list):

  full-text / internal -- every column, blob included; pages keep body
                          (and therefore tsv, which is GENERATED from it)
  metadata-only        -- the documents row WITHOUT source_blob; pages keep
                          page_number + embedding, and body is written as
                          the empty string
  excluded             -- no row is written at all, in either table: the
                          COPY selects filter the document out (SHIPPED_SQL)
                          instead of blanking its columns

The difference between the last two is the whole reason `excluded` exists.
Blanking columns still ships a row saying "this work is in our corpus, here
is its bibliography"; for a document whose legal regime the owner has not
established, that sentence is itself the decision the packager must not
make. Filtering happens in the SELECT, so nothing about the document reaches
the dump file even in a form a reader could count.

body='' rather than "no page row" is deliberate and load-bearing: the pages
row still exists, so the page's vector is still searchable (semantic search
finds the DOCUMENT, the reader then goes to the publisher's site for the
text), the page count still matches the source, and corpus.pages.body's NOT
NULL survives. Empty body means to_tsvector('russian','') -- an empty tsv,
i.e. no fulltext content, which profile_checks.py verifies rather than
assumes.

Which rows of each table travel, and how each column is projected, is
deploy/corpus_cut.py -- the corpus counterpart of citation_profile.py /
citation_dump.py's own maps. This module assembles the file.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import citation_dump  # noqa: E402
from artifact_bundle import DUMP_COMPRESSLEVEL  # noqa: E402
from copy_rows import DumpedRows  # noqa: E402
from copy_writer import write_copy_block  # noqa: E402
from corpus_cut import plan_corpus  # noqa: E402
from manifest_contract import Profile, base_schemas_for  # noqa: E402
from legal_profile import require_classified  # noqa: E402
from pg_stream import CommandFailed, stream_stdout  # noqa: E402

# What _dump_ddl() asks pg_dump for: the public profile's schemas before
# the citation mode is applied, because citation_dump.dump_citation()
# appends that schema's DDL itself, under the mode. Asked of
# manifest_contract as THAT question rather than derived by naming a mode
# that ships nothing: a policy value is not a mechanism, and the two sides
# of the invariant were aligned only by accident -- pg_dump emitting the
# citation DDL here as well would put duplicated CREATE statements into one
# file, which aborts the restore and which no manifest check catches
# (profile_checks compares schema NAMES, not statements).
PUBLIC_SCHEMAS = base_schemas_for(Profile.PUBLIC)


def _dump_ddl(env: dict, dst: IO[bytes]) -> None:
    schema_args = [f"--schema={name}" for name in PUBLIC_SCHEMAS]
    stream_stdout(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges", "--no-tablespaces",
         *schema_args, "--exclude-schema=citation_graph", "--exclude-schema=ag_catalog"],
        env, dst,
    )


PREAMBLE = (
    "--\n"
    "-- ortopol knowledge base, PUBLIC profile.\n"
    "--\n"
    "-- Schema: real pg_dump --schema-only output for schema corpus (measurements\n"
    "-- is deliberately absent). Data: COPY blocks filtered by\n"
    "-- corpus.documents.public_distribution -- documents classified\n"
    "-- 'metadata-only' carry no source_blob and their pages carry an empty body\n"
    "-- (and therefore an empty tsv), only page_number and embedding; documents\n"
    "-- classified 'excluded' have no row here at all, in either table. See\n"
    "-- manifest.json's `legal` block for the classification this build used\n"
    "-- (documents_by_distribution names every document, shipped_distributions\n"
    "-- says which of those lists this file carries) and deploy/public_dump.py\n"
    "-- for how it was applied. Schema `citation`, when present, is cut per\n"
    "-- manifest.json's `citation` block instead (deploy/citation_profile.py /\n"
    "-- deploy/citation_dump.py) -- a separate policy, decided once for the\n"
    "-- whole crawl record rather than per document. The per-document cut\n"
    "-- still reaches it: a citation.work row naming a document absent from\n"
    "-- this file is absent too, and so are its edges and its journal rows.\n"
    "--\n\n"
)


def dump_public(env: dict, gz_path: Path, *, citation_mode: str) -> DumpedRows:
    """Writes the filtered dump to gz_path, gzip-streamed in one pass, and
    returns {table: rows} per schema for what it actually wrote.

    Those counts are the blocks' own output, counted as it streamed past
    (copy_rows.py), and they are what manifest.json is stamped from: the
    numbers the package declares and the COPY blocks it holds are then one
    fact rather than two readings of a live instance the crawl keeps
    writing to (MANIFEST_DESCRIBES_ARTIFACT). The recipient's check
    compares them against the shipped bytes, and that equality now holds by
    construction.

    Refuses to write anything at all while any document lacks a usable
    classification (legal_profile.require_classified) -- that must stop the
    build, not be quietly assigned a default.

    citation_mode is the value build_package.main() resolved once
    (citation_profile, by profile) and handed to the manifest as
    well; this module does not re-read the policy, so the dump and the
    manifest cannot describe two different cuts of the same schema. It has
    no default for the same reason gather_manifest()'s has none: an omitted
    mode would be a cut nobody chose, and CitationMode.NONE happens to be
    the cut that ships nothing.
    """
    # BOTH schemas are resolved here, BEFORE the file is opened. Their
    # refusals -- TableUnclassified, ColumnUnclassified -- are neither
    # CommandFailed, so asked from inside the gzip context they go past the
    # handler below and leave a truncated dump on disk: for the citation
    # half, after the whole corpus DDL and every corpus COPY block; for the
    # corpus half, after the preamble and the DDL. The promise in this
    # docstring is that nothing is written at all, and a refusal keeps that
    # promise only while it is still cheap.
    require_classified(env)
    corpus_plan = plan_corpus(env)
    citation_plan = citation_dump.plan_citation(env, citation_mode)
    try:
        with gzip.open(gz_path, "wb", compresslevel=DUMP_COMPRESSLEVEL) as dst:
            dst.write(PREAMBLE.encode())
            _dump_ddl(env, dst)
            dst.write(b"\n")
            corpus_rows = {block.table: write_copy_block(env, dst, block)
                           for block in corpus_plan}
            dst.write(b"\n")
            citation = citation_dump.dump_citation(env, dst, citation_plan)
    except CommandFailed as exc:
        gz_path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    return DumpedRows(corpus=corpus_rows, citation=citation.tables,
                      work_by_kind=citation.work_by_kind)
