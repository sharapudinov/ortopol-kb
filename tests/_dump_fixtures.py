"""Fixtures the two public-dump test modules share.

Beside _pathfix.py rather than in one of them: the corpus half's fake
catalog answer and the citation table list are the same input on both
sides of the split (the corpus dump's own shape, and its handshake with
citation_dump.py), and two copies drift the moment one is corrected.
"""
from __future__ import annotations

# What schema_catalog would answer for schema corpus: the columns each
# table contributes to a COPY block, in one read.
CORPUS_COLUMNS = {
    "documents": ["id", "source_blob"],
    "pages": ["document_id", "page_number", "body", "embedding"],
    "embedding_model": ["id", "model"],
}

# Which corpus columns own a sequence, i.e. which get a setval() after
# their block -- corpus.pages.id is the only BIGSERIAL pg_schema.sql
# declares, and the live assertion that the catalog says so belongs to the
# catalog, not here.
CORPUS_SERIALS = {"pages": ["id"]}

# The citation tables a dump writes today, in restore order -- the live
# assertion that this is what the catalog and the foreign keys say lives in
# test_citation_dump_live.py.
DUMPED_CITATION_TABLES = ("work", "cites", "crawl_step", "public_policy",
                          "schema_backfill")
