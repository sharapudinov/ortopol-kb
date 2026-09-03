"""What every column of every dumped corpus table IS: topology, or content
the legal cut withholds from a document not cleared for it.

The engine and the reasoning behind the polarity are column_classes.py;
this is schema corpus's own declaration, and it has the same two readers on
opposite sides of the artifact boundary the citation half has:
deploy/public_dump.py builds the COPY select from it, and
deploy/corpus_content_checks.py -- which travels INSIDE the package -- holds
the shipped bytes to it. Both read THIS map: the checker used to name
source_blob and body as constants of its own, i.e. to agree with the
producer by coincidence, on the one question it exists to answer.

Content here is what corpus.documents.public_distribution decides per
document (legal_profile.py, LEGAL_IS_DATA), not a whole-schema mode:

  documents.source_blob   the publisher's own PDF/djvu file, byte for byte.
      Withheld as NULL for a metadata-only document -- the row still ships,
      because the bibliography is precisely what metadata-only means.
  pages.body              the extracted page text. Withheld as the EMPTY
      STRING rather than NULL, and that is load-bearing: the page row still
      exists, so the page's vector stays searchable and the page count
      still matches the source, corpus.pages.body's NOT NULL survives, and
      the generated tsv becomes to_tsvector('russian', '') -- no fulltext
      content, which profile_checks.py verifies rather than assumes.
  everything else         bibliography, provenance and the classification
      itself: ids, filenames, extraction state, source tier, counts, our
      own notes, load timestamps, the source URL and its sha256, the legal
      columns (which the public artifact exists to carry), the page number
      and the page vector, and the whole of embedding_model, which names
      the model every vector was computed with and no document at all.

A page vector ships for a document whose text does not, deliberately:
semantic search finds the DOCUMENT and the reader goes to the publisher for
the text. It is 1024 floats, not a reproduction of the page.
"""
from __future__ import annotations

from column_classes import CONTENT, TOPOLOGY, ColumnClasses, ColumnUnclassified  # noqa: F401

CORPUS_COLUMN_CLASS: dict[str, dict[str, str]] = {
    "documents": {
        "id": TOPOLOGY, "filename": TOPOLOGY, "extraction_state": TOPOLOGY,
        "source_tier": TOPOLOGY, "pages_count": TOPOLOGY,
        "chars_extracted": TOPOLOGY, "note": TOPOLOGY, "loaded_at": TOPOLOGY,
        "source_url": TOPOLOGY, "source_blob": CONTENT,
        "source_sha256": TOPOLOGY, "legal_class": TOPOLOGY,
        "public_distribution": TOPOLOGY, "legal_note": TOPOLOGY,
        "source_dir": TOPOLOGY,
    },
    "pages": {
        "id": TOPOLOGY, "document_id": TOPOLOGY, "page_number": TOPOLOGY,
        "body": CONTENT, "embedding": TOPOLOGY,
    },
    "embedding_model": {
        "id": TOPOLOGY, "model": TOPOLOGY, "dims": TOPOLOGY,
        "computed_at": TOPOLOGY,
    },
}

# What stands in for a withheld value, per column. Not a typed NULL
# throughout, as the citation half's is: these two columns are cut per
# document inside a CASE, and what the row carries instead is part of the
# legal decision -- an absent blob, and an empty (not absent, not NULL)
# page body. The cast on the blob is what keeps the COPY column list typed.
CONTENT_WITHHELD = {
    ("documents", "source_blob"): "NULL::bytea",
    ("pages", "body"): "''",
}

CORPUS = ColumnClasses(
    "corpus", CORPUS_COLUMN_CLASS, CONTENT_WITHHELD,
    hint="дополните CORPUS_COLUMN_CLASS в deploy/corpus_columns.py",
    withheld_hint="CONTENT_WITHHELD (deploy/corpus_columns.py)",
)


def corpus_column_class(table: str, column: str) -> str:
    return CORPUS.class_of(table, column)


def content_columns(table: str) -> tuple[str, ...]:
    """The columns the legal cut withholds in `table`, in declaration order."""
    return CORPUS.content_columns(table)


def withheld_value(table: str, column: str) -> str | None:
    """What the cut writes here instead of the value, or None to ship the
    column as it is. Raises for a column nobody classified, and for a
    content column with no replacement declared.
    """
    return CORPUS.withheld_value(table, column)
