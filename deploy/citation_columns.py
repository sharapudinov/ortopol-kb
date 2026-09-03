"""What every column of every dumped citation table IS: topology, or
content the topology-only mode must not ship.

The engine is column_classes.py, shared with the corpus half
(corpus_columns.py); here is schema citation's own declaration and the two
readers on opposite sides of the artifact boundary it answers:
deploy/citation_dump.py builds the COPY select from it, and
deploy/citation_content_checks.py -- which travels INSIDE the package, so
the recipient can re-answer "does this dump match the mode it declares"
without a database or this repository -- holds the dump to it. That is why
this is a module of its own rather than a section of citation_profile.py:
that one reads the live database and stays behind with the builder, and a
second hand-written copy of the classification on the artifact side is
exactly the drift the checker exists to catch.

The classification itself, and the reasoning behind each group:

  abstract, evidence    third-party text carried verbatim -- the abstract
      itself, and the source records `evidence` keeps so a verdict can be
      re-derived without re-fetching. This is what topology-only exists to
      withhold; it is the one thing the mode's name promises.
  reason                our own prose, but prose about somebody else's
      work, and free to quote its title. The journal's machine-readable
      facts are columns of their own (JOURNAL_FACTS_ARE_COLUMNS), so a
      blanked `reason` costs a query on the artifact nothing.
  embedding             1024 floats, not text. A vector ranks and clusters;
      it does not reproduce the abstract it was computed from. The same
      call the corpus already makes for corpus.pages embeddings of
      metadata-only documents, which ship while their text does not.
  everything else       bibliography and provenance: key, doi, title, year,
      authors, external_ids, source, kind, document_id, exclusion_reason,
      fetched_at on a work; the endpoints and source of an edge; the whole
      of crawl_step, public_policy and schema_backfill otherwise, which
      are OUR journal, OUR decision record and OUR migration bookkeeping,
      and name no third party's content. A citation IS
      the bibliography -- a skeleton with the titles cut out is not a
      lighter artifact, it is an empty one.
"""
from __future__ import annotations

from column_classes import CONTENT, TOPOLOGY, ColumnClasses, ColumnUnclassified  # noqa: F401

CITATION_COLUMN_CLASS: dict[str, dict[str, str]] = {
    "work": {
        "id": TOPOLOGY, "key": TOPOLOGY, "doi": TOPOLOGY, "title": TOPOLOGY,
        "abstract": CONTENT, "year": TOPOLOGY, "authors": TOPOLOGY,
        "external_ids": TOPOLOGY, "source": TOPOLOGY, "kind": TOPOLOGY,
        "document_id": TOPOLOGY, "exclusion_reason": TOPOLOGY,
        "evidence": CONTENT, "fetched_at": TOPOLOGY, "embedding": TOPOLOGY,
    },
    "cites": {
        "citing": TOPOLOGY, "cited": TOPOLOGY, "source": TOPOLOGY,
        "evidence": CONTENT, "fetched_at": TOPOLOGY,
    },
    "crawl_step": {
        "id": TOPOLOGY, "crawl_id": TOPOLOGY, "depth": TOPOLOGY,
        "frontier_key": TOPOLOGY, "candidate_key": TOPOLOGY, "action": TOPOLOGY,
        "n_found": TOPOLOGY, "n_kept": TOPOLOGY, "reason": CONTENT,
        "at": TOPOLOGY, "node_key": TOPOLOGY, "score": TOPOLOGY,
        "tau": TOPOLOGY, "relation": TOPOLOGY, "cited_by_count": TOPOLOGY,
    },
    "public_policy": {
        "id": TOPOLOGY, "mode": TOPOLOGY, "note": TOPOLOGY, "decided_at": TOPOLOGY,
    },
    "schema_backfill": {"name": TOPOLOGY, "applied_at": TOPOLOGY},
}

# What a blanked column writes instead of its value. Separate from the
# classification because it answers a different question -- not "may this
# leave" but "what does a COPY select's column list need so the value is
# typed" (an untyped NULL leaves Postgres guessing, and some psql builds
# refuse to restore it). The whole schema is blanked at once, by mode, so
# every replacement here is a typed NULL; the corpus half substitutes per
# document instead and its map says so.
CONTENT_WITHHELD = {
    ("work", "abstract"): "NULL::text",
    ("work", "evidence"): "NULL::jsonb",
    ("cites", "evidence"): "NULL::jsonb",
    ("crawl_step", "reason"): "NULL::text",
}

# Which columns of the journal NAME something -- a document id or a work key,
# from either vocabulary: frontier_key is a document id on seed/twin rows and
# a work key on the rest, candidate_key is the record the decision was about,
# node_key is the node it resolved to (a seed work on a twin promotion, see
# citations/journal.py). So the cut matches all three against both
# vocabularies, and the artifact-side check collects all three.
#
# Declared HERE, beside the column classes, for the reason this module
# exists at all: the producer builds the journal cut's three-branch UNION
# from it (deploy/citation_profile.py) and the recipient's bundled checker
# collects exactly those columns from the dumped rows
# (deploy/citation_content_checks.py). Two hand-written copies could only
# agree by accident, and crawl_step grows a column at a time
# (JOURNAL_FACTS_ARE_COLUMNS): a fourth key column added to the producer's
# UNION alone leaves the checker blind to the very cut it verifies, added to
# the checker alone fails a correct package. CITATION_COLUMN_CLASS cannot
# answer this -- it says whether a column may leave, not what it names.
JOURNAL_KEY_COLUMNS = ("frontier_key", "candidate_key", "node_key")

CITATION = ColumnClasses(
    "citation", CITATION_COLUMN_CLASS, CONTENT_WITHHELD,
    hint="дополните CITATION_COLUMN_CLASS в deploy/citation_columns.py",
    withheld_hint="CONTENT_WITHHELD (deploy/citation_columns.py)",
)


def citation_column_class(table: str, column: str) -> str:
    return CITATION.class_of(table, column)


def content_columns(table: str) -> tuple[str, ...]:
    """The columns topology-only blanks in `table`, in declaration order."""
    return CITATION.content_columns(table)


def blanked_value(table: str, column: str) -> str | None:
    """The typed NULL a topology-only cut writes here, or None to ship the
    column. Raises for a column nobody classified, and for a content column
    with no replacement declared -- either way the build stops instead of
    guessing.
    """
    return CITATION.withheld_value(table, column)
