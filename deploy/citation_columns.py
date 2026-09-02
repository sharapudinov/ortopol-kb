"""What every column of every dumped citation table IS: topology, or
content the topology-only mode must not ship.

One classification, two readers on opposite sides of the artifact
boundary: deploy/citation_dump.py builds the COPY select from it, and
deploy/citation_content_checks.py -- which travels INSIDE the package, so
the recipient can re-answer "does this dump match the mode it declares"
without a database or this repository -- holds the dump to it. That is why
this is a module of its own rather than a section of citation_profile.py:
that one reads the live database and stays behind with the builder, and a
second hand-written copy of the classification on the artifact side is
exactly the drift the checker exists to catch.

The polarity matters. schema_catalog.schema_columns() reads the column list
from pg_attribute, so no column can silently vanish from the artifact --
but the stripping used to be a hardcoded denylist, so any column not named
in it SHIPPED by default. citation.work.embedding walked straight through
that gap. Here every column is named: an unclassified one raises
ColumnUnclassified and fails the build, the same answer
legal_profile.require_classified gives an unclassified document
(kb/CLAUDE.md UNCLASSIFIED_FAILS_BUILD). Neither shipping nor stripping by
default is available, because both are decisions.

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

TOPOLOGY = "topology"
CONTENT = "content"

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

# The SQL type a blanked column's NULL is cast to. Separate from the
# classification because it answers a different question -- not "may this
# leave" but "what does a COPY select's column list need so the value is
# typed" (an untyped NULL leaves Postgres guessing, and some psql builds
# refuse to restore it). A content column with no cast declared here is as
# unbuildable as an unclassified one.
CONTENT_NULL_CAST = {
    ("work", "abstract"): "text",
    ("work", "evidence"): "jsonb",
    ("cites", "evidence"): "jsonb",
    ("crawl_step", "reason"): "text",
}


class ColumnUnclassified(RuntimeError):
    """A dumped column nobody has said topology or content about."""


def citation_column_class(table: str, column: str) -> str:
    try:
        return CITATION_COLUMN_CLASS[table][column]
    except KeyError:
        raise ColumnUnclassified(
            f"колонка citation.{table}.{column} не классифицирована "
            f"(topology | content) -- дополните CITATION_COLUMN_CLASS в "
            f"deploy/citation_columns.py; сборка отказывается угадывать, "
            f"уезжает ли новая колонка в public-артефакт"
        ) from None


def content_columns(table: str) -> tuple[str, ...]:
    """The columns topology-only blanks in `table`, in declaration order."""
    return tuple(column for column, kind in CITATION_COLUMN_CLASS[table].items()
                 if kind == CONTENT)


def blanked_cast(table: str, column: str) -> str | None:
    """The cast a topology-only NULL needs here, or None to ship the column.

    Raises for a column nobody classified, and for a content column with no
    declared cast -- either way the build stops instead of guessing.
    """
    if citation_column_class(table, column) == TOPOLOGY:
        return None
    try:
        return CONTENT_NULL_CAST[(table, column)]
    except KeyError:
        raise ColumnUnclassified(
            f"citation.{table}.{column} — content, но без типа для NULL "
            f"в CONTENT_NULL_CAST (deploy/citation_columns.py)"
        ) from None
