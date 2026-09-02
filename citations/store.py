#!/usr/bin/env python3
"""Every Postgres write the snowball makes, through one seam.

The reads a run starts from are next door in citations/inputs.py; nothing
below writes anywhere except through the two Writer implementations, which
is what makes "--dry-run writes nothing" a property of construction.

Same two mechanisms as the rest of the repository (pg_common.py): script
variables for parameters, `\\copy` from a csv module-built temp file for
bulk -- no string interpolation of source-controlled text anywhere near
SQL, because titles and abstracts here come from a third party. The
statements themselves are next door in store_sql.py: this module is who
writes, that one is what is written.

Every bulk write is ONE psql script -- staging DDL, the `\\copy`, the upsert
that consumes it -- and the staging relation is a TEMP table, so a writer
running at the same time cannot see it and neither can drop the other's
rows between two invocations. Each method returns what the database
ACCEPTED, which the statements themselves count.

Upserts, never truncate-and-reload. The rule LOADERS_PRESERVE was paid for
twice in this project, and it has two specific consequences below:

- an existing `our-document` or `indexed` row is NOT demoted to
  `external-skeleton` when the crawl meets the same work again as a
  stranger's citation;
- an embedding survives unless the row arrives with a new one; a re-crawl
  that changed nothing must not send pg_embed.py back over the whole graph.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pg_copy import copy_csv_rows

from .store_sql import (
    CITES_COLUMNS,
    PROMOTE_COLUMNS,
    STEP_COLUMNS,
    WORK_COLUMNS,
    CITES_STAGE_DDL,
    CITES_UPSERT,
    PROMOTE_STAGE_DDL,
    PROMOTE_UPDATE,
    WORK_STAGE_DDL,
    WORK_UPSERT,
    json_or_null,
    vector_literal,
)

# -- writes --------------------------------------------------------------
@runtime_checkable
class Writer(Protocol):
    """The seam crawl.py writes through, and the counting convention both
    implementations follow.

    Every method takes the rows produced by ONE step of the crawl and
    returns how many of them IT accepted -- not the running total, and not
    the number submitted. counts accumulates those same per-call numbers
    under 'work' / 'cites' / 'step' / 'twin'. Spelled out here because the
    two implementations had drifted into three conventions between them, and
    the dry run's numbers are what the decision to spend a real quota window
    is made on.

    "Accepted" is the database's own count for PostgresWriter: the upserts
    refuse rows on purpose (a self-edge left by the twin union, an edge the
    graph already carries, a promote key with no work row), so each
    statement ends by counting what it wrote and that is what comes back.
    DryRunWriter has no database to refuse anything, so its number is the
    ceiling the same batch would reach against an empty graph -- which is
    what an estimate of a crawl is, and why the two agree on new ground and
    diverge over ground already covered.

    A Protocol rather than a base class: neither writer inherits anything,
    and the point is to pin the contract the crawl depends on, not to share
    code between a database and a list.
    """

    counts: dict[str, int]

    def works(self, nodes) -> int: ...

    def edges(self, edges) -> int: ...

    def journal(self, steps) -> int: ...

    def promote(self, merged) -> int: ...


class PostgresWriter:
    """The live-database implementation of what crawl.py needs written."""

    def __init__(self, env, source: str = "openalex"):
        self.env = env
        self.source = source
        self.counts = {"work": 0, "cites": 0, "step": 0, "twin": 0}

    def works(self, nodes) -> int:
        if not nodes:
            return 0
        accepted = copy_csv_rows(
            self.env,
            f"stage_work ({', '.join(WORK_COLUMNS)})",
            (self._work_row(node) for node in nodes),
            preamble=WORK_STAGE_DDL,
            epilogue=WORK_UPSERT,
        ).accepted()
        self.counts["work"] += accepted
        return accepted

    def _work_row(self, node) -> list:
        return [
            node.key,
            node.doi,
            node.title,
            node.abstract,
            node.year,
            json_or_null(node.authors or None),
            json_or_null(node.external_ids()),
            self.source,
            node.kind,
            node.document_id,
            json_or_null(self.evidence_of(node)),
            vector_literal(getattr(node, "embedding", None)),
        ]

    @staticmethod
    def evidence_of(node) -> dict:
        """The raw source records, plus how the abstract was obtained --
        a zbMATH review standing in for a missing OpenAlex abstract must be
        re-derivable as such without another network call."""
        evidence = {"records": node.records}
        if node.abstract_source:
            evidence["abstract_source"] = node.abstract_source
        if getattr(node, "zbmath_id", None):
            evidence["zbmath_id"] = node.zbmath_id
        if node.relation:
            evidence["relation"] = node.relation
        if node.discovered_from:
            evidence["discovered_from"] = node.discovered_from
        if node.score is not None:
            evidence["frontier_score"] = round(node.score, 6)
        return evidence

    def edges(self, edges) -> int:
        """Accepted is what the INSERT took: a self-edge (two OpenAlex
        records of one work after the twin union) is filtered out by the
        statement, and an edge the graph already carries is skipped by ON
        CONFLICT. Reporting the submitted count instead made a re-crawl of
        known ground look like a level of new edges.
        """
        if not edges:
            return 0
        accepted = copy_csv_rows(
            self.env,
            f"stage_cites ({', '.join(CITES_COLUMNS)})",
            ([citing, cited, self.source,
              json_or_null({"relation": relation, "fetched_from": fetched_from})]
             for citing, cited, relation, fetched_from in edges),
            preamble=CITES_STAGE_DDL,
            epilogue=CITES_UPSERT,
        ).accepted()
        self.counts["cites"] += accepted
        return accepted

    def journal(self, steps) -> int:
        """The journal has no upsert to refuse a row: COPY takes every step
        or the whole batch fails, so accepted is what was streamed.
        """
        if not steps:
            return 0
        written = copy_csv_rows(
            self.env,
            f"citation.crawl_step ({', '.join(STEP_COLUMNS)})",
            ([step.get(column) for column in STEP_COLUMNS] for step in steps),
        ).rows
        self.counts["step"] += written
        return written

    def promote(self, merged) -> int:
        """Promotes the whole batch in ONE statement, staged the way every
        other bulk write here is: a psql process, connection and transaction
        per promoted node is the N+1 write pattern, and it also made the
        pass non-atomic -- a failure halfway through left some nodes
        promoted and no journal rows written at all.

        Idempotent, as the per-row form was: rerunning over already-promoted
        nodes writes the same values again.
        """
        if not merged:
            return 0
        accepted = copy_csv_rows(
            self.env,
            f"stage_twin ({', '.join(PROMOTE_COLUMNS)})",
            ([m["key"], m["document_id"], m["seed_key"], m["rule"]] for m in merged),
            preamble=PROMOTE_STAGE_DDL,
            epilogue=PROMOTE_UPDATE,
        ).accepted()
        self.counts["twin"] += accepted
        return accepted


class DryRunWriter:
    """Collects what a real run would write, and writes nothing.

    Same Writer contract as PostgresWriter, per-call counts included: this
    is the estimate a crawl is authorised against, so its numbers have to
    be the numbers the real writer would report over ground it has not
    covered yet. Nothing here can refuse a row, so these are upper bounds:
    against a graph that already carries an edge, the live writer reports
    fewer (see the Writer contract above).
    """

    def __init__(self, source: str = "openalex"):
        self.source = source
        self.works_seen, self.edges_seen, self.steps_seen = [], [], []
        self.promoted_seen = []
        self.counts = {"work": 0, "cites": 0, "step": 0, "twin": 0}

    def works(self, nodes) -> int:
        accepted = list(nodes)
        self.works_seen += accepted
        self.counts["work"] += len(accepted)
        return len(accepted)

    def edges(self, edges) -> int:
        accepted = list(edges)
        self.edges_seen += accepted
        self.counts["cites"] += len(accepted)
        return len(accepted)

    def journal(self, steps) -> int:
        accepted = list(steps)
        self.steps_seen += accepted
        self.counts["step"] += len(accepted)
        return len(accepted)

    def promote(self, merged) -> int:
        accepted = list(merged)
        self.promoted_seen += accepted
        self.counts["twin"] += len(accepted)
        return len(accepted)
