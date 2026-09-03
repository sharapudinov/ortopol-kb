#!/usr/bin/env python3
"""Every Postgres write the snowball makes, through one seam.

The reads a run starts from are next door in citations/inputs.py; nothing
below writes anywhere except through the two Writer implementations, which
is what makes "--dry-run writes nothing" a property of construction.

The rehearsal implementation of the same contract is in dry_store.py; the
two are built in exactly one place, pg_load_citations.writers_for().

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

from typing import NamedTuple, Protocol, runtime_checkable

from pg_common import vector_literal
from pg_copy import copy_csv_rows
from pg_graph_common import GRAPH_NAME, kind_counts, projection_diff, projection_faults
from pg_graph_common import project as project_graph

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
)

class ProjectionOutcome(NamedTuple):
    """What a mode may say about the graph after the write, and with which
    exit code.

    A projection is a WRITE to Postgres -- citation.project_graph() rewrites
    the citation_graph label tables -- so it belongs to the seam like the
    other four, and GRAPH_IS_PROJECTION makes it a consequence of having
    written to citation.work/cites rather than of which argparse branch ran.
    The caller prints `report` and returns `code` unconditionally: rendered
    behind `if writer.dry`, the flag was back in the procedure the seam took
    it out of, and a programmatic driver calling Snowball.run() without a
    command line either reprojected under a dry writer or left the graph
    stale until corpus_completeness.py reported PROJECTION STALE much later.
    """

    report: str
    code: int


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
    DryRunWriter (next door in dry_store.py, module size) has no database
    to refuse anything, so its number is the ceiling the same batch would
    reach against an empty graph -- which is what an estimate of a crawl is,
    and why the two agree on new ground and diverge over ground already
    covered.

    A Protocol rather than a base class: neither writer inherits anything,
    and the point is to pin the contract the crawl depends on, not to share
    code between a database and a list.

    `dry` is part of the contract for the same reason `counts` is: what the
    caller says about the run afterwards -- "nothing was written", a graph
    projection, an acceptance count -- is a property of WHICH writer it
    built, and asking the writer is the only way to be told the truth about
    it. Asked of the command line instead, the two answers are free to
    disagree, and they already do: --calibrate builds a DryRunWriter with
    --dry-run unset, and today nothing breaks only because argparse
    dispatch order puts the calibration mode first. The measurements seam
    next door (spike_runs.py) carries the same attribute for the same
    reason.
    """

    counts: dict[str, int]
    dry: bool

    def works(self, nodes) -> int: ...

    def edges(self, edges) -> int: ...

    def journal(self, steps) -> int: ...

    def promote(self, merged) -> int: ...

    def project(self) -> ProjectionOutcome: ...

    def census(self) -> str: ...


class PostgresWriter:
    """The live-database implementation of what crawl.py needs written."""

    dry = False

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

    def project(self) -> ProjectionOutcome:
        """Rebuilds citation_graph from what this writer just wrote, and
        checks that the rebuild is faithful.

        Both halves belong here rather than in the caller: the AGE graph is
        a projection of citation.work/cites (GRAPH_IS_PROJECTION), so the
        obligation to rebuild it is the obligation of having written them.

        The verdict is projection_faults() over projection_diff(), the
        value-returning pair pg_graph_common.py documents as the answer --
        not pg_graph.check(), which is the CLI's shape: it prints its own
        "OK: |V|=..." / "MISMATCH: ..." and returns an exit code. Through
        check(), one projection produced two overlapping report lines (the
        caller prints `report` on top), and any embedder of Snowball without
        a command line got library writes to stdout it never asked for.
        """
        vertices, edges = project_graph(self.env)
        written = f"проекция графа: V={vertices} E={edges}"
        seen = projection_diff(self.env)
        if seen is None:
            return ProjectionOutcome(
                f"{written}; графа {GRAPH_NAME} нет в ag_catalog.ag_graph — "
                "проекция не строилась", 1)
        faults = projection_faults(seen)
        if not faults:
            return ProjectionOutcome(f"{written}, сверка: |V|={seen.vertex_n} "
                                     f"|E|={seen.edge_n}", 0)
        return ProjectionOutcome(f"{written}; MISMATCH: " + "; ".join(faults), 1)

    def census(self) -> str:
        """The kind breakdown of the graph, read back out of the table this
        writer just wrote to.

        A method rather than a line the caller guards with `if not
        writer.dry`: the census is only ABOUT the promotion when the
        promotion happened, so which answer is true is a property of which
        writer ran -- the same shape as the measurements seam's hub_stats(),
        and the same reason.
        """
        counts = kind_counts(self.env)
        return "kind после склейки: " + " ".join(
            f"{kind}={n}" for kind, n in sorted(counts.items()))
