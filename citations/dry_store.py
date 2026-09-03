#!/usr/bin/env python3
"""The rehearsal half of the Writer seam (citations/store.py): counts what a
real run would write, keeps a sample of it, and writes nothing.

Its own module for size (kb/CLAUDE.md FILE_SIZE) and along the seam that was
already there -- store.py is the contract and the database; this is the
implementation that has no database. Nothing re-exports it: the one place
that builds either writer is pg_load_citations.writers_for().
"""
from __future__ import annotations

from .store import ProjectionOutcome


class DryRunWriter:
    """Counts what a real run would write, keeps a sample of it, and writes
    nothing.

    Same Writer contract as store.PostgresWriter, per-call counts included: this
    is the estimate a crawl is authorised against, so its numbers have to
    be the numbers the real writer would report over ground it has not
    covered yet. Nothing here can refuse a row, so these are upper bounds:
    against a graph that already carries an edge, the live writer reports
    fewer (see store.Writer's own docstring).

    The rows themselves are kept only up to SAMPLE_LIMIT per kind. The
    live writer streams a level through copy_csv_rows and frees it, so its
    peak follows ONE level; retaining every row made the rehearsal's peak
    follow the whole crawl instead -- a depth-2 journal is ~100k rows
    (pg_copy.py), and the mode whose job is to cost a run cheaply held
    them all on the machine about to make it. What reads them is a report
    about the shape of the rows, and a report reads a sample: the counts
    above are the quantity, these are the specimen.
    """

    dry = True

    # Enough to see every kind of row a level produces, few enough that
    # the whole crawl's worth is a rounding error next to one level's.
    SAMPLE_LIMIT = 50

    def __init__(self, source: str = "openalex"):
        self.source = source
        self.works_seen, self.edges_seen, self.steps_seen = [], [], []
        self.promoted_seen = []
        self.counts = {"work": 0, "cites": 0, "step": 0, "twin": 0}

    def _sample(self, kept: list, rows: list) -> int:
        """Extends `kept` up to SAMPLE_LIMIT and returns how many rows the
        batch held -- the count is of everything, the list is of the first
        few. Emptying a sample makes room again, which is how a caller
        asks for the specimen of the NEXT phase rather than of the run.
        """
        room = self.SAMPLE_LIMIT - len(kept)
        if room > 0:
            kept += rows[:room]
        return len(rows)

    def works(self, nodes) -> int:
        accepted = self._sample(self.works_seen, list(nodes))
        self.counts["work"] += accepted
        return accepted

    def edges(self, edges) -> int:
        accepted = self._sample(self.edges_seen, list(edges))
        self.counts["cites"] += accepted
        return accepted

    def journal(self, steps) -> int:
        accepted = self._sample(self.steps_seen, list(steps))
        self.counts["step"] += accepted
        return accepted

    def promote(self, merged) -> int:
        accepted = self._sample(self.promoted_seen, list(merged))
        self.counts["twin"] += accepted
        return accepted

    def project(self) -> ProjectionOutcome:
        """No rows were written, so there is nothing to project and nothing
        to check -- and the mode says so through the same object every other
        mode reports through, rather than through a flag the caller reads
        back off the writer.
        """
        return ProjectionOutcome(
            "--dry-run: в базу ничего не записано, граф не пересобирался", 0)

    def census(self) -> str:
        """Nothing was promoted, so the graph's kind breakdown is whatever it
        was before this rehearsal -- which is not an answer about the run,
        and is not offered as one.
        """
        return "--dry-run: состав kind в графе не менялся"
