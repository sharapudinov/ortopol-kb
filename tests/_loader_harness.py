"""Everything pg_load_citations.main() reaches on its way to a mode.

Beside _pathfix.py rather than inside one test module: the loader's own
promises and the two spike modes' reporting halves are asserted in separate
modules (kb/CLAUDE.md FILE_SIZE), and both have to stand main() up the same
way. Two copies of that patch list drift the moment a new startup step is
added to one of them.
"""
from __future__ import annotations

from unittest import mock

import _pathfix  # noqa: F401

import pg_graph_common
import pg_load_citations
from citations import threshold_store
from citations.store import DryRunWriter

ENV = {"PGHOST": "test"}


class MainHarness:
    """Everything main() reaches on its way to the mode under test."""

    def __init__(self, stack, *, schema_exists=True):
        self.init_schema = stack.enter_context(
            mock.patch.object(pg_load_citations, "init_schema"))
        self.run_sql_file = stack.enter_context(
            mock.patch.object(pg_graph_common, "run_sql_file"))
        self.upsert_run = stack.enter_context(
            mock.patch.object(threshold_store, "upsert_run", return_value=1))
        stack.enter_context(mock.patch.object(pg_load_citations, "load_pgenv",
                                              return_value=ENV))
        stack.enter_context(mock.patch.object(pg_load_citations, "citation_schema_exists",
                                              return_value=schema_exists))
        self.writers: list[DryRunWriter] = []
        stack.enter_context(mock.patch.object(pg_load_citations, "DryRunWriter",
                                              side_effect=self._writer))

    def _writer(self, *_args, **_kwargs):
        self.writers.append(DryRunWriter())
        return self.writers[-1]


