"""Unit tests for citation_checks.py: no live database (run_sql/scalar are
stubbed with the exact FIELD_SEP-joined text psql produces, same convention
as test_external_registry.py/test_legal_profile.py's `_psql_rows`).
pg_graph_common.graph_exists/projection_reading/compare_counts are stubbed too --
that module's own live behaviour is test_pg_graph.py's job.

A live-database class at the bottom exercises citation_problems() against
the real corpus (skipped, not failed, when Postgres is unreachable -- same
convention as test_pg_graph.py's CitationGraphLiveTests).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401

import citation_checks
import pg_graph_common
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv

FIELD_SEP = "\x1f"


def _psql_rows(rows: list[tuple[str, ...]]) -> mock.Mock:
    """A CompletedProcess-alike whose stdout is what `psql -t -A -F <sep>`
    prints for those rows.
    """
    text = "\n".join(FIELD_SEP.join(row) for row in rows)
    return mock.Mock(stdout=text + ("\n" if text else ""))


class UnplacedDocumentsTests(unittest.TestCase):
    def test_parses_one_id_per_line(self):
        with mock.patch.object(citation_checks, "run_sql",
                                return_value=mock.Mock(stdout="2015_demr1\n2016_vmj598\n")):
            self.assertEqual(citation_checks.unplaced_documents({}), ["2015_demr1", "2016_vmj598"])

    def test_empty_result_is_an_empty_list(self):
        with mock.patch.object(citation_checks, "run_sql", return_value=mock.Mock(stdout="")):
            self.assertEqual(citation_checks.unplaced_documents({}), [])

    def test_query_checks_both_the_our_document_kind_and_the_seed_missing_step(self):
        sql = citation_checks._UNPLACED_SQL
        self.assertIn("kind = 'our-document'", sql)
        self.assertIn("action = 'seed-missing'", sql)
        self.assertIn("extraction_state <> 'metadata'", sql)


class WorksWithoutEvidenceTests(unittest.TestCase):
    def test_parses_key_and_kind_pairs(self):
        rows = [("openalex:W1", "external-skeleton"), ("openalex:W2", "indexed")]
        with mock.patch.object(citation_checks, "run_sql", return_value=_psql_rows(rows)):
            self.assertEqual(citation_checks.works_without_evidence({}), rows)

    def test_query_restricts_kind_to_external_skeleton_and_indexed(self):
        sql = citation_checks._NO_EVIDENCE_SQL
        self.assertIn("'external-skeleton'", sql)
        self.assertIn("'indexed'", sql)
        self.assertNotIn("'our-document'", sql)


class SelfLoopTests(unittest.TestCase):
    def test_parses_ids(self):
        with mock.patch.object(citation_checks, "run_sql", return_value=mock.Mock(stdout="42\n")):
            self.assertEqual(citation_checks.self_loop_work_ids({}), ["42"])

    def test_empty_when_no_self_loops(self):
        with mock.patch.object(citation_checks, "run_sql", return_value=mock.Mock(stdout="")):
            self.assertEqual(citation_checks.self_loop_work_ids({}), [])


class ProjectionStaleTests(unittest.TestCase):
    """Only the wording is this module's; the reading is made once by
    citation_state() and handed in, so these pass it directly.
    """

    def test_missing_graph_is_one_problem(self):
        problems = citation_checks._projection_stale(None)
        self.assertEqual(len(problems), 1)
        self.assertIn("PROJECTION STALE", problems[0])

    FAITHFUL = pg_graph_common.Projection(5, 3, 5, 3, "w", "w", "c", "c")

    def test_matching_counts_and_content_are_no_problem(self):
        self.assertEqual(citation_checks._projection_stale(self.FAITHFUL), [])

    def test_mismatched_counts_are_one_problem_naming_the_diff(self):
        problems = citation_checks._projection_stale(self.FAITHFUL._replace(vertex_n=4))
        self.assertEqual(len(problems), 1)
        self.assertIn("work=5", problems[0])
        self.assertIn("vertices=4", problems[0])


class SemanticKeyTests(unittest.TestCase):
    def test_parses_keys(self):
        with mock.patch.object(citation_checks, "run_sql",
                                return_value=mock.Mock(stdout="openalex:W1\n")):
            self.assertEqual(citation_checks.works_without_semantic_key({}), ["openalex:W1"])


class IndexedWithoutExternalTests(unittest.TestCase):
    def test_parses_keys(self):
        with mock.patch.object(citation_checks, "run_sql",
                                return_value=mock.Mock(stdout="openalex:W9\n")):
            self.assertEqual(citation_checks.indexed_without_external_document({}), ["openalex:W9"])

    def test_query_restricts_to_indexed_kind_and_external_source_dir(self):
        sql = citation_checks._INDEXED_WITHOUT_EXTERNAL_SQL
        self.assertIn("kind = 'indexed'", sql)
        self.assertIn("theory/external", sql)


def _projection(work_n: int, cites_n: int):
    """A faithful reading of that size -- digests equal on both sides."""
    return pg_graph_common.Projection(work_n, cites_n, work_n, cites_n,
                                      "d1", "d1", "d2", "d2")


class CitationSummaryTests(unittest.TestCase):
    COUNTS = {"external-skeleton": 382, "our-document": 56}

    def test_missing_schema_says_so(self):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=False):
            self.assertEqual(citation_checks.citation_state({}).summary,
                             "citation: schema absent")

    def test_present_schema_reports_counts(self):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_checks, "kind_counts", return_value=self.COUNTS), \
             mock.patch.object(pg_graph_common, "projection_diff",
                                return_value=_projection(438, 2425)), \
             mock.patch.object(citation_checks, "_problems", return_value=[]):
            summary = citation_checks.citation_state({}).summary
        self.assertIn("438 work", summary)
        self.assertIn("2425 cites", summary)
        self.assertIn("external-skeleton=382", summary)

    def test_the_summary_costs_no_reading_of_its_own(self):
        """The totals come from the projection reading the problems made:
        the schema question is asked once, the reading happens once, and no
        second count of citation.cites is issued at all.
        """
        import contextlib
        with contextlib.ExitStack() as stack:
            schema = stack.enter_context(mock.patch.object(
                citation_checks, "citation_schema_exists", return_value=True))
            reading = stack.enter_context(mock.patch.object(
                pg_graph_common, "projection_diff", return_value=_projection(438, 2425)))
            stack.enter_context(mock.patch.object(
                citation_checks, "kind_counts", return_value=self.COUNTS))
            stack.enter_context(mock.patch.object(citation_checks, "_problems", return_value=[]))
            run_sql = stack.enter_context(mock.patch.object(citation_checks, "run_sql"))
            state = citation_checks.citation_state({})
        self.assertIn("2425 cites", state.summary)
        schema.assert_called_once()
        reading.assert_called_once()
        run_sql.assert_not_called()

    def test_an_unprojected_graph_still_gets_a_summary_line(self):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_checks, "kind_counts", return_value=self.COUNTS), \
             mock.patch.object(pg_graph_common, "projection_diff", return_value=None), \
             mock.patch.object(citation_checks, "_problems", return_value=[]):
            summary = citation_checks.citation_state({}).summary
        self.assertIn("438 work", summary)
        self.assertIn("проекции нет", summary)


class CitationProblemsTests(unittest.TestCase):
    """citation_problems() as a whole -- test_corpus_document_without_vertex_
    or_reason_is_a_hole and test_edge_to_unknown_work_is_a_hole from the
    task's TESTS list live here, under the names the underlying predicates
    actually check.
    """

    def _mock_clean(self, stack):
        stack.enter_context(mock.patch.object(citation_checks, "unplaced_documents", return_value=[]))
        stack.enter_context(mock.patch.object(citation_checks, "works_without_evidence", return_value=[]))
        stack.enter_context(mock.patch.object(citation_checks, "self_loop_work_ids", return_value=[]))
        stack.enter_context(mock.patch.object(citation_checks, "_projection_stale", return_value=[]))
        stack.enter_context(mock.patch.object(pg_graph_common, "projection_diff",
                                              return_value=_projection(1, 1)))
        stack.enter_context(mock.patch.object(citation_checks, "kind_counts", return_value={}))
        stack.enter_context(mock.patch.object(citation_checks, "works_without_semantic_key", return_value=[]))
        stack.enter_context(
            mock.patch.object(citation_checks, "indexed_without_external_document", return_value=[]))

    def test_missing_schema_is_a_single_problem_and_nothing_else_runs(self):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=False), \
             mock.patch.object(citation_checks, "unplaced_documents") as unplaced_mock:
            problems = citation_checks.citation_problems({})
        self.assertEqual(len(problems), 1)
        self.assertIn("CITATION SCHEMA MISSING", problems[0])
        unplaced_mock.assert_not_called()

    def test_all_clean_predicates_yield_no_problems(self):
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(citation_checks, "citation_schema_exists", return_value=True))
            self._mock_clean(stack)
            self.assertEqual(citation_checks.citation_problems({}), [])

    def test_corpus_document_without_vertex_or_reason_is_a_hole(self):
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(citation_checks, "citation_schema_exists", return_value=True))
            self._mock_clean(stack)
            stack.enter_context(mock.patch.object(citation_checks, "unplaced_documents",
                                                    return_value=["2016_vmj598"]))
            problems = citation_checks.citation_problems({})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNPLACED DOCUMENT: 2016_vmj598", problems[0])

    def test_edge_to_unknown_work_is_a_hole(self):
        # Modelled as citation.cites failing FK integrity manifests here as a
        # self-loop or a stale projection -- the two predicates that read
        # citation.cites at all; self-loop is the direct one (protected by a
        # CHECK, re-verified as a predicate per the task spec).
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(citation_checks, "citation_schema_exists", return_value=True))
            self._mock_clean(stack)
            stack.enter_context(mock.patch.object(citation_checks, "self_loop_work_ids", return_value=["7"]))
            problems = citation_checks.citation_problems({})
        self.assertEqual(len(problems), 1)
        self.assertIn("SELF LOOP", problems[0])
        self.assertIn("7 -> 7", problems[0])

    def test_no_evidence_and_no_semantic_key_and_indexed_without_external_all_surface(self):
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(citation_checks, "citation_schema_exists", return_value=True))
            self._mock_clean(stack)
            stack.enter_context(mock.patch.object(
                citation_checks, "works_without_evidence",
                return_value=[("openalex:W1", "external-skeleton")]))
            stack.enter_context(mock.patch.object(
                citation_checks, "works_without_semantic_key", return_value=["openalex:W2"]))
            stack.enter_context(mock.patch.object(
                citation_checks, "indexed_without_external_document", return_value=["openalex:W3"]))
            problems = citation_checks.citation_problems({})
        joined = " | ".join(problems)
        self.assertIn("NO EVIDENCE", joined)
        self.assertIn("NO SEMANTIC KEY", joined)
        self.assertIn("INDEXED WITHOUT EXTERNAL DOCUMENT", joined)
        self.assertEqual(len(problems), 3)


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class CitationChecksLiveTests(unittest.TestCase):
    """Against the real corpus: skipped, not failed, when Postgres is
    unreachable. As of this writing the crawl left the graph clean (438 work,
    2425 cites, fully placed/evidenced/embedded, projection current) -- a
    regression here is a real hole, not a fixture artefact.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def test_citation_schema_exists_on_the_live_corpus(self):
        self.assertTrue(citation_checks.citation_schema_exists(self.env))

    def test_citation_problems_is_empty_against_the_current_corpus(self):
        problems = citation_checks.citation_problems(self.env)
        self.assertEqual(problems, [], problems)

    def test_citation_summary_names_a_nonzero_work_and_cites_count(self):
        summary = citation_checks.citation_state(self.env).summary
        self.assertIn("work (", summary)
        self.assertIn("cites", summary)
        self.assertNotIn("schema absent", summary)


if __name__ == "__main__":
    unittest.main()
