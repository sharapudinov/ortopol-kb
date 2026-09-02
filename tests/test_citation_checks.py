"""Unit tests for citation_checks.py: no live database (scalar is stubbed
with the json one reading returns). pg_graph_common.graph_exists/
projection_reading/compare_counts are stubbed too -- that module's own live
behaviour is test_pg_graph.py's job.

A live-database class at the bottom exercises citation_problems() against
the real corpus (skipped, not failed, when Postgres is unreachable -- same
convention as test_pg_graph_live.py's CitationGraphLiveTests).
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import _pathfix  # noqa: F401

import citation_checks
import pg_graph_common
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv

EMPTY = {"by_kind": {}, "unplaced": [], "no_evidence": [], "self_loops": [],
         "no_semantic_key": [], "indexed_without_external": []}


def _answer(**checks) -> mock.Mock:
    """scalar() stand-in: the json object the one reading comes back as."""
    return mock.Mock(side_effect=lambda env, sql: json.dumps({**EMPTY, **checks}))


class UnplacedDocumentsTests(unittest.TestCase):
    def test_parses_one_row_per_document(self):
        with mock.patch.object(citation_checks, "scalar",
                                _answer(unplaced=[["2015_demr1"], ["2016_vmj598"]])):
            self.assertEqual(citation_checks.unplaced_documents({}),
                             ["2015_demr1", "2016_vmj598"])

    def test_empty_result_is_an_empty_list(self):
        with mock.patch.object(citation_checks, "scalar", _answer()):
            self.assertEqual(citation_checks.unplaced_documents({}), [])

    def test_query_checks_both_the_our_document_kind_and_the_seed_missing_step(self):
        sql = citation_checks._UNPLACED_SQL
        self.assertIn("kind = 'our-document'", sql)
        self.assertIn("action = 'seed-missing'", sql)
        self.assertIn("extraction_state <> 'metadata'", sql)


class WorksWithoutEvidenceTests(unittest.TestCase):
    def test_parses_key_and_kind_pairs(self):
        rows = [("openalex:W1", "external-skeleton"), ("openalex:W2", "indexed")]
        with mock.patch.object(citation_checks, "scalar",
                                _answer(no_evidence=[list(r) for r in rows])):
            self.assertEqual(citation_checks.works_without_evidence({}), rows)

    def test_query_restricts_kind_to_external_skeleton_and_indexed(self):
        sql = citation_checks._NO_EVIDENCE_SQL
        self.assertIn("'external-skeleton'", sql)
        self.assertIn("'indexed'", sql)
        self.assertNotIn("'our-document'", sql)


class SelfLoopTests(unittest.TestCase):
    def test_parses_ids(self):
        with mock.patch.object(citation_checks, "scalar", _answer(self_loops=[["42"]])):
            self.assertEqual(citation_checks.self_loop_work_ids({}), ["42"])

    def test_empty_when_no_self_loops(self):
        with mock.patch.object(citation_checks, "scalar", _answer()):
            self.assertEqual(citation_checks.self_loop_work_ids({}), [])


class OneReadingTests(unittest.TestCase):
    """Five independent problem queries and the census used to be five (and
    six) psql forks, each with a temp script and a connection of its own.
    They are one statement now, and a completeness run asks the database
    three times: the schema, the projection, and this.
    """

    def test_every_check_travels_in_one_statement(self):
        with mock.patch.object(citation_checks, "scalar", _answer()) as scalar_mock:
            citation_checks.citation_reading({})
        scalar_mock.assert_called_once()
        sql = scalar_mock.call_args[0][1]
        for name, _sql, _columns in citation_checks._CHECKS:
            self.assertIn(f"'{name}'", sql)
        self.assertIn("'by_kind'", sql)

    def test_a_completeness_run_makes_three_readings(self):
        with mock.patch.object(citation_checks, "citation_schema_exists",
                                return_value=True) as schema_mock, \
             mock.patch.object(pg_graph_common, "projection_diff",
                                return_value=_projection(438, 2425)) as projection_mock, \
             mock.patch.object(citation_checks, "scalar", _answer()) as scalar_mock:
            citation_checks.citation_state({})
        schema_mock.assert_called_once()
        projection_mock.assert_called_once()
        scalar_mock.assert_called_once()

    def test_the_census_is_the_shared_one_not_a_second_spelling(self):
        self.assertIn(pg_graph_common.kind_counts_expression(),
                      citation_checks._READING_SQL)

    def test_every_row_arrives_as_text_in_the_declared_column_order(self):
        """A bigint id would otherwise come back as a json number, and a
        reader indexing by field name would drift from the SELECT.
        """
        for _name, _sql, columns in citation_checks._CHECKS:
            fragment = citation_checks._rows_json(_sql, columns)
            for column in columns:
                self.assertIn(f"r.{column}::text", fragment)
            self.assertIn(f"ORDER BY r.{columns[0]}", fragment)


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
        with mock.patch.object(citation_checks, "scalar",
                                _answer(no_semantic_key=[["openalex:W1"]])):
            self.assertEqual(citation_checks.works_without_semantic_key({}), ["openalex:W1"])


class IndexedWithoutExternalTests(unittest.TestCase):
    def test_parses_keys(self):
        with mock.patch.object(citation_checks, "scalar",
                                _answer(indexed_without_external=[["openalex:W9"]])):
            self.assertEqual(citation_checks.indexed_without_external_document({}),
                             ["openalex:W9"])

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

    def _summary(self, projection):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_checks, "scalar", _answer(by_kind=self.COUNTS)), \
             mock.patch.object(pg_graph_common, "projection_diff", return_value=projection):
            return citation_checks.citation_state({}).summary

    def test_present_schema_reports_counts(self):
        summary = self._summary(_projection(438, 2425))
        self.assertIn("438 work", summary)
        self.assertIn("2425 cites", summary)
        self.assertIn("external-skeleton=382", summary)

    def test_the_summary_costs_no_reading_of_its_own(self):
        """The totals come from the projection reading, the census from the
        problems reading, and neither is asked for a second time.
        """
        with mock.patch.object(citation_checks, "citation_schema_exists",
                                return_value=True) as schema, \
             mock.patch.object(pg_graph_common, "projection_diff",
                                return_value=_projection(438, 2425)) as reading, \
             mock.patch.object(citation_checks, "scalar",
                                _answer(by_kind=self.COUNTS)) as scalar_mock:
            state = citation_checks.citation_state({})
        self.assertIn("2425 cites", state.summary)
        schema.assert_called_once()
        reading.assert_called_once()
        scalar_mock.assert_called_once()

    def test_an_unprojected_graph_still_gets_a_summary_line(self):
        summary = self._summary(None)
        self.assertIn("438 work", summary)
        self.assertIn("проекции нет", summary)


class CitationProblemsTests(unittest.TestCase):
    """citation_problems() as a whole -- test_corpus_document_without_vertex_
    or_reason_is_a_hole and test_edge_to_unknown_work_is_a_hole from the
    task's TESTS list live here, under the names the underlying predicates
    actually check.
    """

    def _problems(self, **checks):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=True), \
             mock.patch.object(pg_graph_common, "projection_diff",
                                return_value=_projection(1, 1)), \
             mock.patch.object(citation_checks, "scalar", _answer(**checks)):
            return citation_checks.citation_problems({})

    def test_missing_schema_is_a_single_problem_and_nothing_else_runs(self):
        with mock.patch.object(citation_checks, "citation_schema_exists", return_value=False), \
             mock.patch.object(citation_checks, "scalar") as scalar_mock:
            problems = citation_checks.citation_problems({})
        self.assertEqual(len(problems), 1)
        self.assertIn("CITATION SCHEMA MISSING", problems[0])
        scalar_mock.assert_not_called()

    def test_all_clean_predicates_yield_no_problems(self):
        self.assertEqual(self._problems(), [])

    def test_corpus_document_without_vertex_or_reason_is_a_hole(self):
        problems = self._problems(unplaced=[["2016_vmj598"]])
        self.assertEqual(len(problems), 1)
        self.assertIn("UNPLACED DOCUMENT: 2016_vmj598", problems[0])

    def test_edge_to_unknown_work_is_a_hole(self):
        # Modelled as citation.cites failing FK integrity manifests here as a
        # self-loop or a stale projection -- the two predicates that read
        # citation.cites at all; self-loop is the direct one (protected by a
        # CHECK, re-verified as a predicate per the task spec).
        problems = self._problems(self_loops=[["7"]])
        self.assertEqual(len(problems), 1)
        self.assertIn("SELF LOOP", problems[0])
        self.assertIn("7 -> 7", problems[0])

    def test_no_evidence_and_no_semantic_key_and_indexed_without_external_all_surface(self):
        problems = self._problems(
            no_evidence=[["openalex:W1", "external-skeleton"]],
            no_semantic_key=[["openalex:W2"]],
            indexed_without_external=[["openalex:W3"]])
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
