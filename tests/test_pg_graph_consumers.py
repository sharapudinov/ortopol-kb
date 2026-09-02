"""The four graph consumers as ENTRY POINTS, with no database under them.

test_pg_graph_candidates.py and test_pg_graph_cypher.py pin the pure halves
-- the statements that are built and the ranking applied to rows already
decoded -- and test_pg_graph_consumers_live.py answers the same questions
against the real instance. Between the two lay the part that runs on every
call and was covered by neither: the DECODING of what psql prints, and the
two early exits that make an unanswerable question an empty answer instead
of a traceback.

The decoding is the fragile half by construction. Rows arrive as one string
cut on FIELD_SEP and RECORD_SEP (pg_common), positionally, with a fixed
maxsplit per query -- so a column added to the SELECT and not to the split,
or the two put in a different order, either raises ValueError or silently
files a title under a year. Neither needs a server to catch, and a machine
with no Postgres (or no ollama) used to run none of it.

graph_sql is patched at the module the consumer calls it through, and each
canned stdout is written the way psql -A -F -R prints: no trailing newline
inside a record, RECORD_SEP between them.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

import _pathfix  # noqa: F401
import pg_graph_candidates as pgcand
import pg_graph_cypher as pgc
from pg_common import FIELD_SEP, RECORD_SEP


def _stdout(*records: tuple[str, ...]) -> str:
    """What psql prints for these rows, separators and all."""
    return RECORD_SEP.join(FIELD_SEP.join(fields) for fields in records)


def _answer(stdout: str):
    return mock.Mock(stdout=stdout)


class CandidatesDecodingTests(unittest.TestCase):
    ROWS = _stdout(("W1", "1997", "Discrete Chebyshev", "0.91", "3"),
                   ("W2", "", "Undated", "0.80", "0"))

    def _candidates(self, stdout: str, **kwargs):
        with mock.patch.object(pgcand.pg_graph_common, "graph_sql",
                               return_value=_answer(stdout)) as sql:
            return pgcand.candidates(mock.sentinel.env, **kwargs), sql

    def test_each_field_lands_in_its_own_column(self):
        rows, _sql = self._candidates(self.ROWS)
        self.assertEqual(rows[0], {"key": "W1", "year": 1997,
                                   "title": "Discrete Chebyshev",
                                   "score": 0.91, "links": 3})

    def test_an_empty_year_is_absent_and_an_empty_link_count_is_a_zero(self):
        rows, _sql = self._candidates(self.ROWS)
        self.assertIsNone(rows[1]["year"])
        self.assertEqual(rows[1]["links"], 0)

    def test_no_rows_at_all_is_an_empty_answer(self):
        rows, _sql = self._candidates("")
        self.assertEqual(rows, [])

    def test_a_question_nothing_can_embed_is_an_empty_answer_and_a_word(self):
        said = io.StringIO()
        with mock.patch.object(pgcand.pg_search, "embed_query", return_value=None), \
             mock.patch.object(pgcand.pg_graph_common, "graph_sql") as sql, \
             redirect_stderr(said):
            rows = pgcand.candidates(mock.sentinel.env, query="Чебышёв")
        self.assertEqual(rows, [])
        self.assertFalse(sql.called, "запрос ушёл в базу без вектора")
        self.assertIn("эмбеддинги недоступны", said.getvalue())


class CitersDecodingTests(unittest.TestCase):
    ROWS = _stdout(("W2", "Later work", "2001", "external-skeleton"),
                   ("W3", "Undated work", "", "indexed"))

    def _citers(self, stdout: str, escaped: str = "'W1'"):
        with mock.patch.object(pgc, "scalar", return_value=escaped), \
             mock.patch.object(pgc.pg_graph_common, "graph_sql",
                               return_value=_answer(stdout)) as sql:
            return pgc.citers(mock.sentinel.env, "INDEX"), sql

    def test_each_field_lands_in_its_own_column(self):
        rows, _sql = self._citers(self.ROWS)
        self.assertEqual(rows[0], {"key": "W2", "title": "Later work",
                                   "year": 2001, "kind": "external-skeleton"})

    def test_the_undated_citer_comes_last_and_keeps_its_kind(self):
        rows, _sql = self._citers(self.ROWS)
        self.assertEqual([r["key"] for r in rows], ["W2", "W3"])
        self.assertIsNone(rows[1]["year"])
        self.assertEqual(rows[1]["kind"], "indexed")

    def test_a_document_the_graph_does_not_know_is_asked_no_further(self):
        rows, sql = self._citers(self.ROWS, escaped="")
        self.assertEqual(rows, [])
        self.assertFalse(sql.called, "цитирующие спрошены без ключа")


class HybridDecodingTests(unittest.TestCase):
    SEEDS = _stdout(("W1", "'W1'", "0.93"))
    ROWS = _stdout(("W2", "2001", "Later work", "0.81", "cites", "W1", "Seed"),
                   ("W3", "", "Undated", "0.42", "cited-by", "W1", "Seed"))

    def _hybrid(self, answers: list[str], vec="[0.1]"):
        with mock.patch.object(pgc.pg_search, "embed_query", return_value=vec), \
             mock.patch.object(pgc.pg_graph_common, "graph_sql",
                               side_effect=[_answer(a) for a in answers]) as sql:
            return pgc.hybrid(mock.sentinel.env, "вопрос"), sql

    def test_each_field_lands_in_its_own_column(self):
        rows, _sql = self._hybrid([self.SEEDS, self.ROWS])
        self.assertEqual(rows[0], {"key": "W2", "year": 2001, "title": "Later work",
                                   "score": 0.81, "direction": "cites",
                                   "neighbor_key": "W1", "neighbor_title": "Seed"})
        self.assertIsNone(rows[1]["year"])

    def test_no_seed_near_the_question_is_an_empty_answer(self):
        """An empty IN list is not valid Cypher and an empty VALUES list is
        not valid SQL, so the second statement is never issued at all.
        """
        rows, sql = self._hybrid([""])
        self.assertEqual(rows, [])
        self.assertEqual(sql.call_count, 1)

    def test_a_question_nothing_can_embed_is_an_empty_answer_and_a_word(self):
        said = io.StringIO()
        with mock.patch.object(pgc.pg_search, "embed_query", return_value=None), \
             mock.patch.object(pgc.pg_graph_common, "graph_sql") as sql, \
             redirect_stderr(said):
            rows = pgc.hybrid(mock.sentinel.env, "вопрос")
        self.assertEqual(rows, [])
        self.assertFalse(sql.called, "запрос ушёл в базу без вектора")
        self.assertIn("hybrid недоступен", said.getvalue())


if __name__ == "__main__":
    unittest.main()
