"""Unit tests for deploy/legal_profile.py: no live database (run_sql/scalar
are stubbed with the exact FIELD_SEP-joined text psql produces).

What matters here is not SQL plumbing but a refusal: an unclassified or
unknown-classification document must STOP a public build. Everything else in
the packaging chain trusts that guarantee, so it is tested directly rather
than through build_package.
"""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import legal_profile
from manifest_contract import Distribution


def _psql_rows(rows: list[tuple[str, ...]]) -> mock.Mock:
    """A CompletedProcess-alike whose stdout is what psql -t -A -F <sep>
    prints for those rows.
    """
    text = "\n".join(legal_profile.FIELD_SEP.join(row) for row in rows)
    return mock.Mock(stdout=text + ("\n" if text else ""))


class PredicateTests(unittest.TestCase):
    def test_full_content_sql_lists_exactly_the_full_content_distributions(self):
        sql = legal_profile.FULL_CONTENT_SQL
        self.assertIn("'full-text'", sql)
        self.assertIn("'internal'", sql)
        self.assertNotIn("'metadata-only'", sql)

    def test_predicate_is_derived_from_the_constants_not_retyped(self):
        # A second hardcoded list would be free to drift from the tuple the
        # rest of the module reasons about.
        for value in Distribution.FULL_CONTENT:
            self.assertIn(f"'{value}'", legal_profile.FULL_CONTENT_SQL)
        for value in Distribution.SHIPPED:
            self.assertIn(f"'{value}'", legal_profile.SHIPPED_SQL)

    def test_shipped_sql_admits_metadata_only_and_refuses_excluded(self):
        # The two predicates answer different questions: SHIPPED_SQL whether
        # a row exists in the public artifact at all, FULL_CONTENT_SQL how
        # much of it that row carries.
        sql = legal_profile.SHIPPED_SQL
        self.assertIn("'metadata-only'", sql)
        self.assertNotIn("'excluded'", sql)

    def test_unclassified_query_rejects_nulls_and_unknown_values(self):
        sql = legal_profile.UNCLASSIFIED_SQL
        self.assertIn("legal_class IS NULL", sql)
        self.assertIn("public_distribution IS NULL", sql)
        self.assertIn("legal_note IS NULL", sql)
        # 'excluded' is a KNOWN value: the owner deciding a document stays
        # out is a classification, not the absence of one, and must not be
        # reported as unclassified.
        self.assertIn(
            "NOT IN ('full-text', 'metadata-only', 'internal', 'excluded')", sql,
        )

    def test_no_document_ids_or_filename_patterns_anywhere_in_the_module(self):
        # The whole point of task 033's revision: classification is DATA.
        # A year cutoff or a LIKE '____\\_isu%' pattern here would put the
        # legal decision back into the packager.
        source = pathlib.Path(legal_profile.__file__).read_text()
        code = source.split('"""', 2)[2]  # skip the module docstring
        for smell in ("LIKE '____", "substring(id", "1992", "_isu", "_demr"):
            self.assertNotIn(smell, code, f"legal knowledge hardcoded in code: {smell}")


class RequireClassifiedTests(unittest.TestCase):
    def test_empty_result_means_every_document_is_classified(self):
        with mock.patch.object(legal_profile, "run_sql", return_value=_psql_rows([])):
            self.assertEqual(legal_profile.unclassified_documents({}), [])
            legal_profile.require_classified({})  # must not raise

    def test_missing_classification_raises_naming_every_offender(self):
        rows = [
            ("2026_newpaper", "", ""),
            ("2026_other", "author-copyright-book", "sort-of-public"),
        ]
        with mock.patch.object(legal_profile, "run_sql", return_value=_psql_rows(rows)):
            with self.assertRaises(legal_profile.Unclassified) as ctx:
                legal_profile.require_classified({})
        message = str(ctx.exception)
        self.assertIn("2026_newpaper", message)
        self.assertIn("2026_other", message)
        self.assertIn("sort-of-public", message)
        self.assertIn("2 document(s)", message)


class GroupingTests(unittest.TestCase):
    def test_documents_by_distribution_groups_and_keeps_every_known_key(self):
        rows = [
            ("excluded", "2016_vmj598"),
            ("full-text", "2009_isu34"),
            ("full-text", "2015_demr1"),
            ("metadata-only", "1997_sm280"),
        ]
        with mock.patch.object(legal_profile, "run_sql", return_value=_psql_rows(rows)):
            grouped = legal_profile.documents_by_distribution({})
        self.assertEqual(grouped["full-text"], ["2009_isu34", "2015_demr1"])
        self.assertEqual(grouped["metadata-only"], ["1997_sm280"])
        # The excluded document is listed too: the artifact must be able to
        # say WHICH works it leaves out, not merely be silent about them.
        self.assertEqual(grouped["excluded"], ["2016_vmj598"])
        # Present but empty, not absent: a consumer iterating the classes
        # should see "no internal documents", not KeyError.
        self.assertEqual(grouped["internal"], [])

    def test_unknown_distribution_still_appears_so_it_cannot_hide(self):
        rows = [("mystery", "2026_x")]
        with mock.patch.object(legal_profile, "run_sql", return_value=_psql_rows(rows)):
            grouped = legal_profile.documents_by_distribution({})
        self.assertEqual(grouped["mystery"], ["2026_x"])

    def test_class_counts_parses_the_distinct_legal_notes(self):
        # json_agg, not a separator-joined string: psql output is read with
        # str.splitlines(), which also breaks on \x1c-\x1e -- a chr(30)
        # separator shredded one row into several (fixed, guarded here).
        rows = [
            ("author-copyright-book", "full-text", "4",
             '["© на обороте титула", "авторское издание"]'),
            ("cc-by-4.0", "full-text", "4", '["CC-BY 4.0 плашка"]'),
        ]
        with mock.patch.object(legal_profile, "run_sql", return_value=_psql_rows(rows)):
            counts = legal_profile.class_counts({})
        self.assertEqual(counts[0]["documents"], 4)
        # sorted() by code point: "©" (U+00A9) precedes Cyrillic.
        self.assertEqual(counts[0]["legal_basis"], ["© на обороте титула", "авторское издание"])
        self.assertEqual(counts[1]["legal_basis"], ["CC-BY 4.0 плашка"])


class LegalSummaryTests(unittest.TestCase):
    GROUPED = {
        "full-text": ["2009_isu34"],
        "metadata-only": ["1997_sm280"],
        "internal": ["INDEX"],
        "excluded": ["2016_vmj598"],
    }

    def _summary(self, grouped=None):
        with mock.patch.object(legal_profile, "scalar", return_value="0"), \
             mock.patch.object(legal_profile, "class_counts", return_value=[]), \
             mock.patch.object(legal_profile, "documents_by_distribution",
                                return_value=dict(self.GROUPED if grouped is None else grouped)):
            return legal_profile.legal_summary({})

    def test_summary_carries_the_verify_query_and_its_answer(self):
        summary = self._summary(grouped={})
        self.assertEqual(summary["verify_query"], legal_profile.UNCLASSIFIED_SQL)
        self.assertEqual(summary["unclassified_documents"], 0)
        self.assertEqual(summary["full_content_distributions"], ["full-text", "internal"])

    def test_summary_says_which_classes_the_artifact_carries(self):
        # documents_by_distribution describes the corpus; this field is the
        # only thing separating the lists a public artifact contains from
        # the ones it deliberately omits.
        summary = self._summary()
        self.assertEqual(
            summary["shipped_distributions"], ["full-text", "metadata-only", "internal"],
        )

    def test_shipped_ids_is_every_listed_id_except_the_excluded_ones(self):
        summary = self._summary()
        self.assertEqual(
            legal_profile.shipped_ids(summary), {"2009_isu34", "1997_sm280", "INDEX"},
        )

    def test_shipped_ids_of_a_summary_without_the_field_is_empty(self):
        # An older summary must not be read as "everything ships": absence
        # of the field is absence of the answer.
        summary = self._summary()
        del summary["shipped_distributions"]
        self.assertEqual(legal_profile.shipped_ids(summary), set())


if __name__ == "__main__":
    unittest.main()
