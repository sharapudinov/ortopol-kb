"""Unit tests for deploy/citation_profile.py: no live database (scalar/
run_sql are stubbed). What matters here is the same refusal shape
legal_profile.require_classified has: a missing schema, or a missing/
unrecognised public_policy.mode, must STOP a public build.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_profile
from manifest_contract import CitationMode

FIELD_SEP = "\x1f"


def _psql_rows(rows: list[tuple[str, ...]]) -> mock.Mock:
    text = "\n".join(FIELD_SEP.join(row) for row in rows)
    return mock.Mock(stdout=text + ("\n" if text else ""))


class SchemaAndPolicyReadTests(unittest.TestCase):
    def test_schema_exists_true_false(self):
        with mock.patch.object(citation_profile, "scalar", return_value="t"):
            self.assertTrue(citation_profile.citation_schema_exists({}))
        with mock.patch.object(citation_profile, "scalar", return_value="f"):
            self.assertFalse(citation_profile.citation_schema_exists({}))

    def test_public_policy_returns_none_when_row_is_absent(self):
        with mock.patch.object(citation_profile, "scalar", return_value=""):
            self.assertIsNone(citation_profile.citation_public_policy({}))

    def test_public_policy_returns_the_recorded_mode(self):
        with mock.patch.object(citation_profile, "scalar", return_value="topology-only"):
            self.assertEqual(citation_profile.citation_public_policy({}), "topology-only")


class RequireCitationModeTests(unittest.TestCase):
    def test_missing_schema_raises(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False):
            with self.assertRaises(citation_profile.CitationUnclassified) as ctx:
                citation_profile.require_citation_mode({})
        self.assertIn("citation schema not found", str(ctx.exception))

    def test_missing_policy_row_raises_naming_the_valid_modes(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "citation_public_policy", return_value=None):
            with self.assertRaises(citation_profile.CitationUnclassified) as ctx:
                citation_profile.require_citation_mode({})
        message = str(ctx.exception)
        self.assertIn("full-skeleton", message)
        self.assertIn("topology-only", message)
        self.assertIn("none", message)

    def test_unrecognised_mode_raises(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "citation_public_policy", return_value="sort-of-public"):
            with self.assertRaises(citation_profile.CitationUnclassified) as ctx:
                citation_profile.require_citation_mode({})
        self.assertIn("sort-of-public", str(ctx.exception))

    def test_each_known_mode_is_returned_as_is(self):
        for mode in CitationMode.ALL:
            with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
                 mock.patch.object(citation_profile, "citation_public_policy", return_value=mode):
                self.assertEqual(citation_profile.require_citation_mode({}), mode)


class CitationCountsTests(unittest.TestCase):
    def test_reads_work_cites_and_kind_breakdown(self):
        rows = [("external-skeleton", "382"), ("our-document", "56")]
        with mock.patch.object(citation_profile, "scalar", side_effect=["438", "2425"]), \
             mock.patch.object(citation_profile, "run_sql", return_value=_psql_rows(rows)):
            work_n, cites_n, by_kind = citation_profile.citation_counts({})
        self.assertEqual(work_n, 438)
        self.assertEqual(cites_n, 2425)
        self.assertEqual(by_kind, {"external-skeleton": 382, "our-document": 56})

    def test_empty_kind_breakdown_is_an_empty_dict(self):
        with mock.patch.object(citation_profile, "scalar", side_effect=["0", "0"]), \
             mock.patch.object(citation_profile, "run_sql", return_value=mock.Mock(stdout="")):
            _work_n, _cites_n, by_kind = citation_profile.citation_counts({})
        self.assertEqual(by_kind, {})


if __name__ == "__main__":
    unittest.main()
