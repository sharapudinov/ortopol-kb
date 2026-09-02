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


class ResolveCitationModeTests(unittest.TestCase):
    """The ONE reading of the policy per build (build_package.main hands the
    result to the manifest and to the dump alike).
    """

    def test_no_citation_schema_means_nothing_to_carry(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            for profile in ("full", "public"):
                self.assertEqual(
                    citation_profile.resolve_citation_mode({}, profile), CitationMode.NONE)
        require_mock.assert_not_called()

    def test_an_override_cannot_conjure_a_schema_that_is_absent(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False):
            self.assertEqual(
                citation_profile.resolve_citation_mode({}, "public", CitationMode.FULL_SKELETON),
                CitationMode.NONE,
            )

    def test_full_profile_ships_the_whole_schema_whatever_the_policy_row_says(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            self.assertEqual(citation_profile.resolve_citation_mode({}, "full"),
                             CitationMode.FULL_SKELETON)
        require_mock.assert_not_called()

    def test_public_profile_defers_to_the_owners_row(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode",
                                return_value=CitationMode.TOPOLOGY_ONLY) as require_mock:
            self.assertEqual(citation_profile.resolve_citation_mode({}, "public"),
                             CitationMode.TOPOLOGY_ONLY)
        require_mock.assert_called_once()

    def test_override_bypasses_the_database_read(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            self.assertEqual(
                citation_profile.resolve_citation_mode({}, "public", CitationMode.FULL_SKELETON),
                CitationMode.FULL_SKELETON,
            )
        require_mock.assert_not_called()

    def test_undecided_policy_still_refuses(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode",
                                side_effect=citation_profile.CitationUnclassified("no row")):
            with self.assertRaises(citation_profile.CitationUnclassified):
                citation_profile.resolve_citation_mode({}, "public")


class ShippedRowPredicateTests(unittest.TestCase):
    """The citation slice honours corpus.documents.public_distribution
    through legal_profile's own predicate -- LEGAL_IS_DATA, no id list here.
    """

    def test_work_predicate_lets_a_document_less_row_through(self):
        sql = citation_profile.shipped_work_sql("w")
        self.assertIn("w.document_id IS NULL", sql)
        self.assertIn("public_distribution IN (", sql)

    def test_work_predicate_uses_shipped_not_full_content(self):
        sql = citation_profile.shipped_work_sql()
        self.assertIn("'metadata-only'", sql, "библиография metadata-only уезжает")
        self.assertNotIn("'excluded'", sql)

    def test_crawl_step_predicate_checks_both_vocabularies_in_three_columns(self):
        sql = citation_profile.shipped_crawl_step_sql("s")
        self.assertEqual(sql.count("s.frontier_key"), 2)
        self.assertEqual(sql.count("s.candidate_key"), 2)
        self.assertEqual(sql.count("strpos(coalesce(s.reason, '')"), 2)
        self.assertIn("cut_documents", sql)
        self.assertIn("cut_keys", sql)

    def test_crawl_step_predicate_is_membership_not_a_per_row_derivation(self):
        # Every crawl_step row used to re-scan corpus.documents and the whole
        # of citation.work; the derivation belongs to the statement, once.
        sql = citation_profile.shipped_crawl_step_sql("s")
        self.assertNotIn("corpus.documents", sql)
        self.assertNotIn("citation.work", sql)
        self.assertNotIn("public_distribution", sql)

    def test_the_cut_sets_are_derived_once_in_the_ctes(self):
        ctes = citation_profile.crawl_step_cut_ctes()
        self.assertTrue(ctes.startswith("WITH cut_documents AS ("), ctes[:40])
        self.assertEqual(ctes.count("FROM corpus.documents d"), 1)
        self.assertEqual(ctes.count("cut_keys AS ("), 1)
        self.assertIn("FROM citation.work w", ctes)
        # LEGAL_IS_DATA: still the column, never a list of ids.
        self.assertIn("public_distribution IN (", ctes)


class ShippedOnlyCountsTests(unittest.TestCase):
    def test_shipped_only_counts_apply_the_predicate(self):
        seen = []
        with mock.patch.object(citation_profile, "scalar",
                                side_effect=lambda env, sql: seen.append(sql) or "0"), \
             mock.patch.object(citation_profile, "run_sql", return_value=mock.Mock(stdout="")):
            citation_profile.citation_counts({}, shipped_only=True)
        self.assertTrue(all("public_distribution IN (" in sql for sql in seen), seen)
        self.assertIn("JOIN citation.work wa", seen[1])

    def test_default_counts_the_whole_schema(self):
        seen = []
        with mock.patch.object(citation_profile, "scalar",
                                side_effect=lambda env, sql: seen.append(sql) or "0"), \
             mock.patch.object(citation_profile, "run_sql", return_value=mock.Mock(stdout="")):
            citation_profile.citation_counts({})
        self.assertTrue(all("public_distribution" not in sql for sql in seen), seen)


if __name__ == "__main__":
    unittest.main()
