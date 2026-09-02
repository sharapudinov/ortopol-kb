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
from manifest_contract import CitationMode, PolicySource

class SchemaAndPolicyReadTests(unittest.TestCase):
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
        counts = {"external-skeleton": 382, "our-document": 56}
        with mock.patch.object(citation_profile, "scalar", side_effect=["2425"]), \
             mock.patch.object(citation_profile, "kind_counts", return_value=counts):
            work_n, cites_n, by_kind = citation_profile.citation_counts({})
        self.assertEqual(work_n, 438)
        self.assertEqual(cites_n, 2425)
        self.assertEqual(by_kind, counts)

    def test_empty_kind_breakdown_is_an_empty_dict(self):
        with mock.patch.object(citation_profile, "scalar", side_effect=["0"]), \
             mock.patch.object(citation_profile, "kind_counts", return_value={}):
            work_n, _cites_n, by_kind = citation_profile.citation_counts({})
        self.assertEqual(by_kind, {})
        self.assertEqual(work_n, 0)

    def test_the_work_total_costs_no_scan_of_its_own(self):
        """kind is NOT NULL, so the census already IS the work count -- and
        under shipped_only the duplicated predicate is the per-row EXISTS
        against corpus.documents, evaluated twice in two psql processes.
        """
        seen = []
        with mock.patch.object(citation_profile, "scalar",
                                side_effect=lambda env, sql: seen.append(sql) or "7"), \
             mock.patch.object(citation_profile, "kind_counts",
                                return_value={"our-document": 3}) as census:
            work_n, cites_n, _by_kind = citation_profile.citation_counts(
                {}, shipped_only=True)
        self.assertEqual((work_n, cites_n), (3, 7))
        self.assertEqual(len(seen), 1, seen)
        self.assertIn("citation.cites", seen[0])
        self.assertNotIn("count(*) FROM citation.work", seen[0])
        census.assert_called_once()


class ResolveCitationModeTests(unittest.TestCase):
    """The ONE reading of the policy per build (build_package.main hands the
    result to the manifest and to the dump alike).
    """

    def test_no_citation_schema_leaves_a_full_build_nothing_to_carry(self):
        """full describes the database as it is: no schema means no citation
        block, and no policy applies to full in the first place."""
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            self.assertEqual(
                citation_profile.resolve_citation_mode({}, "full"),
                (CitationMode.NONE, PolicySource.NOT_APPLICABLE))
        require_mock.assert_not_called()

    def test_a_public_build_against_a_schemaless_database_refuses(self):
        """Shipping no citation graph at all is a decision, and the packager
        does not get to make it by omission."""
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False):
            with self.assertRaises(citation_profile.CitationUnclassified) as ctx:
                citation_profile.resolve_citation_mode({}, "public")
        self.assertIn("citation schema not found", str(ctx.exception))

    def test_an_override_cannot_conjure_a_schema_that_is_absent(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=False):
            with self.assertRaises(citation_profile.CitationUnclassified):
                citation_profile.resolve_citation_mode(
                    {}, "public", CitationMode.FULL_SKELETON)

    def test_full_profile_ships_the_whole_schema_whatever_the_policy_row_says(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            self.assertEqual(citation_profile.resolve_citation_mode({}, "full"),
                             (CitationMode.FULL_SKELETON, PolicySource.NOT_APPLICABLE))
        require_mock.assert_not_called()

    def test_public_profile_defers_to_the_owners_row(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode",
                                return_value=CitationMode.TOPOLOGY_ONLY) as require_mock:
            self.assertEqual(citation_profile.resolve_citation_mode({}, "public"),
                             (CitationMode.TOPOLOGY_ONLY, PolicySource.OWNER))
        require_mock.assert_called_once()

    def test_override_bypasses_the_database_read(self):
        with mock.patch.object(citation_profile, "citation_schema_exists", return_value=True), \
             mock.patch.object(citation_profile, "require_citation_mode") as require_mock:
            self.assertEqual(
                citation_profile.resolve_citation_mode({}, "public", CitationMode.FULL_SKELETON),
                (CitationMode.FULL_SKELETON, PolicySource.OVERRIDE),
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

    def test_crawl_step_ctes_check_both_vocabularies_in_three_columns(self):
        ctes = citation_profile.crawl_step_cut_ctes()
        self.assertEqual(ctes.count("j.frontier_key = r.ref"), 1)
        self.assertEqual(ctes.count("j.candidate_key = r.ref"), 1)
        self.assertEqual(ctes.count("j.node_key = r.ref"), 1)
        # Both vocabularies reach all three columns through one union.
        self.assertIn("SELECT ref FROM cut_documents UNION SELECT ref FROM cut_keys", ctes)

    def test_the_journal_cut_matches_names_in_columns_only(self):
        """The node key is a column of crawl_step, not a fragment of its
        prose: a substring branch over `reason` served no index and could
        match a name inside a sentence that named nothing sensitive.
        """
        ctes = citation_profile.crawl_step_cut_ctes()
        self.assertNotIn("strpos(", ctes)
        self.assertNotIn("j.reason", ctes)

    def test_every_branch_is_a_separate_indexable_equality(self):
        """An OR of three equalities is ONE non-sargable join qualifier:
        none of them could then reach crawl_step_frontier_key_idx /
        crawl_step_candidate_key_idx / crawl_step_node_key_idx. Split into
        UNION branches, each is an ordinary indexable join.
        """
        ctes = citation_profile.crawl_step_cut_ctes()
        branches = ctes[ctes.index("cut_steps AS MATERIALIZED"):].split("UNION")
        self.assertEqual(len(branches), 3, ctes)
        for branch in branches:
            self.assertLessEqual(branch.count("ON "), 1, "two match tests in one branch")
        self.assertNotIn(" OR ", ctes, "an OR-ed qualifier reaches no index")

    def test_crawl_step_predicate_is_membership_not_a_per_row_derivation(self):
        # Every crawl_step row used to re-scan corpus.documents and the whole
        # of citation.work; the derivation belongs to the statement, once.
        sql = citation_profile.shipped_crawl_step_sql("s")
        self.assertEqual(sql, "(NOT EXISTS (SELECT 1 FROM cut_steps x WHERE x.id = s.id))")
        self.assertNotIn("corpus.documents", sql)
        self.assertNotIn("citation.work", sql)
        self.assertNotIn("public_distribution", sql)

    def test_the_cut_sets_are_derived_once_in_the_ctes(self):
        ctes = citation_profile.crawl_step_cut_ctes()
        self.assertTrue(ctes.startswith("WITH cut_documents AS MATERIALIZED ("), ctes[:60])
        self.assertEqual(ctes.count("FROM corpus.documents d"), 1)
        self.assertEqual(ctes.count("cut_keys AS MATERIALIZED ("), 1)
        self.assertIn("FROM citation.work w", ctes)
        # LEGAL_IS_DATA: still the column, never a list of ids.
        self.assertIn("public_distribution IN (", ctes)

    def test_every_cut_set_is_materialised(self):
        """A single-reference CTE is inlined by default on PostgreSQL 12+,
        which would put each derivation back inside the subquery it feeds --
        "derived once per statement" would then be the planner's choice
        rather than the statement's.
        """
        ctes = citation_profile.crawl_step_cut_ctes()
        for name in ("cut_documents", "cut_keys", "cut_names", "cut_steps"):
            self.assertIn(f"{name} AS MATERIALIZED (", ctes)


class ShippedOnlyCountsTests(unittest.TestCase):
    def _counted(self, **kwargs) -> tuple[list[str], list[str]]:
        """(scalar SQL seen, narrowing clauses handed to the shared census)."""
        seen, narrowed = [], []
        with mock.patch.object(citation_profile, "scalar",
                                side_effect=lambda env, sql: seen.append(sql) or "0"), \
             mock.patch.object(citation_profile, "kind_counts",
                                side_effect=lambda env, where="": narrowed.append(where) or {}):
            citation_profile.citation_counts({}, **kwargs)
        return seen, narrowed

    def test_shipped_only_counts_apply_the_predicate(self):
        seen, narrowed = self._counted(shipped_only=True)
        self.assertTrue(all("public_distribution IN (" in sql for sql in seen), seen)
        self.assertIn("JOIN citation.work wa", seen[0])
        self.assertTrue(all("public_distribution IN (" in where for where in narrowed), narrowed)

    def test_default_counts_the_whole_schema(self):
        seen, narrowed = self._counted()
        self.assertTrue(all("public_distribution" not in sql for sql in seen), seen)
        self.assertEqual(narrowed, [""])


if __name__ == "__main__":
    unittest.main()
