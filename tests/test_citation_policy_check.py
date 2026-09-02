"""Unit tests for deploy/citation_policy_check.py: the provenance question,
asked of a manifest alone -- no dump, no database, no artifact directory.

Every combination of profile and policy_source is judged, and the two ways
a manifest can say nothing (no citation block, no source in it) are refused
rather than read as "nothing to certify": that early return was how a
manifest with a hole in it certified clean.
"""
from __future__ import annotations

import pathlib
import unittest

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_policy_check
import profile_checks
from manifest_contract import CitationMode, Key, PolicySource, Profile


class PolicySourceTests(unittest.TestCase):
    """Who decided the citation mode, asked of the package rather than of
    its filename. An override build is otherwise indistinguishable from an
    owner-classified one to every consumer: same profile, same schemas,
    same counts, a consistent dump -- and the name is not part of the file.

    The answer is profile-dependent: only public applies a policy, so only
    public may name the owner, and a full artifact must say so rather than
    assert a decision the packager never read.
    """

    def _manifest(self, profile=Profile.PUBLIC, **citation):
        base = {Key.CITATION_MODE: CitationMode.FULL_SKELETON,
                Key.WORK_COUNT: 1, Key.CITES_COUNT: 0}
        base.update(citation)
        return {Key.PROFILE: profile, Key.CITATION: base}

    # (profile, policy_source) -> may this artifact be certified.
    COMBINATIONS = {
        (Profile.PUBLIC, PolicySource.OWNER): True,
        (Profile.PUBLIC, PolicySource.OVERRIDE): False,
        (Profile.PUBLIC, PolicySource.NOT_APPLICABLE): False,
        (Profile.PUBLIC, None): False,
        (Profile.PUBLIC, "somebody-else"): False,
        (Profile.FULL, PolicySource.NOT_APPLICABLE): True,
        (Profile.FULL, PolicySource.OWNER): False,
        (Profile.FULL, PolicySource.OVERRIDE): False,
        (Profile.FULL, None): False,
        (Profile.FULL, "somebody-else"): False,
    }

    def test_every_profile_and_source_pair_is_judged(self):
        for (profile, source), expected in self.COMBINATIONS.items():
            with self.subTest(profile=profile, source=source):
                citation = {} if source is None else {
                    Key.CITATION_POLICY_SOURCE: source}
                ok, detail = citation_policy_check.check_policy_is_the_owners(
                    self._manifest(profile, **citation))
                self.assertEqual(ok, expected, detail)

    def test_a_full_artifact_does_not_claim_the_owner_decided(self):
        """The packager never reads citation.public_policy for full, so
        naming the owner there is a provenance nobody supplied.
        """
        ok, detail = citation_policy_check.check_policy_is_the_owners(
            self._manifest(Profile.FULL,
                           **{Key.CITATION_POLICY_SOURCE: PolicySource.OWNER}))
        self.assertFalse(ok)
        self.assertIn(PolicySource.NOT_APPLICABLE, detail)

    def test_a_public_artifact_may_not_duck_the_question(self):
        ok, detail = citation_policy_check.check_policy_is_the_owners(
            self._manifest(Profile.PUBLIC,
                           **{Key.CITATION_POLICY_SOURCE:
                              PolicySource.NOT_APPLICABLE}))
        self.assertFalse(ok)
        self.assertIn(PolicySource.OWNER, detail)

    def test_an_override_build_is_refused_and_says_why(self):
        ok, detail = citation_policy_check.check_policy_is_the_owners(
            self._manifest(**{Key.CITATION_POLICY_SOURCE: PolicySource.OVERRIDE}))
        self.assertFalse(ok)
        self.assertIn("--policy-override", detail)
        self.assertIn("публиковать нельзя", detail)

    def test_an_override_is_refused_even_when_it_shipped_no_graph(self):
        """`--policy-override none` carries no citation schema at all, so
        nothing leaked -- but the refusal is about whose decision it was.
        """
        ok, detail = citation_policy_check.check_policy_is_the_owners({
            Key.PROFILE: Profile.PUBLIC,
            Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                           Key.CITATION_POLICY_SOURCE: PolicySource.OVERRIDE}})
        self.assertFalse(ok)
        self.assertIn("--policy-override", detail)

    def test_a_shipping_manifest_with_no_source_is_refused_not_defaulted(self):
        """Read with a default, an artifact predating the field would be
        certified as owner-classified -- which is the one outcome the flag
        must never be able to produce.
        """
        ok, detail = citation_policy_check.check_policy_is_the_owners(self._manifest())
        self.assertFalse(ok)
        self.assertIn("None", detail)

    def test_an_artifact_carrying_no_graph_still_names_who_decided(self):
        """mode 'none' is a decision, not an absence: the owner records
        "the graph does not travel" in a schema that exists, and from the
        artifact's side that case is indistinguishable from a build that
        never asked. So it is certified, not excused.
        """
        ok, detail = citation_policy_check.check_policy_is_the_owners({
            Key.PROFILE: Profile.PUBLIC,
            Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE,
                           Key.CITATION_POLICY_SOURCE: PolicySource.OWNER}})
        self.assertTrue(ok, detail)
        ok, detail = citation_policy_check.check_policy_is_the_owners({
            Key.PROFILE: Profile.PUBLIC,
            Key.CITATION: {Key.CITATION_MODE: CitationMode.NONE}})
        self.assertFalse(ok, detail)

    def test_a_manifest_with_no_citation_block_is_not_a_clean_package(self):
        """Every citation assertion used to degrade to a pass here: with no
        block there is no mode, with no mode nothing ships, and with
        nothing shipped there is "nothing to check". A manifest that says
        nothing about the policy is a manifest that cannot be certified.
        """
        for manifest in ({}, {Key.PROFILE: Profile.PUBLIC},
                         {Key.PROFILE: Profile.PUBLIC, Key.CITATION: {}}):
            with self.subTest(manifest=manifest):
                ok, detail = citation_policy_check.check_policy_is_the_owners(manifest)
                self.assertFalse(ok, detail)

    def test_a_mode_this_reader_does_not_know_is_refused(self):
        ok, detail = citation_policy_check.check_policy_is_the_owners({
            Key.PROFILE: Profile.PUBLIC,
            Key.CITATION: {Key.CITATION_MODE: "skeleton-with-notes",
                           Key.CITATION_POLICY_SOURCE: PolicySource.OWNER}})
        self.assertFalse(ok)
        self.assertIn("skeleton-with-notes", detail)

    def test_profile_checks_runs_it(self):
        """The check is only worth anything if the pass that certifies an
        artifact actually calls it -- profile_checks.run_checks() is what
        both the standalone CLI and smoke_test.py go through.
        """
        source = pathlib.Path(profile_checks.__file__).read_text(encoding="utf-8")
        self.assertIn("check_policy_is_the_owners", source)


if __name__ == "__main__":
    unittest.main()
