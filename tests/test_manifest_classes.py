"""deploy/manifest_classes.py: the manifest-only legal reading, and the two
gates that have to pass before anything reads it.

manifest.json is not signed and profile_checks.py travels INSIDE the
artifact, so every field the checks branch on is a field a recipient must
be able to see refused rather than obeyed (ARTIFACT_SIDE_FAILS_CLOSED).
The profile is one such field and has had a gate since it acquired its
first `!= Profile.PUBLIC`; shipped_distributions and
full_content_distributions are the other two, and they decide WHICH
documents the certification then looks for.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import profile_checks
from _artifact_fixtures import ArtifactBuilder
from manifest_classes import (check_legal_vocabulary_is_known, check_profile_is_known,
                              content_expectation, cut_applies, expected_ids)
from manifest_contract import Distribution, Profile
from manifest_keys import Key

GATE = "правовой словарь манифеста известен"


def _manifest(profile=Profile.PUBLIC, shipped=None, full_content=None) -> dict:
    return {
        Key.PROFILE: profile,
        Key.LEGAL: {
            Key.DOCUMENTS_BY_DISTRIBUTION: {},
            Key.SHIPPED_DISTRIBUTIONS: (list(Distribution.SHIPPED) if shipped is None
                                        else shipped),
            Key.FULL_CONTENT_DISTRIBUTIONS: (list(Distribution.FULL_CONTENT)
                                             if full_content is None else full_content),
        },
    }


class LegalVocabularyGateTests(unittest.TestCase):
    def test_a_build_of_either_profile_passes(self):
        for profile in Profile.ALL:
            with self.subTest(profile=profile):
                ok, detail = check_legal_vocabulary_is_known(_manifest(profile))
                self.assertTrue(ok, detail)

    def test_a_distribution_nobody_has_heard_of_is_refused(self):
        """Read leniently, an unknown name simply names no document: the
        expectation quietly shrinks and every check about it passes.
        """
        ok, detail = check_legal_vocabulary_is_known(
            _manifest(shipped=[*Distribution.SHIPPED, "cc-by-probably"]))
        self.assertFalse(ok)
        self.assertIn("cc-by-probably", detail)

    def test_a_public_manifest_claiming_to_ship_excluded_is_refused(self):
        """The whole reason this gate exists. `excluded` inside
        shipped_distributions empties the `absent` set, so "excluded: not a
        document row, not a page" certifies [OK] on an artifact that ships
        excluded third-party documents in full -- the single outcome
        FULL_NEVER_PUBLISHED exists to prevent. The class is in
        Distribution.ALL, so a vocabulary check alone would let it through:
        what the public profile may ship is Distribution.SHIPPED.
        """
        ok, detail = check_legal_vocabulary_is_known(
            _manifest(shipped=[*Distribution.SHIPPED, Distribution.EXCLUDED]))
        self.assertFalse(ok)
        self.assertIn(Distribution.EXCLUDED, detail)

    def test_a_full_manifest_is_held_to_the_vocabulary_only(self):
        """The full profile carries the whole corpus whatever the
        classification says, so its lists describe rather than decide --
        but a name outside the vocabulary is still a manifest nobody can
        read.
        """
        ok, _detail = check_legal_vocabulary_is_known(
            _manifest(Profile.FULL, shipped=[*Distribution.SHIPPED, Distribution.EXCLUDED]))
        self.assertTrue(ok)
        ok, _detail = check_legal_vocabulary_is_known(
            _manifest(Profile.FULL, shipped=["invented"]))
        self.assertFalse(ok)

    def test_content_may_only_be_claimed_full_for_a_class_that_carries_it(self):
        ok, detail = check_legal_vocabulary_is_known(
            _manifest(full_content=[*Distribution.FULL_CONTENT, Distribution.METADATA_ONLY]))
        self.assertFalse(ok)
        self.assertIn(Distribution.METADATA_ONLY, detail)

    def test_an_empty_list_is_refused_rather_than_read_as_a_cut(self):
        """An empty full_content_distributions makes the metadata-only check
        vacuous the same way an over-broad one does, and an empty
        shipped_distributions says the public artifact carries nothing at
        all -- neither is a package this reader can hold to anything.
        """
        for field, manifest in ((Key.SHIPPED_DISTRIBUTIONS, _manifest(shipped=[])),
                                (Key.FULL_CONTENT_DISTRIBUTIONS, _manifest(full_content=[]))):
            with self.subTest(field=field):
                ok, detail = check_legal_vocabulary_is_known(manifest)
                self.assertFalse(ok)
                self.assertIn(field, detail)

    def test_a_missing_legal_block_is_refused_too(self):
        ok, _detail = check_legal_vocabulary_is_known({Key.PROFILE: Profile.PUBLIC})
        self.assertFalse(ok)


class TheGateStopsThePassTests(unittest.TestCase):
    """A column of passes underneath a failed gate reads as a certification,
    so the failing gate is the whole answer -- the polarity the version and
    profile gates already have.
    """

    def _results(self, builder: ArtifactBuilder) -> list[tuple[str, bool, str]]:
        return profile_checks.run_checks(builder.write())

    def test_a_well_formed_artifact_passes_the_gate_and_goes_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self._results(ArtifactBuilder(Path(tmp)))
        names = [name for name, _ok, _detail in results]
        self.assertIn(GATE, names)
        self.assertGreater(len(names), names.index(GATE) + 1)

    def test_a_shipped_class_the_packager_may_never_ship_stops_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = ArtifactBuilder(Path(tmp))
            builder.shipped = [*builder.shipped, Distribution.EXCLUDED]
            results = self._results(builder)
        self.assertEqual([name for name, _ok, _detail in results][-1], GATE)
        self.assertFalse(results[-1][1])
        self.assertTrue(all(ok for _name, ok, _detail in results[:-1]))

    def test_an_unknown_profile_is_still_refused_first(self):
        ok, _detail = check_profile_is_known({Key.PROFILE: "publicish"})
        self.assertFalse(ok)


class SetArithmeticTests(unittest.TestCase):
    """expected_ids/cut_applies/content_expectation on synthetic manifests.

    The whole legal certification rests on these three: which ids must be
    in the dump, which must be nowhere in it, which must carry content and
    which must not. Reached only through artifact fixtures they were tested
    at the granularity of a green column -- a boundary answered wrongly
    here (an id claimed by two classes, a class named full-content but not
    shipped) is a shrunken expectation, and a shrunken expectation is what
    an [OK] about nothing looks like.
    """

    def _legal(self, by_distribution, shipped, full_content, profile=Profile.PUBLIC):
        return {
            Key.PROFILE: profile,
            Key.LEGAL: {
                Key.DOCUMENTS_BY_DISTRIBUTION: by_distribution,
                Key.SHIPPED_DISTRIBUTIONS: shipped,
                Key.FULL_CONTENT_DISTRIBUTIONS: full_content,
            },
        }

    BY_DISTRIBUTION = {
        Distribution.FULL_TEXT: ["a"],
        Distribution.METADATA_ONLY: ["b"],
        Distribution.EXCLUDED: ["c"],
    }

    def _public(self, by_distribution=None):
        return self._legal(by_distribution or self.BY_DISTRIBUTION,
                           list(Distribution.SHIPPED), list(Distribution.FULL_CONTENT))

    def test_public_expects_the_shipped_classes_and_forbids_the_rest(self):
        self.assertEqual(expected_ids(self._public()), ({"a", "b"}, {"c"}))

    def test_full_expects_every_id_and_forbids_none(self):
        manifest = self._public()
        manifest[Key.PROFILE] = Profile.FULL
        self.assertEqual(expected_ids(manifest), ({"a", "b", "c"}, set()))

    def test_an_id_listed_under_two_classes_is_one_id(self):
        """`everything` is a set: a document named by both a shipped and an
        excluded class is expected AND absent by list arithmetic, and the
        absent set is what "excluded left no trace" is checked against.
        """
        both = {Distribution.FULL_TEXT: ["a"], Distribution.EXCLUDED: ["a"]}
        expected, absent = expected_ids(self._public(both))
        self.assertEqual(expected, {"a"})
        self.assertEqual(absent, set())

    def test_cut_applies_only_where_something_is_actually_cut(self):
        self.assertTrue(cut_applies(self._public()))
        nothing_cut = {Distribution.FULL_TEXT: ["a"], Distribution.METADATA_ONLY: ["b"]}
        self.assertFalse(cut_applies(self._public(nothing_cut)))

    def test_cut_never_applies_to_the_full_profile(self):
        manifest = self._public()
        manifest[Key.PROFILE] = Profile.FULL
        self.assertFalse(cut_applies(manifest))

    def test_public_content_expectation_splits_the_shipped_ids(self):
        self.assertEqual(content_expectation(self._public()), ({"a"}, {"b"}))

    def test_a_full_content_class_that_does_not_ship_expects_nothing(self):
        """full_content is intersected with what is present, so a class
        named full-content but absent from shipped_distributions cannot
        demand content for a document the artifact does not carry at all.
        """
        manifest = self._legal(self.BY_DISTRIBUTION, [Distribution.METADATA_ONLY],
                               [Distribution.FULL_TEXT])
        self.assertEqual(content_expectation(manifest), (set(), {"b"}))

    def test_the_full_profile_strips_nothing(self):
        manifest = self._public()
        manifest[Key.PROFILE] = Profile.FULL
        self.assertEqual(content_expectation(manifest), ({"a", "b", "c"}, set()))


if __name__ == "__main__":
    unittest.main()
