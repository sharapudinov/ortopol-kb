#!/usr/bin/env python3
"""Which document ids a manifest says this artifact carries, and in which
shape -- the legal vocabulary profile_checks.py's checks reason over.

Split out of profile_checks.py for size (kb/CLAUDE.md FILE_SIZE) along the
one seam that was already there: everything here answers a question about
manifest.legal ALONE, with no dump byte in sight, while every check next
door holds the dump against one of these answers. Nothing here reads a
file, a database or a scan.

The profile is part of the reading, not applied afterwards: FULL carries
the whole corpus whatever the classification says -- it is the owner's own
backup, and the legal cut is the public profile's job alone.
"""
from __future__ import annotations

from manifest_keys import Key
from manifest_contract import Distribution, Profile


def classes(manifest: dict) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """(documents_by_distribution, full_content_distributions,
    shipped_distributions) from the manifest's legal block.

    A manifest missing shipped_distributions yields an empty list, not the
    vocabulary's default: read with a default, an artifact that silently
    dropped a class would look exactly like one that carries it. The empty
    list makes every content check fail, and
    profile_checks.check_classification_complete() says why.
    """
    legal = manifest.get(Key.LEGAL, {})
    return (
        legal.get(Key.DOCUMENTS_BY_DISTRIBUTION, {}),
        legal.get(Key.FULL_CONTENT_DISTRIBUTIONS, []),
        legal.get(Key.SHIPPED_DISTRIBUTIONS, []),
    )


def ids(by_distribution: dict[str, list[str]], names: list[str]) -> set[str]:
    return {doc_id for name in names for doc_id in by_distribution.get(name, [])}


def expected_ids(manifest: dict) -> tuple[set[str], set[str]]:
    """(ids the artifact must contain, ids it must not contain at all)."""
    by_distribution, _full_content, shipped = classes(manifest)
    everything = {doc_id for group in by_distribution.values() for doc_id in group}
    if manifest.get(Key.PROFILE) != Profile.PUBLIC:
        return everything, set()
    shipped_ids = ids(by_distribution, shipped)
    return shipped_ids, everything - shipped_ids


def content_expectation(manifest: dict) -> tuple[set[str], set[str]]:
    """(ids that must carry full content, ids that must ship stripped).

    For the FULL profile nothing is stripped -- the artifact carries
    everything regardless of class, and that is exactly what the checks
    then assert about it.
    """
    by_distribution, full_content, _shipped = classes(manifest)
    present, _absent = expected_ids(manifest)
    if manifest.get(Key.PROFILE) != Profile.PUBLIC:
        return present, set()
    full_ids = ids(by_distribution, full_content) & present
    return full_ids, present - full_ids


def check_profile_is_known(manifest: dict) -> tuple[bool, str]:
    """The declared profile is a value this reader knows.

    Both readings above, and check_policy_is_the_owners() next door, select
    what to demand with `!= Profile.PUBLIC`: a profile field that is
    missing, misspelt, corrupted or hand-edited therefore takes the lenient
    branch of every one of them at once, and the certification prints a
    column of passes about a package nothing was actually verified against.
    manifest.json is not signed, and profile_checks.py travels in the
    artifact precisely so a recipient who did not build it can certify it,
    so this closes before anything branches -- the same polarity the
    version gate already has, and the same one a build applies through
    manifest_contract.schemas_for().
    """
    declared = manifest.get(Key.PROFILE)
    ok = declared in Profile.ALL
    return ok, (f"manifest {Key.PROFILE}={declared!r}, этот проверяльщик знает "
                f"{Profile.ALL}"
                + ("" if ok else " — профиль вне словаря, остальные проверки "
                                 "не запускались"))


# What each list may contain, per profile. shipped_distributions decides
# which documents the certification expects to FIND and which it demands are
# absent, so on the public profile it is held to what the packager is
# allowed to ship at all (Distribution.SHIPPED) rather than merely to the
# vocabulary: `excluded` is a perfectly well-known class, and a manifest
# naming it here empties the `absent` set -- "excluded: not a document row,
# not a page" then certifies [OK] on an artifact shipping excluded
# third-party documents in full. The full profile carries the whole corpus
# whatever the classification says, so its lists describe rather than
# decide, and the vocabulary is all that can be asked of them.
_ALLOWED = {
    Key.SHIPPED_DISTRIBUTIONS: {Profile.PUBLIC: Distribution.SHIPPED,
                                Profile.FULL: Distribution.ALL},
    Key.FULL_CONTENT_DISTRIBUTIONS: {Profile.PUBLIC: Distribution.FULL_CONTENT,
                                     Profile.FULL: Distribution.FULL_CONTENT},
}


def check_legal_vocabulary_is_known(manifest: dict) -> tuple[bool, str]:
    """Both legal lists name only classes this reader knows the profile may
    carry, and neither is empty.

    classes() reads them raw from manifest.json and expected_ids() /
    content_expectation() derive the ENTIRE legal expectation of this
    checker from them. Nothing else validates them: check_classification_
    complete() asks only that the key is present, and run_checks() gates the
    profile, the version and the citation block's shape. So an unknown
    string, an over-broad list or an empty one all shrink the expectation
    silently, and a shrunken expectation is what a column of [OK] against a
    package nobody verified looks like.

    Empty is refused rather than read as a cut for the same reason
    classes() refuses to substitute the vocabulary's default: an artifact
    that silently dropped a class would look exactly like one that carries
    none.
    """
    profile = manifest.get(Key.PROFILE)
    legal = manifest.get(Key.LEGAL, {})
    problems = []
    for key, per_profile in _ALLOWED.items():
        allowed = per_profile[profile]
        declared = legal.get(key, [])
        unknown = sorted(set(declared) - set(allowed))
        if unknown:
            problems.append(f"{key}: {unknown} вне {list(allowed)}")
        elif not declared:
            problems.append(f"{key}: пуст — держать дамп не к чему")
    return not problems, (
        f"профиль {profile!r}: " + ("; ".join(problems) or "оба списка в словаре"))
