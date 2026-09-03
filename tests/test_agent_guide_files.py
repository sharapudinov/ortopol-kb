"""The recipient's guide names no file the package does not carry.

AGENT_GUIDE.md used to enumerate the bundled modules in prose beside
artifact_bundle.DEPLOY_FILES / CORPUS_LIB_FILES, and the two drifted: the
guide listed fifteen of the thirty-five deploy files and said nothing about
probe_query.py, the column classification maps, the citation checks or
corpus_cut.py. Two lists of one fact is the defect; a test comparing them
in both directions would only have frozen the duplication in place. So the
enumeration is gone -- manifest.json's `files` is the package's own list,
and bundled_files_check holds the extracted directory to it -- and what is
left to guard is the direction that still matters: every script the guide
tells the recipient to RUN has to be in the package.

The exception list is not written here either. The guide's own first
section declares which parent-repository files it names as absent, and this
reads that sentence: a name may be missing from the bundle only where the
document itself says it is missing.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import artifact_bundle

GUIDE = artifact_bundle.DEPLOY_DIR / "AGENT_GUIDE.md"
# Every `...py` the guide spells in code formatting, and the parenthesis the
# guide's opening paragraph puts the parent-repository names in.
MENTION = re.compile(r"`([A-Za-z0-9_/.-]+\.py)`")
DECLARED_ABSENT = re.compile(r"Файлов родительского репозитория \(([^)]*)\)")


def guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8")


class GuideNamesOnlyBundledScriptsTests(unittest.TestCase):
    def _bundled(self) -> set[str]:
        return set(artifact_bundle.DEPLOY_FILES) | {
            f"corpus_lib/{name}" for name in artifact_bundle.CORPUS_LIB_FILES}

    def _declared_absent(self) -> set[str]:
        text = guide_text()
        found = DECLARED_ABSENT.search(text)
        self.assertIsNotNone(
            found, "гайд больше не объявляет, каких файлов репозитория здесь нет")
        return set(MENTION.findall(found.group(1)))

    def test_every_script_the_guide_names_travels_in_the_package(self):
        unaccounted = sorted(
            set(MENTION.findall(guide_text())) - self._bundled() - self._declared_absent())
        self.assertEqual(unaccounted, [], "гайд получателя называет файлы, которых "
                                          "в пакете нет")

    def test_the_scan_would_see_a_name_that_is_not_bundled(self):
        """The positive control: without it a regex that matched nothing
        would report a clean guide forever."""
        self.assertIn("corpus_completeness.py", MENTION.findall(guide_text()))
        self.assertNotIn("corpus_completeness.py", self._bundled())

    def test_the_guide_does_not_re_enumerate_the_bundle(self):
        """One list of one fact. The package's own is manifest.json's
        `files`, which the guide points at and bundled_files_check verifies;
        a second list in prose is the one that goes stale silently.
        """
        named = set(MENTION.findall(guide_text())) & self._bundled()
        self.assertLess(len(named), 10, sorted(named))
        self.assertIn("manifest.json", guide_text())


if __name__ == "__main__":
    unittest.main()
