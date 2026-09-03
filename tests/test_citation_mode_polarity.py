"""One predicate answers "does this mode ship the citation schema".

manifest_contract.ships_citation() is an ALLOWLIST over CitationMode.SHIPPED,
and SHIPPED is hand-written while CitationMode.ALL is inherited from the
column's own vocabulary and grows with it. A site spelling the question as
`!= CitationMode.NONE` is therefore a denylist beside an allowlist: they
agree only while SHIPPED happens to be ALL minus NONE, and the day a mode is
added to citation_vocab.PublicPolicyMode the dump writes no citation byte
while the manifest stamps the live work/cites counts into the block
describing it -- MANIFEST_DESCRIBES_ARTIFACT broken by the packager, and the
recipient failing certification on a build reported as successful.

So the scan below refuses the denylist spelling anywhere a build or a
verifier decides by it, and the positive control proves the scan can see
one.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_cut_checks
import citation_dump
import manifest_citation
import smoke_checks
from citation_vocab import PublicPolicyMode
from manifest_contract import CitationMode, schemas_for, ships_citation
from paths import kb_root

DEPLOY_DIR = kb_root() / "deploy"
# The classes whose .NONE a decision must never be taken against. Both
# spellings, because CitationMode inherits the value and either name reaches
# it.
MODE_CLASSES = ("CitationMode", "PublicPolicyMode")
# The modules that ask the question at all: the bytes, their description and
# the two verifiers. Named so that a site dropping the call is a failure
# here rather than a silently unguarded module.
ASKING_MODULES = (
    "manifest_contract.py", "citation_dump.py", "manifest_citation.py",
    "citation_cut_checks.py", "smoke_checks.py",
)


def _is_mode_none(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "NONE"
            and isinstance(node.value, ast.Name) and node.value.id in MODE_CLASSES)


def denylist_comparisons(source: str) -> list[int]:
    """Line numbers of every comparison against a mode class's NONE.

    Prose naming the value, and an f-string quoting it into a refusal
    message, are not decisions -- only a Compare is.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        if _is_mode_none(node.left) or any(_is_mode_none(c) for c in node.comparators):
            found.append(node.lineno)
    return found


class ShipsCitationIsTheAllowlistTests(unittest.TestCase):
    def test_every_declared_mode_answers_by_membership_in_shipped(self):
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode):
                self.assertIs(ships_citation(mode), mode in CitationMode.SHIPPED)

    def test_the_column_vocabulary_and_the_packagers_agree_on_what_exists(self):
        self.assertEqual(tuple(CitationMode.ALL), tuple(PublicPolicyMode.ALL))

    def test_a_mode_nobody_declared_ships_nothing(self):
        """The half of "unheard-of" that is safe: an artifact carrying no
        citation byte and declaring none.
        """
        for mode in (None, "", "graph-only", "full-skeleton-v2"):
            with self.subTest(mode=mode):
                self.assertFalse(ships_citation(mode))

    def test_the_schema_list_is_built_on_the_same_predicate(self):
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode):
                self.assertEqual("citation" in schemas_for("public", mode),
                                 ships_citation(mode))


class NoSiteDecidesByNoneTests(unittest.TestCase):
    def test_no_deploy_module_compares_a_mode_against_none(self):
        for path in sorted(DEPLOY_DIR.glob("*.py")):
            lines = denylist_comparisons(path.read_text(encoding="utf-8"))
            with self.subTest(module=path.name):
                self.assertEqual(lines, [], f"{path.name}:{lines}: спросите "
                                            "manifest_contract.ships_citation(mode)")

    def test_the_scan_catches_the_spelling_it_is_for(self):
        """Positive control: a passing scan and a blind one look the same."""
        self.assertEqual(
            denylist_comparisons("def f(m):\n    return m != CitationMode.NONE\n"), [2])
        self.assertEqual(
            denylist_comparisons("def f(m):\n    if m == PublicPolicyMode.NONE:\n"
                                 "        return 1\n    return 0\n"), [2])

    def test_prose_and_a_refusal_message_are_not_a_decision(self):
        self.assertEqual(denylist_comparisons(
            '"""Mode CitationMode.NONE carries nothing."""\n'
            'def f():\n    return f"mode {CitationMode.NONE!r} recorded"\n'), [])

    def test_every_asking_module_reads_the_one_predicate(self):
        for name in ASKING_MODULES:
            source = (DEPLOY_DIR / name).read_text(encoding="utf-8")
            called = {node.func.id for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            with self.subTest(module=name):
                self.assertIn("ships_citation", called)

    def test_the_four_verifier_sites_answer_through_it(self):
        """Not "the name appears": each of the four reads it for the mode
        the manifest declares, so a block naming an unheard-of mode is
        treated as shipping nothing rather than as shipping unchecked.
        """
        self.assertFalse(citation_cut_checks._ships_citation(
            {"citation": {"mode": "graph-only"}}))
        ok, _detail = citation_cut_checks.check_citation_schema_matches_mode(
            {"citation": {"mode": "graph-only"}}, {"citation.work": object()})
        self.assertFalse(ok)
        verdict, _detail = smoke_checks.check_citation_projection(
            {}, {"citation": {"mode": "graph-only"}})
        self.assertIsNone(verdict)

    def test_the_manifest_block_of_an_unheard_of_mode_carries_no_counts(self):
        block = manifest_citation.citation_block({}, "graph-only", True, "owner")
        self.assertEqual((block["work_count"], block["cites_count"], block["work_by_kind"]),
                         (0, 0, {}))

    def test_the_dump_of_an_unheard_of_mode_writes_nothing(self):
        class _Sink:
            def __init__(self):
                self.written = []

            def write(self, chunk):
                self.written.append(chunk)

        sink = _Sink()
        citation_dump.dump_citation(
            {}, sink, citation_dump.plan_citation({}, "graph-only"))
        self.assertEqual(sink.written, [])


if __name__ == "__main__":
    unittest.main()
