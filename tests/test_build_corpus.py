"""Full-corpus integration tests: coverage counts and the git-layout guarantee.

Deliberately runs the real pipeline against theory/iis/ into the real
default output directory (a sibling of theory/, outside git) rather than a
throwaway temp dir — this IS the "full run" the acceptance checks refer to.
A discrepancy against the measured ground truth below is a signal to
investigate (corpus changed, or the extractor broke), never to adjust.
"""
import json
import subprocess
import unittest

import _pathfix  # noqa: F401
from build_corpus import build_corpus, write_manifest
from paths import default_corpus_dir, default_pdf_dir, kb_root
from report import write_coverage_report

EXPECTED_COUNTS = {
    # clean went 40 -> 41 on 2026-07-26: 2012_vestnikdnc45.pdf (Vestnik DNC RAN
    # 2012 no.45, the VP x p(x) x Sobolev intersection) was acquired and added
    # to theory/iis/ deliberately; it carries a clean text layer.
    "clean": 41,
    "mojibake_recovered": 21,
    "degraded": 2,
    "broken": 4,
    "no_text_layer": 0,
}
EXPECTED_BROKEN = {
    "1996_sm105",
    "1997_sm280",
    "2000_sm480",
    "2003_sm723",
}


class FullCorpusRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = default_corpus_dir()
        cls.extractions = build_corpus(default_pdf_dir(), cls.output_dir)
        write_manifest(cls.extractions, cls.output_dir)
        cls.report_paths = write_coverage_report(cls.extractions, cls.output_dir)

    def test_matches_measured_ground_truth(self):
        counts = {key: 0 for key in EXPECTED_COUNTS}
        for extraction in self.extractions:
            counts[extraction.category.value] += 1
        self.assertEqual(counts, EXPECTED_COUNTS)

    def test_reports_unreadable_files_explicitly(self):
        report = json.loads(self.report_paths[0].read_text(encoding="utf-8"))
        not_covered_ids = {entry["id"] for entry in report["not_covered"]}
        self.assertEqual(not_covered_ids, EXPECTED_BROKEN)

        markdown = self.report_paths[1].read_text(encoding="utf-8")
        for stem in EXPECTED_BROKEN:
            self.assertIn(stem, markdown)

    def test_index_is_outside_git(self):
        # kb_root(), i.e. THIS repository -- the one thing derived corpus
        # content must never land in. It used to be data_root()/"lib", the
        # ortopol repository this code was extracted from: after the split
        # that directory is neither the git repo these scripts live in nor
        # guaranteed to exist at all, so the guarantee under test would have
        # been checked against the wrong tree (or crashed).
        code_root = kb_root()
        result = subprocess.run(
            ["git", "status", "--short"], cwd=code_root, capture_output=True, text=True, check=True
        )
        # The claim under test is that DERIVED corpus content never lands in git —
        # extracted text, coverage reports, manifests, the index. It is deliberately
        # not "nothing else in the repository changed": other tasks legitimately edit
        # source and docs, and asserting otherwise would turn this into a test that
        # fails for reasons unrelated to what it is checking.
        derived = ("text/", "coverage.", "manifest.", ".db", ".pgenv")
        offenders = [
            path for line in result.stdout.splitlines()
            if (path := line[3:])
            and (path.startswith("corpus/") or any(m in path for m in derived))
        ]
        self.assertEqual(offenders, [], f"derived corpus content inside {code_root}: {offenders}")
        self.assertFalse(str(self.output_dir).startswith(str(code_root)))


if __name__ == "__main__":
    unittest.main()
