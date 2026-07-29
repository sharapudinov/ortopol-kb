"""Unit tests for the external-literature class: registry, loader, boundary.

Three questions, in order of how expensive getting them wrong is:

1. Can a source by another author reach the public artifact? It must not be
   able to, by construction rather than by anyone remembering -- so the class
   the loader stamps is asserted against the packager's own SHIPPED vocabulary
   (deploy/manifest_contract.Distribution), and against the SQL the public
   dump actually runs.
2. Does the registry refuse what it cannot read? A silently skipped row is a
   source on disk that no longer has a stated reason for being held.
3. Does the classifier see the new directory at all? An unclassified file must
   fail the completeness predicate -- that failure is the entry point of the
   whole extension procedure, not a defect.

Nothing here needs a live Postgres.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import corpus_completeness
import external_registry
import pg_load_external
import public_dump
from external_registry import RegistryError
from manifest_contract import Distribution

HEADER = "| " + " | ".join(external_registry.COLUMNS) + " |\n| --- | --- | --- | --- | --- | --- |\n"
ROW = ("| paper.pdf | arxiv-oa | https://arxiv.org/abs/1 | arXiv non-exclusive "
       "licence | Author A. Title. Journal (2020) | задача 1 постановки |\n")


def registry(rows: str = ROW, header: str = HEADER) -> str:
    return "# Реестр\n\nПроза перед таблицей игнорируется.\n\n" + header + rows


class RegistryFormatTests(unittest.TestCase):
    def test_a_well_formed_row_becomes_a_source(self):
        (source,) = external_registry.parse_registry(registry())
        self.assertEqual(source.document_id, "paper")
        self.assertEqual(source.source_tier, "arxiv-oa")
        self.assertTrue(source.is_pdf)
        # note carries BOTH the bibliography and the reason: a reader who
        # lands on the page through search sees the note, not this registry.
        self.assertIn("Author A.", source.note)
        self.assertIn("задача 1", source.note)

    def test_a_bibliography_record_is_not_a_pdf(self):
        row = ROW.replace("paper.pdf", "bib_author_2020.md")
        (source,) = external_registry.parse_registry(registry(row))
        self.assertFalse(source.is_pdf)
        self.assertEqual(source.document_id, "bib_author_2020")

    def test_a_row_with_the_wrong_cell_count_raises_instead_of_being_skipped(self):
        # Skipping it would drop a source from the registry while its file
        # stays on disk -- exactly the state the registry exists to prevent.
        with self.assertRaises(RegistryError) as ctx:
            external_registry.parse_registry(registry(ROW.replace("| задача 1 постановки |", "|")))
        self.assertIn("ячеек вместо", str(ctx.exception))

    def test_every_column_is_mandatory(self):
        for empty in ("arxiv-oa", "arXiv non-exclusive licence",
                      "Author A. Title. Journal (2020)", "задача 1 постановки"):
            with self.subTest(empty=empty):
                with self.assertRaises(RegistryError):
                    external_registry.parse_registry(registry(ROW.replace(empty, "")))

    def test_a_source_without_a_canonical_url_is_refused(self):
        with self.assertRaises(RegistryError):
            external_registry.parse_registry(registry(ROW.replace("https://arxiv.org/abs/1", "-")))

    def test_an_unknown_suffix_is_refused(self):
        with self.assertRaises(RegistryError):
            external_registry.parse_registry(registry(ROW.replace("paper.pdf", "paper.djvu")))

    def test_a_duplicate_file_is_refused(self):
        with self.assertRaises(RegistryError):
            external_registry.parse_registry(registry(ROW + ROW))

    def test_a_file_without_the_table_is_refused(self):
        with self.assertRaises(RegistryError):
            external_registry.parse_registry("# Реестр\n\nтаблицы нет\n")


class RegistryAgainstDiskTests(unittest.TestCase):
    def _problems(self, files: tuple[str, ...], rows: str = ROW) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / external_registry.REGISTRY_FILENAME).write_text(registry(rows))
            for name in files:
                (directory / name).write_bytes(b"%PDF-1.4\n")
            sources = external_registry.load_registry(directory)
            return external_registry.registry_problems(directory, sources)

    def test_matching_registry_and_disk_is_clean(self):
        self.assertEqual(self._problems(("paper.pdf",)), [])

    def test_a_file_nobody_can_explain_is_a_problem(self):
        problems = self._problems(("paper.pdf", "mystery.pdf"))
        self.assertEqual(len(problems), 1)
        self.assertIn("mystery.pdf", problems[0])

    def test_a_row_without_its_file_is_a_problem(self):
        problems = self._problems(())
        self.assertEqual(len(problems), 1)
        self.assertIn("paper.pdf", problems[0])

    def test_a_directory_with_sources_but_no_registry_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "paper.pdf").write_bytes(b"%PDF-1.4\n")
            with self.assertRaises(RegistryError):
                external_registry.load_registry(Path(tmp))

    def test_no_external_tree_at_all_is_legitimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "external"
            self.assertEqual(external_registry.load_registry(missing), [])
            self.assertEqual(external_registry.registry_problems(missing, []), [])


class PublicBoundaryTests(unittest.TestCase):
    """Somebody else's copyright leaves no trace in the public artifact."""

    def test_the_class_ships_at_a_distribution_the_packager_never_ships(self):
        # The strongest form available without a live database: the value the
        # loader writes is not in the packager's own SHIPPED tuple, so no
        # documents row, no page row and no vector can be produced for it.
        self.assertNotIn(pg_load_external.PUBLIC_DISTRIBUTION, Distribution.SHIPPED)
        self.assertNotIn(pg_load_external.PUBLIC_DISTRIBUTION, Distribution.FULL_CONTENT)
        self.assertIn(pg_load_external.PUBLIC_DISTRIBUTION, Distribution.ALL)

    def test_the_dump_filters_such_a_document_out_of_both_tables(self):
        predicate = f"public_distribution IN ("
        documents = public_dump._copy_select("documents", ["id", "source_blob"])
        pages = public_dump._copy_select("pages", ["document_id", "page_number", "embedding"])
        for sql in (documents, pages):
            self.assertIn(predicate, sql)
            self.assertNotIn(f"'{pg_load_external.PUBLIC_DISTRIBUTION}'", sql)

    def test_the_loader_stamps_the_regime_and_does_not_take_it_from_the_registry(self):
        # Per-row wording decides the BASIS (which licence, checked when); it
        # never decides the distribution. A registry column that could say
        # "full-text" would put somebody else's copyright one typo away from
        # the artifact.
        self.assertEqual(pg_load_external.LEGAL_CLASS, "external-literature")
        self.assertNotIn("public_distribution", external_registry.COLUMNS)
        (source,) = external_registry.parse_registry(registry())
        self.assertFalse(hasattr(source, "public_distribution"))


class ClassifierTests(unittest.TestCase):
    def test_the_external_tree_is_classified(self):
        for rel, kind in (
            ("external/arxiv_2210_11331.pdf", "include-external"),
            ("external/bib_author_2020.md", "include-external"),
            ("external/EXTERNAL_INDEX.md", "include-metadata"),
            ("iis/1997_sm280.pdf", "include-pdf"),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(corpus_completeness.classify(rel)[0], kind)

    def test_an_unknown_file_still_fails_the_predicate(self):
        # The entry point of the extension procedure, not a defect: a new kind
        # of file must be included or excluded WITH a reason, by hand.
        kind, reason = corpus_completeness.classify("external/notes.txt")
        self.assertEqual(kind, "unclassified")
        self.assertIn("reason", reason)


if __name__ == "__main__":
    unittest.main()
