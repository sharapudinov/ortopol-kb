"""TDD tests for pdf_extract.py — the pdftotext + classify glue,
including page splitting (pdftotext's form-feed page breaks)."""
import unittest

import _pathfix  # noqa: F401
import encoding
from paths import default_pdf_dir
from pdf_extract import extract_document

# See test_encoding.py's PDF_DIR for why this is paths.default_pdf_dir()
# rather than a hardcoded Path(__file__).parents[N] depth.
PDF_DIR = default_pdf_dir()


class ExtractDocumentTests(unittest.TestCase):
    def test_extracts_clean_text_layer(self):
        doc = extract_document(PDF_DIR / "2019_smj3104.pdf")
        self.assertEqual(doc.category, encoding.Category.CLEAN)
        self.assertGreater(len(doc.pages), 0)
        joined = "\n".join(doc.pages)
        self.assertIn("Валле", joined)
        self.assertIn("второго порядка", joined)

    def test_recovers_mojibake_files(self):
        doc = extract_document(PDF_DIR / "2015_demr1.pdf")
        self.assertEqual(doc.category, encoding.Category.MOJIBAKE_RECOVERED)
        self.assertGreater(len(doc.pages), 0)
        self.assertIn("Дагестанские", "\n".join(doc.pages))

    def test_broken_files_produce_no_pages(self):
        doc = extract_document(PDF_DIR / "1996_sm105.pdf")
        self.assertEqual(doc.category, encoding.Category.BROKEN)
        self.assertEqual(doc.pages, [])
        self.assertIn("not recoverable", doc.note)


if __name__ == "__main__":
    unittest.main()


class DegradedExtractionTests(unittest.TestCase):
    def test_degraded_files_still_yield_pages(self):
        # Смысл категории: текст индексируется, потери записаны в note.
        doc = extract_document(PDF_DIR / "2017_demr34.pdf")
        self.assertEqual(doc.category, encoding.Category.DEGRADED)
        self.assertGreater(len(doc.pages), 0)
        self.assertIn("ligature", doc.note)
