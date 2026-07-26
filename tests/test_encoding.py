"""TDD tests for encoding.py classification.

Ground truth (measured by a full sweep over all 67 PDFs under theory/iis/,
cross-checked corpus-wide in test_build_corpus.py): 40 clean, 21
cp1251-as-latin1 mojibake recoveries, 2 degraded, 4 broken, 0 image-only
scans.
"""
import subprocess
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import encoding
from paths import default_pdf_dir

# paths.default_pdf_dir(), not Path(__file__).parents[N]: a hardcoded
# ancestor depth silently resolved to the WRONG directory the moment this
# tree moved (ortopol/lib/tools/corpus/tests/ -> <repo>/tests/, one of the
# failures the extraction of this repository produced), and a
# missing theory/iis/ shows up as "pdftotext: no such file" rather than as
# the layout error it is. paths.py locates the data tree by walking up for
# it, so it is correct under any checkout depth.
PDF_DIR = default_pdf_dir()

# Измерено: плотность букв 0.023-0.067 — глифы перемешаны, текста нет.
BROKEN_STEMS = [
    "1996_sm105",
    "1997_sm280",
    "2000_sm480",
    "2003_sm723",
]


def _pdftotext(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, timeout=60
    )
    return result.stdout.decode("utf-8", errors="replace")


class ClassifyAndRecoverTests(unittest.TestCase):
    def test_extracts_clean_text_layer(self):
        text = _pdftotext(PDF_DIR / "2019_smj3104.pdf")
        result = encoding.classify_and_recover(text)
        self.assertEqual(result.category, encoding.Category.CLEAN)
        self.assertIn("Валле", result.text)
        self.assertIn("Пуссена", result.text)

    def test_recovers_mojibake_files(self):
        text = _pdftotext(PDF_DIR / "2015_demr1.pdf")
        result = encoding.classify_and_recover(text)
        self.assertEqual(result.category, encoding.Category.MOJIBAKE_RECOVERED)
        self.assertIn("Дагестанские", result.text)
        self.assertNotIn("Äàãåñòàíñêèå", result.text)

    def test_reports_unreadable_files_explicitly(self):
        for stem in BROKEN_STEMS:
            with self.subTest(stem=stem):
                text = _pdftotext(PDF_DIR / f"{stem}.pdf")
                result = encoding.classify_and_recover(text)
                self.assertEqual(result.category, encoding.Category.BROKEN)

    def test_synthetic_mojibake_roundtrip(self):
        # Self-contained check independent of any PDF: encode a known
        # Cyrillic string as cp1251 bytes, misread as Latin-1 (exactly the
        # bug pdftotext hits), and confirm recovery undoes it losslessly.
        original = (
            "Дагестанские Электронные Математические Известия. "
            "Специальный выпуск. Смешанные ряды по классическим "
            "ортогональным полиномам."
        )
        mojibake = original.encode("cp1251").decode("latin-1")
        result = encoding.classify_and_recover(mojibake)
        self.assertEqual(result.category, encoding.Category.MOJIBAKE_RECOVERED)
        self.assertEqual(result.text, original)

    def test_no_text_layer_below_letter_floor(self):
        result = encoding.classify_and_recover("   \n\x0c   123 456 \n")
        self.assertEqual(result.category, encoding.Category.NO_TEXT_LAYER)


if __name__ == "__main__":
    unittest.main()


# Плотность букв 0.705-0.717 — ВЫШЕ, чем у заведомо чистых файлов (0.645-0.701).
# Выпали только лигатуры ff/fi, ~0.3% слов. Выбрасывать их из индекса было ошибкой.
DEGRADED_STEMS = ["2017_demr32", "2017_demr34"]


class DegradedClassificationTests(unittest.TestCase):
    def test_readable_files_with_lost_ligatures_are_degraded_not_broken(self):
        for stem in DEGRADED_STEMS:
            with self.subTest(stem=stem):
                raw = _pdftotext(PDF_DIR / f"{stem}.pdf")
                result = encoding.classify_and_recover(raw)
                self.assertEqual(result.category, encoding.Category.DEGRADED)

    def test_degraded_letter_density_exceeds_the_scrambled_band(self):
        # Порог измерен, а не выбран: между полосами разрыв на порядок.
        for stem in DEGRADED_STEMS:
            with self.subTest(stem=stem):
                raw = _pdftotext(PDF_DIR / f"{stem}.pdf")
                self.assertGreater(encoding.letter_density(raw), 0.6)
        for stem in BROKEN_STEMS:
            with self.subTest(stem=stem):
                raw = _pdftotext(PDF_DIR / f"{stem}.pdf")
                self.assertLess(encoding.letter_density(raw), 0.1)
