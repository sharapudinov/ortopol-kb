"""Run pdftotext over one PDF and classify/recover its encoding.

Page boundaries come for free: pdftotext inserts a form-feed (\\x0c) between
pages unless told not to, so splitting on it gives page-level granularity
without any extra parsing.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from encoding import Category, classify_and_recover

PDFTOTEXT_TIMEOUT_SECONDS = 120

_NOTES = {
    Category.CLEAN: "clean text layer",
    Category.MOJIBAKE_RECOVERED: "cp1251-as-latin1 mojibake, recovered by re-encoding",
    Category.DEGRADED: "readable text with known losses: some ligature glyphs (ff/fi) have no ToUnicode entry and drop out (~0.3% of words). Indexed on purpose — discarding tens of thousands of usable characters over that is the worse trade",
    Category.BROKEN: "broken font encoding, no ToUnicode map; not recoverable by re-encoding",
    Category.NO_TEXT_LAYER: "no extractable text layer (image-only scan)",
}


class PdfTotextError(RuntimeError):
    pass


@dataclass(frozen=True)
class DocumentExtraction:
    stem: str
    filename: str
    category: Category
    pages: list[str]  # empty for BROKEN / NO_TEXT_LAYER; populated for DEGRADED
    chars_extracted: int
    note: str


def run_pdftotext(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            timeout=PDFTOTEXT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise PdfTotextError("pdftotext not found on PATH; install poppler-utils") from exc
    except subprocess.TimeoutExpired as exc:
        raise PdfTotextError(f"pdftotext timed out on {pdf_path}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise PdfTotextError(f"pdftotext failed on {pdf_path}: {stderr}")
    return result.stdout.decode("utf-8", errors="replace")


def _split_pages(text: str) -> list[str]:
    pages = text.split("\f")
    # pdftotext emits a trailing form feed after the last page; drop the
    # resulting empty tail chunk rather than storing a spurious blank page.
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    return pages


def extract_document(pdf_path: Path) -> DocumentExtraction:
    raw_text = run_pdftotext(pdf_path)
    classification = classify_and_recover(raw_text)

    # DEGRADED belongs here: its text is prose with a handful of glyphs missing,
    # not noise. Withholding it leaves a real source out of the index entirely,
    # which is a far larger loss than the words it is missing.
    if classification.category in (
        Category.CLEAN,
        Category.MOJIBAKE_RECOVERED,
        Category.DEGRADED,
    ):
        pages = _split_pages(classification.text)
    else:
        pages = []

    return DocumentExtraction(
        stem=pdf_path.stem,
        filename=pdf_path.name,
        category=classification.category,
        pages=pages,
        chars_extracted=classification.stats.total,
        note=_NOTES[classification.category],
    )
