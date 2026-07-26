"""Classify pdftotext output by encoding health and recover cp1251-as-latin1 mojibake.

Three failure modes show up in this corpus (measured over all 67 PDFs under
theory/iis/, see tests/test_report.py):

- CLEAN: pdftotext already produced correct Cyrillic (or the paper is in
  English/Latin script to begin with). No transformation needed.
- MOJIBAKE: the PDF's font declares cp1251 code points but no ToUnicode map,
  and pdftotext falls back to treating the raw bytes as Latin-1. Every
  Cyrillic letter (cp1251 0xC0-0xFF) reappears as a Latin-1 accented letter
  in the same byte range (e.g. "Дагестанские" -> "Äàãåñòàíñêèå"). This is
  fully recoverable: re-encode to Latin-1 bytes, decode as cp1251.
- BROKEN: the font has neither a ToUnicode map nor a code page pdftotext can
  guess (arbitrary glyph-index-to-codepoint mapping). Output is a mix of
  stray Latin Extended-A letters, private-use characters and mis-picked
  ASCII with no systematic correction. Not recoverable by re-encoding.

Thresholds below are measured on the real corpus, not chosen to fit a
result. Two independent signals, both over all letter-like characters:

- Cyrillic ratio: clean files sit at 0.60-0.93; every mojibake and broken
  file sits at exactly 0.00 (pdftotext never produces a Cyrillic code point
  for either failure mode).
- Latin-1 Supplement ratio (the mojibake fingerprint: cp1251 bytes 0xC0-0xFF
  misread as Latin-1 land in U+00C0-U+00FF): clean files aren't exactly at
  0 — stray accented Latin letters in citations/foreign names put them at
  0.0000-0.0009 — but the 21 mojibake files sit at 0.32-0.78 and the 6
  broken files sit at 0.007-0.055. Both bands clear the clean-file noise
  floor, so *any* non-trivial Latin-1 Supplement presence is treated as
  "the encoding bug touched this document somewhere" and triggers a
  recode attempt. What separates mojibake from broken is not that ratio
  itself but whether the recode actually produces dominant Cyrillic text:
  recovered mojibake reaches 0.50-0.79 Cyrillic, while recovered broken
  files stay at 0.007-0.11 — because in the broken files the bug affects
  only a small fragment (e.g. an abstract) while the bulk of the document
  has separately, unrecoverably lost characters (missing ligature glyphs
  with no ToUnicode entry at all).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    CLEAN = "clean"
    MOJIBAKE_RECOVERED = "mojibake_recovered"
    DEGRADED = "degraded"
    BROKEN = "broken"
    NO_TEXT_LAYER = "no_text_layer"


# Measured thresholds — see module docstring for the observed distributions.
CYR_RATIO_CLEAN_MIN = 0.30
LATIN_SUPPLEMENT_CONTAMINATION_FLOOR = 0.003  # clean noise tops out at 0.0009; broken bottoms out at 0.0067
MIN_LETTERS_FOR_TEXT_LAYER = 50

# Letter density = letters / non-whitespace characters. It separates a document
# whose glyphs are scrambled from one that merely lost a few of them, and the
# separation is an order of magnitude wide. Measured over the six files this
# classifier previously lumped together as BROKEN:
#     1996_sm105 0.031   1997_sm280 0.034   2000_sm480 0.067   2003_sm723 0.023
#     2017_demr32 0.717  2017_demr34 0.705
# For scale, files that are unambiguously clean sit at 0.645-0.701 — so the two
# 2017 papers read BETTER than known-good ones. Discarding 44k characters of
# usable text (including a source for the Sobolev-operator question) over ~0.3%
# of words missing an ff/fi ligature was the wrong trade.
LETTER_DENSITY_READABLE_MIN = 0.30  # anywhere in the 0.067-0.645 gap; 0.30 mirrors CYR_RATIO_CLEAN_MIN

_CYRILLIC_RANGE = (0x0400, 0x04FF)
_LATIN1_SUPPLEMENT_RANGE = (0x00C0, 0x00FF)

# cp1251 byte values 0x80-0xFF, decoded once as the correction table used by
# recover_mojibake(). Bytes with no cp1251 assignment are simply omitted, so
# str.translate() leaves the original (already-wrong) character untouched
# rather than raising.
_CP1251_FROM_LATIN1_BYTE: dict[int, str] = {}
for _byte in range(0x80, 0x100):
    try:
        _CP1251_FROM_LATIN1_BYTE[_byte] = bytes([_byte]).decode("cp1251")
    except UnicodeDecodeError:
        pass


@dataclass(frozen=True)
class LetterStats:
    cyrillic: int
    latin1_supplement: int
    ascii_letters: int
    other_letters: int

    @property
    def total(self) -> int:
        return self.cyrillic + self.latin1_supplement + self.ascii_letters + self.other_letters

    @property
    def cyrillic_ratio(self) -> float:
        return self.cyrillic / self.total if self.total else 0.0

    @property
    def latin1_supplement_ratio(self) -> float:
        return self.latin1_supplement / self.total if self.total else 0.0


def letter_stats(text: str) -> LetterStats:
    cyr = lat_sup = ascii_l = other = 0
    for ch in text:
        cp = ord(ch)
        if _CYRILLIC_RANGE[0] <= cp <= _CYRILLIC_RANGE[1]:
            cyr += 1
        elif _LATIN1_SUPPLEMENT_RANGE[0] <= cp <= _LATIN1_SUPPLEMENT_RANGE[1]:
            lat_sup += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            ascii_l += 1
        elif ch.isalpha():
            other += 1
    return LetterStats(cyr, lat_sup, ascii_l, other)


def letter_density(text: str) -> float:
    """Letters per non-whitespace character.

    A scrambled-glyph document is mostly punctuation and private-use noise; a
    document that merely lost some ligatures still reads as prose. See
    LETTER_DENSITY_READABLE_MIN for the measured bands.
    """
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return sum(1 for ch in compact if ch.isalpha()) / len(compact)


def recover_mojibake(text: str) -> str:
    """Re-map cp1251-as-latin1 mojibake back to the intended Cyrillic text."""
    return text.translate(_CP1251_FROM_LATIN1_BYTE)


@dataclass(frozen=True)
class Classification:
    category: Category
    text: str
    stats: LetterStats


def classify_and_recover(raw_text: str) -> Classification:
    """Decide what pdftotext actually gave us and recover it if possible."""
    stats = letter_stats(raw_text)

    if stats.total < MIN_LETTERS_FOR_TEXT_LAYER:
        return Classification(Category.NO_TEXT_LAYER, raw_text, stats)

    if stats.cyrillic_ratio >= CYR_RATIO_CLEAN_MIN:
        return Classification(Category.CLEAN, raw_text, stats)

    if stats.latin1_supplement_ratio > LATIN_SUPPLEMENT_CONTAMINATION_FLOOR:
        # The mojibake fingerprint is present somewhere in this document.
        # Whether that means the whole thing recovers cleanly (MOJIBAKE) or
        # only a fragment does while the rest is separately broken depends
        # on whether recoding actually makes Cyrillic dominant.
        recovered = recover_mojibake(raw_text)
        recovered_stats = letter_stats(recovered)
        if recovered_stats.cyrillic_ratio >= CYR_RATIO_CLEAN_MIN:
            return Classification(Category.MOJIBAKE_RECOVERED, recovered, recovered_stats)
        # Recoding did not make Cyrillic dominant — but that alone does not mean
        # the document is lost. Letter density separates the two cases cleanly.
        if letter_density(raw_text) >= LETTER_DENSITY_READABLE_MIN:
            return Classification(Category.DEGRADED, raw_text, stats)
        return Classification(Category.BROKEN, raw_text, stats)

    # No Cyrillic and no mojibake fingerprint: presumably a legitimately
    # non-Cyrillic (e.g. English-only) document, not an encoding failure.
    return Classification(Category.CLEAN, raw_text, stats)
