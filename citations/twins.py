#!/usr/bin/env python3
"""Twins of OUR OWN works: the rule of survey §7, aimed at the corpus.

registry.py unions records that share an identifier. That is not enough for
the pair that matters most. Observed on live data: `pg_graph.py candidates`
ranked W2972250820 "Sobolev-orthogonal systems of functions and some of
their applications" (2019, 13 links) third among the works we should read
next -- it is the English translation of 2019_rm9846, already in the corpus
as W2966149037. The original carries 10.4213/rm9846 and the translation
10.1070/RM2019v074n04ABEH004856: different DOIs, disjoint id sets, so the id
union cannot see it. Recommending a translation of our own paper as
something to go read is the failure this closes.

Two exact tests, applied in that order.

1. **Math-Net's own DOI convention.** Measured on the live graph: the
   original carries 10.4213/<mathnet id> and the translation
   10.1070/<mathnet id> -- SAME SUFFIX, different registrant. So a node whose
   DOI suffix equals a corpus document's Math-Net id IS that document, in any
   language, with no string comparison of titles at all. This is what catches
   2019_rm9846 / W2972250820, and 13 nodes in total on the current graph.
2. **Normalized title + year within +-1**, the rule run 85 used against the
   sources, pointed the other way. Needed because rule 1 only fires where
   Math-Net minted both DOIs; it uses the Russian AND English names off the
   Math-Net page. Measured limitation: the /rus/rm9846 page prints the
   translation's journal reference but NOT its English title, so for that
   document rule 2 alone would have found nothing -- which is exactly why
   rule 1 exists and goes first.

Normalization is deliberately blunt (lowercase, letters and digits only,
ё -> е): the two sides come from different typesetting traditions and differ
in punctuation, dashes and TeX residue far more often than in words. Unlike
match.py's protocol there is no fuzzy tier here -- promoting a work to
'our-document' rewrites what the corpus claims about itself, so only exact
normalized equality is allowed to do it.
"""
from __future__ import annotations

import re

TEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
ALNUM = re.compile(r"[0-9a-zа-я]+")


def doi_suffix(doi: str | None) -> str:
    """'https://doi.org/10.1070/RM9846' -> 'rm9846'. Math-Net mints the
    original and the translation with the same suffix and different
    registrants (10.4213 / 10.1070), so the suffix IS the work's name."""
    if not doi:
        return ""
    return str(doi).strip().lower().rstrip(".,").rsplit("/", 1)[-1]


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    cleaned = TEX_COMMAND.sub(" ", str(title).replace("$", " "))
    return "".join(ALNUM.findall(cleaned.lower().replace("ё", "е")))


def year_matches(candidate_year, document_years, span: int = 1) -> bool:
    """A translation lands a year after its original, hence +-1.

    A document with no year at all (the four monographs) matches on the title
    alone: refusing every yearless document would make the rule silently
    inapplicable to exactly the works that are hardest to identify.
    """
    years = [y for y in (document_years or []) if y]
    if not years:
        return True
    if candidate_year is None:
        return False
    return any(abs(int(candidate_year) - int(y)) <= span for y in years)


def build_index(seeds) -> dict[str, tuple[str, str]]:
    """normalized title -> (document_id, seed key), over every known title.

    `seeds` are dicts with keys `key`, `document_id`, `titles`. A title
    claimed by two DIFFERENT documents is dropped from the index rather than
    resolved: an ambiguous name must not silently promote a work into the
    wrong document.
    """
    index: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for seed in seeds:
        for title in seed.get("titles") or []:
            key = normalize_title(title)
            if not key:
                continue
            held = index.get(key)
            if held is not None and held[0] != seed["document_id"]:
                ambiguous.add(key)
            index[key] = (seed["document_id"], seed["key"])
    for key in ambiguous:
        index.pop(key, None)
    return index


def build_mathnet_index(seeds) -> dict[str, tuple[str, str]]:
    """Math-Net id -> (document_id, seed key), for rule 1."""
    index: dict[str, tuple[str, str]] = {}
    for seed in seeds:
        identifier = (seed.get("mathnet_id") or "").strip().lower()
        if identifier:
            index[identifier] = (seed["document_id"], seed["key"])
    return index


def find_twin(title, year, doi, title_index, mathnet_index, years_of):
    """(document_id, seed key, rule) when this work IS one of ours, else None.

    No year check on rule 1: a DOI suffix minted by Math-Net for this exact
    work is a stronger statement than any year window, and translations are
    routinely dated a year or two off their original.
    """
    hit = mathnet_index.get(doi_suffix(doi))
    if hit is not None:
        return hit[0], hit[1], "mathnet-doi"
    hit = title_index.get(normalize_title(title))
    if hit is None:
        return None
    document_id, seed_key = hit
    if not year_matches(year, years_of.get(document_id)):
        return None
    return document_id, seed_key, "title+year"
