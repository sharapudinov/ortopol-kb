#!/usr/bin/env python3
"""Math-Net.Ru as the identity anchor: both titles of one work, at once.

Not a source of citations -- a source of NAMES. corpus.documents holds one
title per document, in whichever language Math-Net showed at export time
(some Russian, some the translated English). OpenAlex indexes sometimes the
original and sometimes the translation, so matching one language against the
other returns nothing where the work plainly exists. Measured in run 85: the
protocol found 43 of 69 works before Math-Net was consulted and 56 after --
the whole difference was cross-language misses.

The /rus/<id> page carries both citations in its <title>:

  И. И. Шарапудинов, "<рус. название>", Матем. сб., 180:9 (1989), ...;
  I. I. Sharapudinov, "<англ. название>", Math. USSR-Sb., 68:1 (1991), ...

Two traps, both paid for in run 53 and re-measured in 85: the page 302s to
archive.phtml, so redirects must be followed, and it is windows-1251, not
UTF-8. The HTTP 403 recorded in run 53 does not reproduce with an ordinary
User-Agent.
"""
from __future__ import annotations

import html
import re
import time
import urllib.request
from pathlib import Path

from .http_cache import cache_for

BASE = "https://www.mathnet.ru/rus/"
USER_AGENT = "Mozilla/5.0 (ortopol-kb-citations; mailto:tooba.mexico@gmail.com)"
QUOTED = re.compile("“(.*?)”", re.S)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
TAG = re.compile(r"<[^>]+>")
TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def mathnet_id(source_url: str | None) -> str | None:
    """'https://www.mathnet.ru/rus/mzm8442' -> 'mzm8442'.

    None for a document Math-Net does not carry (the Vestnik DNC paper and
    the four monographs): absence is a fact about the document, and the
    caller journals it rather than guessing an id from the filename.
    """
    if not source_url or "mathnet.ru" not in source_url:
        return None
    tail = str(source_url).rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def parse_titles(raw: str) -> tuple[list[str], list[int]]:
    """(titles, years) from the page head -- both languages, both years."""
    found = TITLE.search(raw)
    if not found:
        return [], []
    head = html.unescape(TAG.sub("", found.group(1))).replace("\xa0", " ")
    titles = [t.strip() for t in QUOTED.findall(head) if t.strip()]
    years = sorted({int(y) for y in YEAR.findall(head)})
    return titles, years


class MathnetClient:
    """Pages are cached on disk and failures are COUNTED, never swallowed.

    Both were paid for on 2026-09-02: an uncached run of 64 pages got the
    site to start timing out, five documents silently came back with no
    English title, and the twin index was quietly weaker than it looked --
    including for 2019_rm9846, the very pair the twin rule exists for. A gap
    in the identity anchor must be visible in the output, and a retry must
    not re-fetch the 59 pages that already worked.
    """

    def __init__(self, *, opener=urllib.request.urlopen, sleep=time.sleep,
                 pause=0.6, cache_dir: Path | None = None,
                 read_only_cache: bool = False):
        self._opener = opener
        self._sleep = sleep
        self.pause = pause
        self._cache = cache_for(cache_dir, read_only=read_only_cache)
        self.n_requests = 0
        self.failures: list[str] = []

    @property
    def n_cache_hits(self) -> int:
        return self._cache.hits if self._cache is not None else 0

    # A body shorter than this is a truncated page, not the page: it must
    # not stand in for one, and the cache enforces the floor on the read.
    PAGE_FLOOR = 2000

    def _cached(self, identifier: str) -> str:
        return f"{identifier}.html"

    def _no_citation_line(self, identifier: str) -> str:
        """Where "the site answered, and its <title> carries no citation" is
        recorded -- the archive-redirect page the module docstring describes.

        A marker of its own rather than the body alone: the body IS kept
        beside it (a better parser deserves the chance to re-read it), but
        such a page can be shorter than the 2000-byte floor that keeps a
        truncated body from being trusted, and then nothing would stand
        between it and a fresh request on every startup. The same rule
        ZbmathClient states: cache what the source ANSWERED, never cache a
        failure -- a transport error still bypasses this entirely.
        """
        return f"{identifier}.no-citation.json"

    def titles(self, identifier: str) -> tuple[list[str], list[int]]:
        """([titles], [years]); ([], []) with the id recorded in .failures."""
        page = self._cached(identifier)
        negative = self._no_citation_line(identifier)
        if self._cache is not None:
            hit = self._cache.read(page, floor=self.PAGE_FLOOR)
            if hit is not None:
                return parse_titles(hit)
            if self._cache.read(negative) is not None:
                self.failures.append(f"{identifier}: страница без цитат в <title>")
                return [], []
        request = urllib.request.Request(BASE + identifier,
                                         headers={"User-Agent": USER_AGENT})
        try:
            with self._opener(request, timeout=90) as response:
                raw = response.read().decode("windows-1251", "replace")
        except Exception as err:
            self.n_requests += 1
            self.failures.append(f"{identifier}: {type(err).__name__}")
            return [], []
        self.n_requests += 1
        self._sleep(self.pause)
        titles, years = parse_titles(raw)
        if self._cache is not None:
            self._cache.write(page, raw)
        if not titles:
            if self._cache is not None:
                self._cache.write(negative, '{"titles": []}')
            self.failures.append(f"{identifier}: страница без цитат в <title>")
            return [], []
        return titles, years
