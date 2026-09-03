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

from .http_session import HttpSession

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
                 pause=0.6, cache=None):
        # windows-1251, not UTF-8 (module docstring), and no retry: a page
        # that did not arrive is COUNTED, and a silent second attempt is
        # exactly the thing that made the gap invisible for 2019_rm9846.
        self._session = HttpSession(
            user_agent=USER_AGENT, opener=opener, sleep=sleep, pause=pause,
            timeout=90, encoding="windows-1251", cache=cache)
        self.pause = pause
        self.failures: list[str] = []
        # The same gaps, keyed by the id they are about. .failures is a list
        # of sentences for a human to read; a caller that has to JOURNAL the
        # gap needs the page it belongs to, and re-deriving that by parsing
        # the sentence back apart is the prose-parsing this repository bans
        # outright one layer down (JOURNAL_FACTS_ARE_COLUMNS). One writer
        # (_note) fills both, so they cannot say different things.
        self.problems: dict[str, str] = {}

    @property
    def n_requests(self) -> int:
        return self._session.n_requests

    @property
    def n_cache_hits(self) -> int:
        return self._session.n_cache_hits

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

    NO_CITATION = "страница без цитат в <title>"

    def _note(self, identifier: str, what: str) -> None:
        """Records one gap in both channels: the sentence for the log, and
        the (id -> what) pair for a caller that journals it."""
        self.failures.append(f"{identifier}: {what}")
        self.problems[identifier] = what

    def titles(self, identifier: str) -> tuple[list[str], list[int]]:
        """([titles], [years]); ([], []) with the id recorded in .failures
        and in .problems."""
        page = self._cached(identifier)
        negative = self._no_citation_line(identifier)
        hit = self._session.cached(page, floor=self.PAGE_FLOOR)
        if hit is not None:
            return parse_titles(hit)
        if self._session.cached(negative) is not None:
            self._note(identifier, self.NO_CITATION)
            return [], []
        answer = self._session.fetch(BASE + identifier)
        if answer.problem():
            # Never cached: a blank stored here would turn one timeout (or
            # one 503) into a permanent verdict about this document.
            self._note(identifier, answer.problem())
            return [], []
        titles, years = parse_titles(answer.body)
        self._session.store(page, answer.body)
        if not titles:
            self._session.store(negative, '{"titles": []}')
            self._note(identifier, self.NO_CITATION)
            return [], []
        return titles, years
