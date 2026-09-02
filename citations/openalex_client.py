#!/usr/bin/env python3
"""OpenAlex HTTP client for the snowball crawl.

stdlib only, for the same reason the rest of this repository is: the
installer runs on other people's machines and may not add packages.

Three things here are measured facts, not conventions (survey.md §8,
run 85):

- an id filter accepts exactly 50 values per request (`cites:W1|...|W50` ->
  HTTP 200 / 285 works; `openalex_id:` -> 50). ID_BATCH is that cap and
  batched() refuses to exceed it, so a future caller cannot silently turn
  one request into a 4xx;
- `x-ratelimit-*` describes a *shared, slowly refilling* budget, not a
  per-second throttle: measured 289 of 1000 remaining with
  `x-ratelimit-reset: 83942` (~23 h). Waiting out that reset is not a
  strategy, so falling under the floor raises QuotaExhausted past
  max_quota_wait instead of sleeping the crawl into next day; the caller
  journals it and stops with everything written so far still in the base;
- the abstract arrives only as `abstract_inverted_index` (word -> positions)
  and has to be reassembled.

The optional on-disk cache exists for the same quota reason: the tau
calibration and the crawl proper fetch the same depth-1 pages, and with a
budget of a few hundred requests paying twice is a real cost. Cached bodies
are disposable scratch -- the durable evidence is citation.work.evidence in
the database.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator

from .http_session import HttpSession, Retry
from .openalex_records import (  # noqa: F401  (re-exported: one import site per name)
    SIDECAR_SUFFIX,
    note_direction,
    page_index,
    restore_abstract,
    short_id,
    sidecar_name,
)

API = "https://api.openalex.org"
MAILTO = "tooba.mexico@gmail.com"
USER_AGENT = f"ortopol-kb-citations/1.0 (mailto:{MAILTO})"

ID_BATCH = 50
PER_PAGE = 200

WORK_SELECT = (
    "id,doi,title,display_name,publication_year,abstract_inverted_index,ids,"
    "referenced_works,referenced_works_count,cited_by_count,authorships,"
    "type,language"
)

RATE_HEADERS = (
    "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "x-ratelimit-credits-used", "x-ratelimit-cost-usd",
    "x-ratelimit-remaining-usd", "retry-after",
)
QUOTA_FLOOR = 30
MAX_QUOTA_WAIT = 900.0
PAUSE = 0.35
RETRY_CODES = (429, 500, 502, 503, 504, 0)


class QuotaExhausted(RuntimeError):
    """The remaining request budget is under the floor and the reset is
    further away than the caller is willing to sleep."""


class OpenAlexError(RuntimeError):
    """A request failed after every retry, or came back unparseable."""


def batched(items, size: int = ID_BATCH):
    """Chunks of at most `size` ids, with the measured cap enforced."""
    if size > ID_BATCH:
        raise ValueError(f"OpenAlex accepts at most {ID_BATCH} ids per filter, got {size}")
    if size < 1:
        raise ValueError(f"batch size must be positive, got {size}")
    seq = list(items)
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


class OpenAlexClient:
    """The quota policy on top of the shared request layer (http_session).

    What is this client's and not the session's: the retry SET (429, the
    5xx family and a transport failure are the answers a shared public API
    gives under load; a 404 is not), the quota headers read off every
    answer, and the refusal to sleep out a reset window that is hours away.
    """

    def __init__(
        self,
        *,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
        cache=None,
        quota_floor: int = QUOTA_FLOOR,
        max_quota_wait: float = MAX_QUOTA_WAIT,
        pause: float = PAUSE,
        tries: int = 5,
    ):
        self._sleep = sleep
        self._session = HttpSession(
            user_agent=USER_AGENT, accept="application/json", opener=opener,
            sleep=sleep, pause=pause, timeout=120, cache=cache,
            retry=Retry(tries=tries, codes=RETRY_CODES),
        )
        self.quota_floor = quota_floor
        self.max_quota_wait = max_quota_wait
        self.pause = pause
        self.tries = tries
        self.last_rate: dict[str, str] = {}

    @property
    def n_requests(self) -> int:
        return self._session.n_requests

    @property
    def n_cache_hits(self) -> int:
        return self._session.n_cache_hits

    # -- HTTP ------------------------------------------------------------
    def url(self, path: str, **params) -> str:
        params["mailto"] = MAILTO
        return f"{API}/{path}?" + urllib.parse.urlencode(params, safe="|:.,-")

    def _cache_name(self, url: str) -> str | None:
        if self._session.cache is None:
            return None
        return hashlib.sha1(url.encode()).hexdigest() + ".json"

    def _honour_quota(self) -> None:
        """Called after every response; the budget is read, never guessed."""
        remaining = self.last_rate.get("x-ratelimit-remaining")
        if remaining is None or not str(remaining).lstrip("-").isdigit():
            return
        if int(remaining) >= self.quota_floor:
            return
        reset = self.last_rate.get("x-ratelimit-reset") or "0"
        wait = float(reset) if str(reset).replace(".", "", 1).isdigit() else 0.0
        if wait > self.max_quota_wait:
            raise QuotaExhausted(
                f"осталось {remaining} запросов OpenAlex, окно сбросится через "
                f"{wait:.0f} с — это больше допустимого ожидания "
                f"{self.max_quota_wait:.0f} с; обход остановлен, записанное сохранено"
            )
        self._sleep(wait)

    def get_json(self, url: str) -> dict:
        cached = self._cache_name(url)
        hit = self._session.cached(cached)
        if hit is not None:
            return json.loads(hit)

        answer = self._session.fetch(url)
        self.last_rate = {
            name: answer.headers.get(name)
            for name in RATE_HEADERS
            if hasattr(answer.headers, "get") and answer.headers.get(name) is not None
        }
        if answer.code != 200:
            raise OpenAlexError(f"HTTP {answer.code} от {url}: {answer.detail()[:300]}")
        self._honour_quota()
        try:
            body = json.loads(answer.body)
        except json.JSONDecodeError as err:
            raise OpenAlexError(f"не JSON от {url}: {err}") from err
        if cached is not None:
            self._session.store(cached, answer.body)
            self._write_sidecar(cached, body)
        return body

    def _write_sidecar(self, cached: str, body: dict) -> None:
        """The page's own index, written the moment the page is cached.

        A reader that needs only what batch a page belongs to and how many
        works the batch promised (citations/hub_cache.py) would otherwise
        json.loads() the whole cache to find two numbers: 253 pages, 217
        MiB, every one a full object graph. Best effort -- the cache is
        disposable scratch, and a directory that cannot be written is the
        caller's problem, not a failed request's.
        """
        try:
            self._session.store(sidecar_name(cached),
                                json.dumps(page_index(body), ensure_ascii=False))
        except OSError:
            pass

    # -- queries ---------------------------------------------------------
    def _paged(self, **params) -> Iterator[dict]:
        """Cursor pagination; `*` is OpenAlex's first-cursor sentinel.

        A generator, and the two query methods below stay generators over
        it: one depth-2 batch set answered with over 51000 works (crawl.py's
        docstring), of which the caller keeps the few that reference the
        frontier. Returning a list held every page of every batch at once,
        each record carrying the bulky referenced_works, before the caller
        could drop a single one. Nothing is requested until the consumer
        asks for the next record.
        """
        cursor = "*"
        while cursor:
            body = self.get_json(self.url("works", cursor=cursor, **params))
            results = body.get("results") or []
            yield from results
            if not results:
                break
            cursor = (body.get("meta") or {}).get("next_cursor")

    def works_by_ids(self, ids) -> Iterator[dict]:
        """Metadata for up to 50 OpenAlex ids per request."""
        for chunk in batched(short_id(i) for i in ids):
            if not chunk:
                continue
            yield from self._paged(
                filter="openalex_id:" + "|".join(chunk),
                select=WORK_SELECT,
                **{"per-page": PER_PAGE},
            )

    def citers_of(self, ids) -> Iterator[dict]:
        """Everything citing ANY of `ids`, 50 cited ids per request.

        The response does not say which of the 50 a given citer cites --
        that is recovered from the citer's own `referenced_works`, which is
        why WORK_SELECT keeps that field even though it is the bulkiest one.
        """
        for chunk in batched(short_id(i) for i in ids):
            if not chunk:
                continue
            yield from self._paged(
                filter="cites:" + "|".join(chunk),
                select=WORK_SELECT,
                **{"per-page": PER_PAGE},
            )
