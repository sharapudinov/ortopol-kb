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
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

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


def restore_abstract(inverted: dict | None) -> str | None:
    """`abstract_inverted_index` (word -> [positions]) back into text.

    Gaps in the position sequence are left alone rather than padded: the
    index is what the source published, and inventing filler words would
    put text we made up into work.abstract.
    """
    if not inverted:
        return None
    placed: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions or []:
            placed.append((int(position), word))
    if not placed:
        return None
    placed.sort()
    return " ".join(word for _, word in placed).strip() or None


def batched(items, size: int = ID_BATCH):
    """Chunks of at most `size` ids, with the measured cap enforced."""
    if size > ID_BATCH:
        raise ValueError(f"OpenAlex accepts at most {ID_BATCH} ids per filter, got {size}")
    if size < 1:
        raise ValueError(f"batch size must be positive, got {size}")
    seq = list(items)
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def short_id(value: str | None) -> str:
    """'https://openalex.org/W123' -> 'W123'; already-short ids pass through."""
    if not value:
        return ""
    return str(value).rstrip("/").rsplit("/", 1)[-1]


class OpenAlexClient:
    def __init__(
        self,
        *,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
        cache_dir: Path | None = None,
        quota_floor: int = QUOTA_FLOOR,
        max_quota_wait: float = MAX_QUOTA_WAIT,
        pause: float = PAUSE,
        tries: int = 5,
    ):
        self._opener = opener
        self._sleep = sleep
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_floor = quota_floor
        self.max_quota_wait = max_quota_wait
        self.pause = pause
        self.tries = tries
        self.n_requests = 0
        self.n_cache_hits = 0
        self.last_rate: dict[str, str] = {}

    # -- HTTP ------------------------------------------------------------
    def url(self, path: str, **params) -> str:
        params["mailto"] = MAILTO
        return f"{API}/{path}?" + urllib.parse.urlencode(params, safe="|:.,-")

    def _cache_path(self, url: str) -> Path | None:
        if not self._cache_dir:
            return None
        return self._cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

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
        cached = self._cache_path(url)
        if cached is not None and cached.is_file():
            self.n_cache_hits += 1
            return json.loads(cached.read_text(encoding="utf-8"))

        delay = 2.0
        for attempt in range(1, self.tries + 1):
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            try:
                with self._opener(request, timeout=120) as response:
                    raw = response.read().decode("utf-8", "replace")
                    headers = response.headers
                    code = getattr(response, "status", 200)
            except urllib.error.HTTPError as err:
                raw, headers, code = err.read().decode("utf-8", "replace"), err.headers, err.code
            except Exception as err:  # transport: DNS, timeout, reset
                raw, headers, code = json.dumps({"_transport_error": str(err)}), {}, 0
            self.n_requests += 1
            self.last_rate = {
                name: headers.get(name)
                for name in RATE_HEADERS
                if hasattr(headers, "get") and headers.get(name) is not None
            }
            if code in RETRY_CODES and attempt < self.tries:
                pause = float(self.last_rate.get("retry-after") or delay)
                self._sleep(min(pause, 120.0))
                delay = min(delay * 2, 120.0)
                continue
            if code != 200:
                raise OpenAlexError(f"HTTP {code} от {url}: {raw[:300]}")
            self._sleep(self.pause)
            self._honour_quota()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as err:
                raise OpenAlexError(f"не JSON от {url}: {err}") from err
            if cached is not None:
                cached.write_text(raw, encoding="utf-8")
            return body
        raise OpenAlexError(f"исчерпаны {self.tries} попытки: {url}")

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
