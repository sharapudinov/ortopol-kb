#!/usr/bin/env python3
"""The request half of fetching, beside the storage half (http_cache.py).

http_cache.py extracted ONE object for where a response is kept, and all
three clients took it. The request itself stayed written three times:
OpenAlexClient.get_json, ZbmathClient._fetch and MathnetClient.titles each
built their own urllib Request, carried their own User-Agent constant,
their own timeout, their own `self._sleep(self.pause)`, their own
n_requests bookkeeping and their own broad `except Exception`. Three
sources needing the same mechanism is the signal to deepen the seam, not
to copy it a fourth time -- a proxy, a transport taxonomy or a request
accounting for the quota journal would otherwise be three edits that can
silently disagree.

What is shared is the MECHANISM: build the request, open it, decode the
body, count it, wait the polite pause, and retry by a policy handed in.
What stays with each client is its POLICY -- the three genuinely differ,
and none of them is more correct than the others:

  OpenAlex   retries 429/5xx/transport with doubling backoff the server's
             retry-after can override, reads quota headers off the answer,
             and raises OpenAlexError on anything else.
  zbMATH     404 is an ANSWER (no record), every other failure raises
             ZbmathUnavailable, because "we asked and it does not have it"
             and "we did not learn anything" must never read the same.
  Math-Net   nothing raises: a failed page is counted in .failures and the
             caller sees an empty title list.

So fetch() never raises for a failed request and never decides what a
status means. It returns a Response carrying everything the three policies
read -- the body, the status, the headers, and the transport exception
itself when nothing answered at all (which is what lets Math-Net report
the exception's class name while zbMATH reports its text).

The pause is taken after the answer that is RETURNED, retried attempts
being paced by the retry policy instead. It is a politeness throttle on a
shared public API, not a success ritual: an error costs the far side the
same request a 200 does.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, NamedTuple

from .http_cache import cache_for


class Retry(NamedTuple):
    """How many times, on what, and how long between.

    The default is no retry at all: a client that wants one says so, and
    one that does not cannot inherit one by accident. `ceiling` caps both
    the doubling backoff and whatever the server asks for in `header` --
    a retry-after of 86400 is not a schedule, it is a refusal.
    """

    tries: int = 1
    codes: tuple[int, ...] = ()
    delay: float = 2.0
    ceiling: float = 120.0
    header: str = "retry-after"


class Response(NamedTuple):
    """One answer, or the absence of one.

    code is 0 and error is set when nothing answered (DNS, timeout, reset);
    error is None whenever the server replied at all, whatever it replied.
    """

    body: str
    code: int
    headers: Any
    reason: str = ""
    error: BaseException | None = None

    def problem(self) -> str:
        """A short name for what went wrong; "" when the answer is usable."""
        if self.error is not None:
            return type(self.error).__name__
        if self.code == 200:
            return ""
        return f"HTTP {self.code} {self.reason}".strip()

    def detail(self) -> str:
        """The same, at length: the body, or the exception's own text."""
        if self.error is None:
            return self.body
        return f"{type(self.error).__name__}: {self.error}"


class HttpSession:
    """One source's request layer: headers, timeout, pause, retries, cache.

    The cache is constructed here rather than handed in as a directory, so
    a client never sees a path and DRY_RUN_WRITES_NOTHING keeps holding by
    construction (http_cache.py's own docstring). What is cached is still
    the client's business: names, floors and whether the stored text is the
    raw body or something derived from it are all decided at the call.
    """

    def __init__(self, *, user_agent: str, opener=urllib.request.urlopen,
                 sleep=time.sleep, pause: float = 0.0, timeout: float = 60,
                 encoding: str = "utf-8", accept: str | None = None,
                 cache_dir=None, read_only_cache: bool = False,
                 retry: Retry = Retry()):
        self._opener = opener
        self._sleep = sleep
        self._headers = {"User-Agent": user_agent}
        if accept:
            self._headers["Accept"] = accept
        self.timeout = timeout
        self.encoding = encoding
        self.pause = pause
        self.retry = retry
        self.cache = cache_for(cache_dir, read_only=read_only_cache)
        self.n_requests = 0

    # -- cache -----------------------------------------------------------
    @property
    def n_cache_hits(self) -> int:
        return self.cache.hits if self.cache is not None else 0

    def cached(self, name: str, *, floor: int = 0, limit: int | None = None) -> str | None:
        if self.cache is None or name is None:
            return None
        return self.cache.read(name, floor=floor, limit=limit)

    def store(self, name: str, text: str) -> None:
        if self.cache is not None and name is not None:
            self.cache.write(name, text)

    # -- request ---------------------------------------------------------
    def _once(self, url: str) -> Response:
        request = urllib.request.Request(url, headers=dict(self._headers))
        try:
            with self._opener(request, timeout=self.timeout) as answer:
                return Response(answer.read().decode(self.encoding, "replace"),
                                getattr(answer, "status", 200), answer.headers)
        except urllib.error.HTTPError as err:
            return Response(err.read().decode(self.encoding, "replace"),
                            err.code, err.headers, str(err.reason))
        except Exception as err:  # transport: DNS, timeout, reset
            return Response("", 0, {}, error=err)

    def _backoff(self, answer: Response, delay: float) -> float:
        """How long to wait before the next attempt: the server's word if it
        gave one, our doubling schedule otherwise, neither above the cap.
        """
        asked = None
        if hasattr(answer.headers, "get"):
            asked = answer.headers.get(self.retry.header)
        try:
            wait = float(asked) if asked is not None else delay
        except (TypeError, ValueError):
            wait = delay
        return min(wait, self.retry.ceiling)

    def fetch(self, url: str) -> Response:
        """One request, retried by the policy. Never raises for a bad answer."""
        delay = self.retry.delay
        for attempt in range(1, max(self.retry.tries, 1) + 1):
            answer = self._once(url)
            self.n_requests += 1
            if answer.code in self.retry.codes and attempt < self.retry.tries:
                self._sleep(self._backoff(answer, delay))
                delay = min(delay * 2, self.retry.ceiling)
                continue
            self._sleep(self.pause)
            return answer
        raise AssertionError("unreachable: the last attempt always returns")
