#!/usr/bin/env python3
"""zbMATH Open, in the one role run 85's verdict left it: abstracts.

Not edges, and not identity. Measured in the source survey (§6): the API
carries no citation count and no citer list at all -- only a boolean in
`states` and the narrow `ci:` ("cited in the text of a review"), which
returned zero on all five key works while the same works have 71 citations
in OpenAlex. So zbMATH is consulted only when a seed matched in run 85 has
no abstract in OpenAlex, and what it contributes is written with
`{"abstract_source": "zbmath"}` in evidence -- an abstract of a different
provenance must not be indistinguishable from an OpenAlex one.

There is no `abstract` field either: the closest thing is
`editorial_contributions[]`, i.e. the review or summary somebody wrote
about the work (measured shape: contribution_type in {review, summary}).
That is a text *about* the work rather than the author's own abstract, and
it is stored as such -- honest, and for the frontier filter's purposes
(what is this work about) equally usable.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .http_session import HttpSession

API = "https://api.zbmath.org/v1/document"
USER_AGENT = "ortopol-kb-citations/1.0 (mailto:tooba.mexico@gmail.com)"
PAUSE = 0.35
# summary first: an author summary describes the work, a review describes
# what a reviewer thought of it. Both beat nothing, the order is a
# preference, not a filter.
CONTRIBUTION_ORDER = ("summary", "review")


class ZbmathUnavailable(RuntimeError):
    """The request failed, so we do NOT know whether zbMATH has the record.

    Distinct from a None return, which means zbMATH answered and does not
    have it. Collapsing the two is what the crawl's whole journal discipline
    exists to prevent: an absence must be a recorded decision, never
    indistinguishable from a transient 429 or a dropped connection. The
    caller records the failure (pg_load_citations.zbmath_abstracts writes an
    action='error' journal row) instead of storing a permanent "no abstract"
    for a work whose abstract was never asked for successfully.
    """


class ZbmathClient:
    """Answers are cached on disk, keyed by the zbMATH document id.

    The same discipline MathnetClient and OpenAlexClient follow, and for the
    same measured reason: these abstracts do not change between runs, the
    API returns 429s under load, and the fallback runs on the startup path
    of every non-offline invocation -- one sequential request per matched
    seed with a pause between them.

    What is cached is what zbMATH ANSWERED, `null` included: "we asked and
    it does not have this one" is knowledge, and re-asking for it buys
    nothing. A failure is never cached -- ZbmathUnavailable means we did not
    learn anything, and a cached blank would turn one 429 into a permanent
    verdict, which is the distinction this whole class is built around.
    """

    def __init__(self, *, opener=urllib.request.urlopen, sleep=time.sleep,
                 pause=PAUSE, cache=None):
        # No retry policy: what this client owes the caller is the
        # distinction between an answer and a failure, and a silent second
        # attempt at a 429 does not change which of the two arrived.
        self._session = HttpSession(
            user_agent=USER_AGENT, accept="application/json", opener=opener,
            sleep=sleep, pause=pause, cache=cache)
        self.pause = pause
        # Mirrors MathnetClient.failures in the same call chain: counted and
        # named, never swallowed.
        self.failures: list[str] = []

    @property
    def n_requests(self) -> int:
        return self._session.n_requests

    @property
    def n_cache_hits(self) -> int:
        return self._session.n_cache_hits

    def _cached(self, zbmath_id: str) -> str:
        # The id is a zbMATH document number ('1234.56789'), not a path: the
        # separator would otherwise make a directory out of it.
        return zbmath_id.replace("/", "_") + ".json"

    def _note(self, zbmath_id: str, what: str) -> str:
        """Names what went wrong in .failures and returns the same text.

        The list is the channel this client reports through, and everything
        that reaches it is "we did not learn anything" -- whether the
        request failed or the answer we had kept turned out unreadable.
        """
        self.failures.append(f"{zbmath_id}: {what}")
        return f"{zbmath_id}: {what}"

    def _failed(self, zbmath_id: str, what: str) -> ZbmathUnavailable:
        return ZbmathUnavailable(self._note(zbmath_id, what))

    def document(self, zbmath_id: str) -> dict | None:
        """One record, or None when zbMATH answered and does not have it.

        404, and a 200 whose `result` is empty, are legitimate answers.
        Every other outcome -- any other HTTP status, a network error, a body
        that is not JSON -- raises ZbmathUnavailable: we did not learn
        anything about this work, and saying "no abstract" would be a claim
        the request never supported.
        """
        if self._session.cache is None:
            return self._fetch(zbmath_id)
        name = self._cached(zbmath_id)
        hit = self._session.cached(name)
        if hit is not None:
            try:
                return json.loads(hit)
            except json.JSONDecodeError as err:
                # An entry cut short by a killed process is non-empty, so
                # the cache serves it as a hit. A bare JSONDecodeError here
                # is neither of the two outcomes this client keeps apart:
                # it is not in .failures and no caller catches it. The entry
                # is disposable -- named, then asked for again, and the
                # answer below overwrites it.
                self._note(zbmath_id, f"запись кэша нечитаема ({err}), перезапрошено")
        record = self._fetch(zbmath_id)
        self._session.store(name, json.dumps(record, ensure_ascii=False))
        return record

    def _fetch(self, zbmath_id: str) -> dict | None:
        answer = self._session.fetch(f"{API}/{zbmath_id}")
        if answer.error is not None:
            raise self._failed(zbmath_id, answer.detail()) from answer.error
        if answer.code == 404:
            return None
        if answer.code != 200:
            raise self._failed(zbmath_id, answer.problem())
        try:
            body = json.loads(answer.body)
        except json.JSONDecodeError as err:
            raise self._failed(zbmath_id, f"ответ не JSON: {err}") from err
        result = body.get("result")
        if isinstance(result, list):
            return result[0] if result else None
        return result if isinstance(result, dict) else None


def abstract_of(record: dict | None) -> tuple[str | None, list[str]]:
    """(text, contribution types) from a zbMATH record.

    Returns (None, []) rather than an empty string when the record carries
    no editorial contribution -- "we looked and there was nothing" and "we
    stored a blank" must not read the same downstream.
    """
    if not record:
        return None, []
    contributions = [c for c in (record.get("editorial_contributions") or []) if c.get("text")]
    if not contributions:
        return None, []
    contributions.sort(
        key=lambda c: CONTRIBUTION_ORDER.index(c.get("contribution_type"))
        if c.get("contribution_type") in CONTRIBUTION_ORDER
        else len(CONTRIBUTION_ORDER)
    )
    texts = [c["text"].strip() for c in contributions if c["text"].strip()]
    if not texts:
        return None, []
    types = [c.get("contribution_type") or "unknown" for c in contributions]
    return "\n\n".join(texts), types
