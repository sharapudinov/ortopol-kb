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
import urllib.error
import urllib.request

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
    def __init__(self, *, opener=urllib.request.urlopen, sleep=time.sleep, pause=PAUSE):
        self._opener = opener
        self._sleep = sleep
        self.pause = pause
        self.n_requests = 0
        # Mirrors MathnetClient.failures in the same call chain: counted and
        # named, never swallowed.
        self.failures: list[str] = []

    def _failed(self, zbmath_id: str, what: str) -> ZbmathUnavailable:
        self.failures.append(f"{zbmath_id}: {what}")
        return ZbmathUnavailable(f"{zbmath_id}: {what}")

    def document(self, zbmath_id: str) -> dict | None:
        """One record, or None when zbMATH answered and does not have it.

        404, and a 200 whose `result` is empty, are legitimate answers.
        Every other outcome -- any other HTTP status, a network error, a body
        that is not JSON -- raises ZbmathUnavailable: we did not learn
        anything about this work, and saying "no abstract" would be a claim
        the request never supported.
        """
        url = f"{API}/{zbmath_id}"
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with self._opener(request, timeout=60) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            self.n_requests += 1
            if err.code == 404:
                return None
            raise self._failed(zbmath_id, f"HTTP {err.code} {err.reason}") from err
        except Exception as err:
            self.n_requests += 1
            raise self._failed(zbmath_id, f"{type(err).__name__}: {err}") from err
        self.n_requests += 1
        self._sleep(self.pause)
        try:
            body = json.loads(raw)
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
