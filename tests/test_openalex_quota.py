"""What the OpenAlex client makes of a FINAL answer.

Not how the answer was fetched: the retry SCHEDULE is the shared session's
and is tested in test_http_session.py. Here: giving up has to be a named
failure rather than an empty result the caller mistakes for "nothing
found", and it has to be the RIGHT named failure -- pg_load_citations.py
journals QuotaExhausted and exits 2 ("wait the window out") where it exits
3 on OpenAlexError ("the source failed, go and look").

Those two used to be one. http_session.fetch() retries RETRY_CODES
internally and then RETURNS the final response, so a 429 on every attempt
-- a genuinely spent budget, the exact case this module tracks rate headers
for -- arrived at get_json() as a non-200 and was raised as a generic
source fault. _honour_quota() could only fire on a 200 that happened to
report a low remainder.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from _http_fixtures import Response, Sequence, http_error

from citations.openalex_client import (
    RATE_LIMITED,
    RETRY_CODES,
    OpenAlexClient,
    OpenAlexError,
    QuotaExhausted,
)


class OpenAlexPolicyTests(unittest.TestCase):
    """Every terminal answer, and which of the two failures it becomes."""

    OK = b'{"results": [], "meta": {}}'

    def _client(self, answers, tries=5):
        slept = []
        opener = Sequence(answers)
        client = OpenAlexClient(opener=opener, sleep=slept.append, pause=0.0, tries=tries)
        return client, opener

    def test_a_429_then_a_200_is_one_successful_call(self):
        client, opener = self._client([http_error(429), Response(self.OK)])
        self.assertEqual(client.get_json("https://api.openalex.org/works?x=1"),
                         {"results": [], "meta": {}})
        self.assertEqual(opener.calls, 2)
        self.assertEqual(client.n_requests, 2, "повтор не посчитан как запрос квоты")

    def test_the_retry_set_is_the_transient_answers_and_nothing_else(self):
        self.assertEqual(set(RETRY_CODES), {429, 500, 502, 503, 504, 0})
        self.assertNotIn(404, RETRY_CODES)

    def test_exhausting_the_tries_raises_and_says_how_many(self):
        client, opener = self._client([http_error(503)], tries=3)
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        message = str(ctx.exception)
        self.assertIn("503", message)
        self.assertIn("api.openalex.org", message)
        self.assertEqual(opener.calls, 3)

    def test_a_non_retryable_code_fails_at_once(self):
        client, opener = self._client([http_error(404)])
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("404", str(ctx.exception))
        self.assertEqual(opener.calls, 1)

    def test_a_transport_failure_that_never_recovers_is_named_too(self):
        client, _opener = self._client([TimeoutError("read timed out")], tries=2)
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("TimeoutError", str(ctx.exception))

    def test_a_200_that_is_not_json_is_not_a_retry_either(self):
        client, _opener = self._client([Response(b"<html>502</html>")])
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("не JSON", str(ctx.exception))

    def test_the_quota_headers_are_read_off_the_answer(self):
        client, _opener = self._client(
            [Response(self.OK, {"x-ratelimit-remaining": "289",
                                 "x-ratelimit-reset": "83942"})])
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(client.last_rate["x-ratelimit-remaining"], "289")

    def test_a_budget_under_the_floor_with_a_distant_reset_stops_the_crawl(self):
        client, _opener = self._client(
            [Response(self.OK, {"x-ratelimit-remaining": "3",
                                 "x-ratelimit-reset": "83942"})])
        with self.assertRaises(QuotaExhausted):
            client.get_json("https://api.openalex.org/works?x=1")

    def test_a_429_on_every_attempt_is_the_quota_not_a_source_fault(self):
        """The case the QuotaExhausted contract exists for, and the one it
        could not reach: retries exhausted against a rate limit is the
        server saying the shared budget is spent.
        """
        client, opener = self._client(
            [http_error(RATE_LIMITED, {"x-ratelimit-reset": "83942"})], tries=3)
        with self.assertRaises(QuotaExhausted) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(opener.calls, 3)
        message = str(ctx.exception)
        self.assertIn(str(RATE_LIMITED), message)
        self.assertIn("83942", message)

    def test_the_reset_window_is_read_off_the_header_not_guessed(self):
        """A 429 with no usable reset header still stops the crawl, and says
        zero rather than inventing a window to sleep out."""
        client, _opener = self._client([http_error(RATE_LIMITED)], tries=2)
        with self.assertRaises(QuotaExhausted) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("0 с", str(ctx.exception))

    def test_a_500_on_every_attempt_is_still_a_source_fault(self):
        """The complement: only the rate limit is an answer about the
        budget, so the other retryable codes must keep the exit code that
        sends a human to look at the source.
        """
        for code in (500, 502, 503, 504):
            client, _opener = self._client([http_error(code)], tries=2)
            with self.subTest(code=code):
                with self.assertRaises(OpenAlexError) as ctx:
                    client.get_json("https://api.openalex.org/works?x=1")
                self.assertNotIsInstance(ctx.exception, QuotaExhausted)
                self.assertIn(str(code), str(ctx.exception))

    def test_a_429_that_recovers_is_neither_failure(self):
        """Already covered above as a successful call; asserted here as the
        boundary: only the FINAL answer decides, so the retry that worked
        must not leave a quota verdict behind.
        """
        client, _opener = self._client([http_error(RATE_LIMITED), Response(self.OK)])
        self.assertEqual(client.get_json("https://api.openalex.org/works?x=1"),
                         {"results": [], "meta": {}})


if __name__ == "__main__":
    unittest.main()
