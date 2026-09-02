"""The request layer all three crawl clients share (citations/http_session).

What lives here is the MECHANISM: the retry schedule, the polite pause, the
request counter, the cache seam and the classification of "nothing
answered" against "the server said no". Each client's POLICY -- what a 404
means, what raises, what is merely counted -- stays in its own tests, which
is the split the module exists to make possible.

Every opener is a stub and every sleep is captured, so the schedule is
asserted rather than waited out.
"""
from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

import _pathfix  # noqa: F401

from citations.http_cache import DiskCache, ReadOnlyCache
from citations.http_session import HttpSession, Response, Retry


class _Answer:
    """urlopen stand-in: body, headers and a status, usable as a context
    manager. The body is bytes so the encoding is the session's problem,
    the way a socket makes it."""

    def __init__(self, body: bytes, headers: dict | None = None, status: int = 200):
        self._body, self.headers, self.status = body, dict(headers or {}), status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Sequence:
    """Answers a scripted list of responses/exceptions, one per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer


def _http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test/x", code, "nope",
                                  dict(headers or {}), io.BytesIO(b'{"error": "nope"}'))


def _session(answers, **kwargs) -> tuple[HttpSession, _Sequence, list]:
    slept: list[float] = []
    opener = _Sequence(answers)
    kwargs.setdefault("user_agent", "ortopol-test/1.0")
    return (HttpSession(opener=opener, sleep=slept.append, **kwargs), opener, slept)


RETRY = Retry(tries=5, codes=(429, 500, 502, 503, 504, 0))
OK = b'{"ok": true}'


class RequestShapeTests(unittest.TestCase):
    def test_the_user_agent_and_accept_travel_on_every_request(self):
        session, opener, _slept = _session([_Answer(OK)], accept="application/json")
        session.fetch("https://example.test/x")
        request, timeout = opener.requests[0]
        self.assertEqual(request.get_header("User-agent"), "ortopol-test/1.0")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(timeout, 60)

    def test_the_body_is_decoded_with_the_sessions_encoding(self):
        session, _opener, _slept = _session([_Answer("Шарапудинов".encode("windows-1251"))],
                                            encoding="windows-1251")
        self.assertEqual(session.fetch("https://example.test/x").body, "Шарапудинов")

    def test_the_pause_is_taken_after_the_answer_that_is_returned(self):
        session, _opener, slept = _session([_Answer(OK)], pause=0.35)
        session.fetch("https://example.test/x")
        self.assertEqual(slept, [0.35])

    def test_every_attempt_counts_as_a_request(self):
        session, _opener, _slept = _session([_http_error(429), _Answer(OK)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(session.n_requests, 2, "повтор не посчитан как запрос квоты")


class AnswerClassificationTests(unittest.TestCase):
    """"The server said no" and "nothing answered" are different facts, and
    every client's policy is built on telling them apart.
    """

    def test_a_status_answer_carries_its_code_and_no_error(self):
        session, _opener, _slept = _session([_http_error(404)])
        answer = session.fetch("https://example.test/x")
        self.assertEqual(answer.code, 404)
        self.assertIsNone(answer.error)
        self.assertEqual(answer.problem(), "HTTP 404 nope")
        self.assertIn("error", answer.body)

    def test_a_transport_failure_is_code_zero_and_keeps_the_exception(self):
        session, _opener, _slept = _session([TimeoutError("read timed out")])
        answer = session.fetch("https://example.test/x")
        self.assertEqual(answer.code, 0)
        self.assertIsInstance(answer.error, TimeoutError)
        self.assertEqual(answer.problem(), "TimeoutError")
        self.assertEqual(answer.detail(), "TimeoutError: read timed out")

    def test_a_usable_answer_reports_no_problem(self):
        session, _opener, _slept = _session([_Answer(OK)])
        self.assertEqual(session.fetch("https://example.test/x").problem(), "")

    def test_a_fetch_never_raises_for_a_bad_answer(self):
        session, _opener, _slept = _session([_http_error(503)])
        self.assertEqual(session.fetch("https://example.test/x").code, 503)


class RetryScheduleTests(unittest.TestCase):
    """429 and 5xx are the answers a shared public API gives under load, so
    they are transient by definition -- but only for a client that asked
    for a retry policy. The default is one attempt.
    """

    def test_no_policy_means_one_attempt(self):
        session, opener, _slept = _session([_http_error(503)])
        session.fetch("https://example.test/x")
        self.assertEqual(opener.calls, 1)

    def test_a_429_then_a_200_is_one_successful_answer(self):
        session, opener, slept = _session([_http_error(429), _Answer(OK)], retry=RETRY)
        self.assertEqual(session.fetch("https://example.test/x").code, 200)
        self.assertEqual(opener.calls, 2)
        self.assertIn(2.0, slept, f"первая пауза не выдержана: {slept}")

    def test_the_servers_retry_after_wins_over_our_backoff(self):
        session, _opener, slept = _session(
            [_http_error(429, {"retry-after": "7"}), _Answer(OK)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(slept[0], 7.0, f"пауза не по заголовку сервера: {slept}")

    def test_an_absurd_retry_after_is_capped(self):
        session, _opener, slept = _session(
            [_http_error(429, {"retry-after": "86400"}), _Answer(OK)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(slept[0], 120.0)

    def test_an_unparseable_retry_after_falls_back_to_our_schedule(self):
        session, _opener, slept = _session(
            [_http_error(429, {"retry-after": "soon"}), _Answer(OK)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(slept[0], 2.0)

    def test_the_backoff_doubles_across_failures(self):
        session, _opener, slept = _session(
            [_http_error(500), _http_error(502), _Answer(OK)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(slept[:2], [2.0, 4.0])

    def test_a_transport_error_is_retried_when_code_zero_is_in_the_policy(self):
        session, opener, _slept = _session([TimeoutError("read timed out"), _Answer(OK)],
                                           retry=RETRY)
        self.assertEqual(session.fetch("https://example.test/x").code, 200)
        self.assertEqual(opener.calls, 2)

    def test_exhausting_the_tries_returns_the_last_answer(self):
        session, opener, _slept = _session([_http_error(503)],
                                           retry=Retry(tries=3, codes=(503,)))
        self.assertEqual(session.fetch("https://example.test/x").code, 503)
        self.assertEqual(opener.calls, 3)

    def test_a_code_outside_the_policy_is_not_retried(self):
        session, opener, _slept = _session([_http_error(404)], retry=RETRY)
        session.fetch("https://example.test/x")
        self.assertEqual(opener.calls, 1)


class CacheSeamTests(unittest.TestCase):
    """The session holds the Cache object, so no client is handed a path --
    DRY_RUN_WRITES_NOTHING stays a property of what was constructed.
    """

    def test_no_cache_directory_means_no_hits_and_no_writes(self):
        session, _opener, _slept = _session([_Answer(OK)])
        self.assertIsNone(session.cache)
        self.assertIsNone(session.cached("anything"))
        session.store("anything", "text")  # must not raise
        self.assertEqual(session.n_cache_hits, 0)

    def test_a_stored_entry_reads_back_and_counts_as_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _opener, _slept = _session([_Answer(OK)], cache=DiskCache(Path(tmp)))
            session.store("page", "тело")
            self.assertEqual(session.cached("page"), "тело")
        self.assertEqual(session.n_cache_hits, 1)

    def test_a_read_only_session_serves_hits_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "cache"
            session, _opener, _slept = _session([_Answer(OK)],
                                                cache=ReadOnlyCache(directory))
            session.store("page", "тело")
            self.assertFalse(directory.exists(), "--dry-run создал каталог кэша")
            directory.mkdir()
            (directory / "page").write_text("тело", encoding="utf-8")
            self.assertEqual(session.cached("page"), "тело")

    def test_a_body_at_or_under_the_floor_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _opener, _slept = _session([_Answer(OK)], cache=DiskCache(Path(tmp)))
            session.store("page", "short")
            self.assertIsNone(session.cached("page", floor=2000))
            self.assertEqual(session.n_cache_hits, 0)


class ResponseTests(unittest.TestCase):
    def test_a_reasonless_status_still_names_itself(self):
        self.assertEqual(Response("", 503, {}).problem(), "HTTP 503")


if __name__ == "__main__":
    unittest.main()
