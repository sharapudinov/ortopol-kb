"""The urlopen stand-ins the crawl's HTTP tests script their answers with.

Beside _pathfix.py rather than inside one test module: more than one module
now drives citations/http_session.py's opener seam (the OpenAlex quota
policy, the three HTTP surfaces' failure branches), and a second copy of a
stub is a second thing that can stop resembling a socket.
"""
from __future__ import annotations

import io
import urllib.error


class Response:
    """urlopen stand-in: body, headers and a status, usable as a context
    manager. The body is bytes so the encoding is the client's problem, the
    way a socket makes it."""

    def __init__(self, body: bytes, headers: dict | None = None, status: int = 200):
        self._body, self.headers, self.status = body, dict(headers or {}), status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class Sequence:
    """Answers a scripted list of responses/exceptions, one per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, request, timeout=None):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer


def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.openalex.org/works", code, "nope",
                                  dict(headers or {}), io.BytesIO(b'{"error": "nope"}'))
