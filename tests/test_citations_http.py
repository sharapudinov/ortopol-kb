"""The three HTTP surfaces of the crawl, at their failure branches.

Every opener here is a stub and every sleep is captured, so the retry
schedule is asserted rather than waited out. What these cover is what a
single-happy-response test cannot: the OpenAlex retry loop and its terminal
give-up, Math-Net's cache/failure bookkeeping, and the two shapes of a
malformed ollama answer -- the branches that only run on a bad day, which
is exactly when nobody is reading the code.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from unittest import mock
import urllib.error
from pathlib import Path

import _pathfix  # noqa: F401

from citations import frontier
from citations.mathnet import MathnetClient, parse_titles
from citations.openalex_client import (
    OpenAlexClient,
    OpenAlexError,
    page_index,
    sidecar_path,
)

PAGE = ('<html><head><title>И. И. Шарапудинов, “Русское название”, Матем. сб., '
        '180:9 (1989), 1–10; I. I. Sharapudinov, “English title”, '
        'Math. USSR-Sb., 68:1 (1991), 1–10</title></head><body>'
        + "x" * 2500 + "</body></html>")


class _Response:
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


class _Sequence:
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


def _http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.openalex.org/works", code, "nope",
                                  dict(headers or {}), io.BytesIO(b'{"error": "nope"}'))


class OpenAlexRetryTests(unittest.TestCase):
    """429 and 5xx are the answers a shared public API gives under load, so
    they are transient by definition -- and giving up has to be a named
    failure, not an empty result the caller mistakes for "nothing found".
    """

    OK = b'{"results": [], "meta": {}}'

    def _client(self, answers, tries=5):
        slept = []
        client = OpenAlexClient(opener=_Sequence(answers), sleep=slept.append,
                                pause=0.0, tries=tries)
        return client, slept

    def test_a_429_then_a_200_is_one_successful_call(self):
        client, slept = self._client([_http_error(429), _Response(self.OK)])
        self.assertEqual(client.get_json("https://api.openalex.org/works?x=1"), {"results": [],
                                                                                "meta": {}})
        self.assertEqual(client._opener.calls, 2)
        self.assertEqual(client.n_requests, 2, "повтор не посчитан как запрос квоты")
        self.assertIn(2.0, slept, f"первая пауза не выдержана: {slept}")

    def test_the_servers_retry_after_wins_over_our_backoff(self):
        client, slept = self._client([_http_error(429, {"retry-after": "7"}),
                                      _Response(self.OK)])
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(slept[0], 7.0, f"пауза не по заголовку сервера: {slept}")

    def test_an_absurd_retry_after_is_capped_at_two_minutes(self):
        client, slept = self._client([_http_error(429, {"retry-after": "86400"}),
                                      _Response(self.OK)])
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(slept[0], 120.0)

    def test_the_backoff_doubles_across_failures(self):
        client, slept = self._client([_http_error(500), _http_error(502),
                                      _Response(self.OK)])
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(slept[:2], [2.0, 4.0])

    def test_a_transport_error_is_retried_like_a_5xx(self):
        client, _slept = self._client([TimeoutError("read timed out"),
                                       _Response(self.OK)])
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertEqual(client._opener.calls, 2)

    def test_exhausting_the_tries_raises_and_says_how_many(self):
        client, _slept = self._client([_http_error(503)], tries=3)
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        message = str(ctx.exception)
        self.assertIn("3", message)
        self.assertIn("api.openalex.org", message)
        self.assertEqual(client._opener.calls, 3)

    def test_a_non_retryable_code_fails_at_once(self):
        client, _slept = self._client([_http_error(404)])
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("404", str(ctx.exception))
        self.assertEqual(client._opener.calls, 1)

    def test_a_200_that_is_not_json_is_not_a_retry_either(self):
        client, _slept = self._client([_Response(b"<html>502</html>")])
        with self.assertRaises(OpenAlexError) as ctx:
            client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn("не JSON", str(ctx.exception))


class OpenAlexSidecarTests(unittest.TestCase):
    """A cached page gets a two-field index beside it, so a reader that
    needs the batch and its promised count does not decode 217 MiB of works.
    """

    BODY = json.dumps({
        "meta": {"count": 18904, "x_query": {
            "oql": "works where it cites (W1 or W2)",
            "url": "/works?filter=referenced_works:W1|W2&per_page=200&cursor=AAA"}},
        "results": [],
    }).encode()

    def _cached(self, tmp: Path) -> Path:
        client = OpenAlexClient(opener=_Sequence([_Response(self.BODY)]),
                                sleep=lambda _s: None, pause=0.0, cache_dir=tmp)
        client.get_json("https://api.openalex.org/works?filter=cites:W1|W2")
        return next(p for p in tmp.glob("*.json") if not p.name.endswith(".meta.json"))

    def test_the_page_is_cached_with_its_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._cached(Path(tmp))
            note = json.loads(sidecar_path(page).read_text(encoding="utf-8"))
        self.assertEqual(note, {"filter": "referenced_works:W1|W2",
                                "oql": "works where it cites (W1 or W2)",
                                "count": 18904})

    def test_a_read_only_cache_writes_neither_page_nor_sidecar(self):
        """--dry-run's third channel: the response cache is in the data
        tree, so a dry run must leave it exactly as it found it -- the
        sidecar included, since it is written beside every cached page.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "openalex"
            client = OpenAlexClient(opener=_Sequence([_Response(self.BODY)]),
                                    sleep=lambda _s: None, pause=0.0,
                                    cache_dir=cache, read_only_cache=True)
            client.get_json("https://api.openalex.org/works?filter=cites:W1|W2")
            self.assertFalse(cache.exists(), "--dry-run создал каталог кэша")

    def test_a_read_only_cache_still_serves_a_hit(self):
        """Read-only, not off: a dry run must cost no more quota than a
        real one, which is the whole reason the cache is kept in the tree.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            self._cached(cache)
            client = OpenAlexClient(opener=_Sequence([]), sleep=lambda _s: None,
                                    pause=0.0, cache_dir=cache, read_only_cache=True)
            body = client.get_json("https://api.openalex.org/works?filter=cites:W1|W2")
        self.assertEqual(body["meta"]["count"], 18904)
        self.assertEqual(client.n_cache_hits, 1)

    def test_the_batch_is_named_by_its_filter_not_by_the_cursor(self):
        """Two pages of one batch differ only in the cursor; keyed on the
        url they counted as two batches -- 3 392 521 instead of 51 652.
        """
        first = page_index(json.loads(self.BODY))
        second = page_index(json.loads(self.BODY.replace(b"cursor=AAA", b"cursor=BBB")))
        self.assertEqual(first["filter"], second["filter"])

    def test_a_page_with_no_filter_still_gets_an_index(self):
        note = page_index({"meta": {"count": 3, "x_query": {"oql": "works", "url": "/works"}}})
        self.assertEqual(note, {"filter": "works", "oql": "works", "count": 3})


class MathnetClientTests(unittest.TestCase):
    """The identity anchor's two disciplines: a retry must not re-fetch the
    pages that already worked, and a page that did not arrive -- or arrived
    without the citations -- must be COUNTED, because a silent gap here
    weakens the twin index invisibly (it did, for 2019_rm9846).
    """

    def _client(self, opener, cache: Path | None = None,
                read_only: bool = False) -> MathnetClient:
        return MathnetClient(opener=opener, sleep=lambda _s: None, cache_dir=cache,
                             read_only_cache=read_only)

    def test_the_page_gives_both_titles_and_both_years(self):
        titles, years = parse_titles(PAGE)
        self.assertEqual(titles, ["Русское название", "English title"])
        self.assertEqual(years, [1989, 1991])

    def test_a_cached_page_costs_no_second_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            opener = _Sequence([_Response(PAGE.encode("windows-1251"))])
            client = self._client(opener, Path(tmp))
            first = client.titles("mzm8442")
            second = client.titles("mzm8442")
        self.assertEqual(first, second)
        self.assertEqual(opener.calls, 1, "кэш не спас от второго запроса")
        self.assertEqual(client.n_requests, 1)
        self.assertEqual(client.n_cache_hits, 1)
        self.assertEqual(client.failures, [])

    def test_a_transport_failure_is_counted_and_named(self):
        client = self._client(_Sequence([TimeoutError("read timed out")]))
        self.assertEqual(client.titles("mzm8442"), ([], []))
        self.assertEqual(client.n_requests, 1)
        self.assertEqual(len(client.failures), 1)
        self.assertIn("mzm8442", client.failures[0])
        self.assertIn("TimeoutError", client.failures[0])

    def test_a_page_without_citations_is_a_failure_not_an_absence(self):
        client = self._client(_Sequence([_Response(b"<html><body>redirected</body></html>")]))
        self.assertEqual(client.titles("mzm8442"), ([], []))
        self.assertEqual(len(client.failures), 1)
        self.assertIn("без цитат", client.failures[0])

    def test_a_titleless_page_is_not_cached(self):
        """Nothing was learned, so a retry must be free to ask again."""
        with tempfile.TemporaryDirectory() as tmp:
            opener = _Sequence([_Response(b"<html><body>redirected</body></html>"),
                                _Response(PAGE.encode("windows-1251"))])
            client = self._client(opener, Path(tmp))
            client.titles("mzm8442")
            titles, _years = client.titles("mzm8442")
        self.assertEqual(opener.calls, 2)
        self.assertEqual(titles, ["Русское название", "English title"])

    def test_a_read_only_cache_serves_hits_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "mathnet"
            opener = _Sequence([_Response(PAGE.encode("windows-1251")),
                                _Response(PAGE.encode("windows-1251"))])
            client = self._client(opener, cache, read_only=True)
            client.titles("mzm8442")
            self.assertFalse(cache.exists(), "--dry-run создал каталог кэша")
            cache.mkdir()
            (cache / "mzm8442.html").write_text(PAGE, encoding="utf-8")
            titles, _years = client.titles("mzm8442")
        self.assertEqual(titles, ["Русское название", "English title"])
        self.assertEqual(opener.calls, 1)


class EmbedTextsTests(unittest.TestCase):
    """A wrong answer from the embedder is not an exception by itself: it is
    a list of the wrong length, or vectors of the wrong width, and either
    would land the crawl's cosines somewhere arbitrary instead of failing.
    """

    MODEL, DIMS = "bge-m3", 1024

    def _opener(self, payload):
        def opener(_request, timeout=None):
            return _Response(json.dumps(payload).encode())
        return opener

    def test_vectors_come_back_in_input_order_across_batches(self):
        def opener(request, timeout=None):
            chunk = json.loads(request.data)["input"]
            return _Response(json.dumps(
                {"embeddings": [[float(len(t))] * self.DIMS for t in chunk]}).encode())

        texts = [f"{'x' * n}" for n in range(1, 20)]
        out = frontier.embed_texts(texts, self.MODEL, self.DIMS, opener=opener, batch=8)
        self.assertEqual([v[0] for v in out], [float(n) for n in range(1, 20)])

    def test_too_few_vectors_is_a_runtime_error(self):
        opener = self._opener({"embeddings": [[0.0] * self.DIMS]})
        with self.assertRaises(RuntimeError) as ctx:
            frontier.embed_texts(["a", "b"], self.MODEL, self.DIMS, opener=opener)
        self.assertIn("1 векторов на 2 текстов", str(ctx.exception))

    def test_a_wrong_dimension_is_a_runtime_error(self):
        opener = self._opener({"embeddings": [[0.0] * 768]})
        with self.assertRaises(RuntimeError) as ctx:
            frontier.embed_texts(["a"], self.MODEL, self.DIMS, opener=opener)
        self.assertIn("ожидалось 1024 измерений, пришло 768", str(ctx.exception))

    def test_the_model_and_the_texts_are_what_is_asked_for(self):
        sent = []

        def opener(request, timeout=None):
            sent.append(json.loads(request.data))
            return _Response(json.dumps({"embeddings": [[0.0] * self.DIMS]}).encode())

        frontier.embed_texts(["Чебышёв"], self.MODEL, self.DIMS, opener=opener)
        self.assertEqual(sent, [{"model": self.MODEL, "input": ["Чебышёв"]}])


class OneEmbeddingSeamTests(unittest.TestCase):
    """frontier.py is the relevance filter, not a second HTTP client: the
    request, the batching and both checks are pg_search's, and so is the
    read of which model the corpus was embedded with.
    """

    def test_embed_texts_delegates_to_the_shared_batch_entry_point(self):
        with mock.patch.object(frontier, "embed_batch",
                                return_value=[[0.0]]) as batch_mock:
            frontier.embed_texts(["a"], "bge-m3", 1024, batch=8)
        self.assertEqual(batch_mock.call_args.args[:3], ("bge-m3", 1024, ["a"]))
        self.assertEqual(batch_mock.call_args.kwargs["batch"], 8)

    def test_the_package_reads_the_model_from_nowhere_of_its_own(self):
        package = Path(frontier.__file__).parent
        for module in sorted(package.glob("*.py")):
            source = module.read_text(encoding="utf-8")
            self.assertNotIn("FROM corpus.embedding_model", source, module.name)


if __name__ == "__main__":
    unittest.main()
