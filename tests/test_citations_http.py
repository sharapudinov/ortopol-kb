"""The three HTTP surfaces of the crawl, at their failure branches.

Every opener here is a stub and every sleep is captured, so the retry
schedule is asserted rather than waited out. What these cover is what a
single-happy-response test cannot: the OpenAlex page cache and its sidecar,
Math-Net's cache/failure bookkeeping, and the two shapes of a malformed
ollama answer -- the branches that only run on a bad day, which is exactly
when nobody is reading the code.

How the OpenAlex client CLASSIFIES a final answer -- source fault against
spent quota -- is one question of its own and lives in
test_openalex_quota.py (module size).
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import _pathfix  # noqa: F401
from _http_fixtures import Response as _Response, Sequence as _Sequence

from citations import frontier, http_cache, openalex_client
from citations.mathnet import MathnetClient, parse_titles
from citations.openalex_client import OpenAlexClient
from citations.openalex_records import (
    SIDECAR_SUFFIX,
    direction_of,
    note_direction,
    page_index,
    sidecar_name,
)

PAGE = ('<html><head><title>И. И. Шарапудинов, “Русское название”, Матем. сб., '
        '180:9 (1989), 1–10; I. I. Sharapudinov, “English title”, '
        'Math. USSR-Sb., 68:1 (1991), 1–10</title></head><body>'
        + "x" * 2500 + "</body></html>")


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
                                sleep=lambda _s: None, pause=0.0,
                                cache=http_cache.DiskCache(tmp))
        client.get_json("https://api.openalex.org/works?filter=cites:W1|W2")
        return next(p for p in tmp.glob("*.json") if not p.name.endswith(".meta.json"))

    def test_the_page_is_cached_with_its_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._cached(Path(tmp))
            sidecar = page.with_name(sidecar_name(page.name))
            note = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(note, {"filter": "referenced_works:W1|W2",
                                "direction": "cites",
                                "oql": "works where it cites (W1 or W2)",
                                "count": 18904})

    def test_a_sidecar_that_cannot_be_written_costs_neither_page_nor_answer(self):
        """The swallow is deliberate -- the cache is disposable scratch and
        a directory that will not take the index is the caller's problem,
        not a failed request's -- but nothing had ever made it fire, so
        "the primary cached page survives it" was an intention rather than
        a fact.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = http_cache.DiskCache(Path(tmp))
            written = cache.write

            def refuse_the_sidecar(name, text):
                if name.endswith(SIDECAR_SUFFIX):
                    raise OSError("read-only file system")
                written(name, text)

            cache.write = refuse_the_sidecar
            client = OpenAlexClient(opener=_Sequence([_Response(self.BODY)]),
                                    sleep=lambda _s: None, pause=0.0, cache=cache)
            body = client.get_json("https://api.openalex.org/works?filter=cites:W1|W2")
            pages = [p for p in Path(tmp).glob("*.json")
                     if not p.name.endswith(SIDECAR_SUFFIX)]
            self.assertEqual(len(pages), 1, "страница кэша не записалась")
            self.assertEqual(json.loads(pages[0].read_text(encoding="utf-8")),
                             json.loads(self.BODY))
        self.assertEqual(body["meta"]["count"], 18904)

    def test_a_read_only_cache_writes_neither_page_nor_sidecar(self):
        """--dry-run's third channel: the response cache is in the data
        tree, so a dry run must leave it exactly as it found it -- the
        sidecar included, since it is written beside every cached page.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "openalex"
            client = OpenAlexClient(opener=_Sequence([_Response(self.BODY)]),
                                    sleep=lambda _s: None, pause=0.0,
                                    cache=http_cache.ReadOnlyCache(cache))
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
                                    pause=0.0, cache=http_cache.ReadOnlyCache(cache))
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
        self.assertEqual(note, {"filter": "works", "direction": None,
                                "oql": "works", "count": 3})

    def test_the_direction_is_read_off_the_filter_in_both_spellings(self):
        """OpenAlex echoes the request's filter back normalised: the crawl
        asks `cites:`/`openalex_id:` and x_query.url answers
        `referenced_works:`/`ids.openalex:`. Both name the same direction,
        and a filter naming neither belongs to no direction at all.
        """
        self.assertEqual(direction_of("referenced_works:W1|W2"), "cites")
        self.assertEqual(direction_of("cites:W1|W2"), "cites")
        self.assertEqual(direction_of("ids.openalex:W1"), "openalex_id")
        self.assertEqual(direction_of("openalex_id:W1"), "openalex_id")
        self.assertIsNone(direction_of("authorships.author.id:A1"))
        self.assertIsNone(direction_of(""))

    def test_an_index_written_before_the_direction_was_a_field_still_has_one(self):
        """Sidecars are durable and are not rewritten. `filter` is what the
        direction was always derived from, so an old index answers too.
        """
        self.assertEqual(
            note_direction({"filter": "referenced_works:W1", "oql": "", "count": 3}),
            "cites")
        self.assertIsNone(note_direction({"filter": "ids.openalex:W1", "direction": None}))


class ACorruptedCacheEntryIsAMissTests(unittest.TestCase):
    """A cache entry truncated by a killed process is non-empty, so the
    cache serves it as a hit. Parsed without a guard, it raised
    json.JSONDecodeError out of get_json past every handler the crawl
    has -- on this run and on every later one, until somebody found the
    file and deleted it by hand.
    """

    BODY = json.dumps({"meta": {"count": 3, "x_query": {"oql": "", "url": ""}},
                       "results": []}).encode()
    URL = "https://api.openalex.org/works?filter=cites:W1"

    def test_a_truncated_page_is_paid_for_again_and_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            client = OpenAlexClient(opener=_Sequence([_Response(self.BODY)]),
                                    sleep=lambda _s: None, pause=0.0,
                                    cache=http_cache.DiskCache(directory))
            entry = directory / client._cache_name(self.URL)
            entry.write_text('{"meta": {"cou', encoding="utf-8")
            said = io.StringIO()
            with mock.patch("sys.stderr", said):
                body = client.get_json(self.URL)
            self.assertEqual(body["meta"]["count"], 3)
            self.assertEqual(client.n_requests, 1, "битая запись не перезапрошена")
            self.assertEqual(json.loads(entry.read_text(encoding="utf-8")), body,
                             "битая запись осталась в кэше")
        self.assertIn("кэш", said.getvalue().lower())


class OnePageIsTheWholeBatchTests(unittest.TestCase):
    """A short page is the last page.

    50 ids per filter (the measured cap) cannot fill a 200-record page, yet
    OpenAlex hands back a next_cursor all the same, and following it costs
    one guaranteed-empty request per batch out of a window of 1000 that
    refills over about a day.
    """

    def _page(self, n: int, cursor: str | None) -> bytes:
        return json.dumps({
            "results": [{"id": f"https://openalex.org/W{i}"} for i in range(n)],
            "meta": {"next_cursor": cursor},
        }).encode()

    def test_a_short_page_ends_the_batch_without_another_request(self):
        opener = _Sequence([_Response(self._page(3, "next")),
                            _Response(self._page(0, None))])
        client = OpenAlexClient(opener=opener, sleep=lambda _s: None, pause=0.0)
        got = list(client.works_by_ids([f"W{i}" for i in range(50)]))
        self.assertEqual(len(got), 3)
        self.assertEqual(opener.calls, 1, "заведомо пустой запрос всё-таки сделан")

    def test_a_full_page_is_still_followed(self):
        """The cut is `shorter than asked for`, not `smaller than the cap`:
        citers_of legitimately spans pages, and a batch whose first page came
        back full must still ask for the next one.
        """
        with mock.patch.object(openalex_client, "PER_PAGE", 2):
            opener = _Sequence([_Response(self._page(2, "next")),
                                _Response(self._page(1, None))])
            client = OpenAlexClient(opener=opener, sleep=lambda _s: None, pause=0.0)
            got = list(client.citers_of(["W0"]))
        self.assertEqual(len(got), 3)
        self.assertEqual(opener.calls, 2)


class NoClientHoldsAPathTests(unittest.TestCase):
    """DRY_RUN_WRITES_NOTHING holds for the response cache BY CONSTRUCTION,
    like store.Writer does for the graph -- which means no client may hold
    a path into the cache directory.

    The write half was already the object's; the read half was three
    independent is_file/read_text/count-the-hit sequences, so nothing but
    habit stopped a fourth client (or an edit to one of the three) from
    writing through the path it was handed, straight past ReadOnlyCache.
    The cache object's own contract is tested in test_http_cache.py; what
    belongs here is that the CLIENTS go through it.
    """

    CLIENTS = ("openalex_client.py", "zbmath_client.py", "mathnet.py",
               "http_session.py")

    def _source(self, name: str) -> str:
        return (Path(http_cache.__file__).resolve().parent / name).read_text(
            encoding="utf-8")

    def test_no_client_reads_the_cache_directory_itself(self):
        for name in self.CLIENTS:
            source = self._source(name)
            for spelling in (".is_file()", ".read_text(", ".write_text(",
                             ".stat()", "mkdir("):
                self.assertNotIn(
                    spelling, source,
                    f"{name}: {spelling} -- чтение/запись кэша принадлежат "
                    "http_cache, иначе --dry-run держится аккуратностью")


class MathnetClientTests(unittest.TestCase):
    """The identity anchor's two disciplines: a retry must not re-fetch the
    pages that already worked, and a page that did not arrive -- or arrived
    without the citations -- must be COUNTED, because a silent gap here
    weakens the twin index invisibly (it did, for 2019_rm9846).
    """

    def _client(self, opener, directory: Path | None = None,
                read_only: bool = False) -> MathnetClient:
        return MathnetClient(opener=opener, sleep=lambda _s: None,
                             cache=http_cache.cache_for(directory, read_only=read_only))

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

    def test_every_gap_is_named_against_the_id_it_is_about(self):
        """.failures is a sentence for a human; the caller that journals the
        gap needs the page it belongs to, and taking that back out of the
        sentence would be prose-parsing (JOURNAL_FACTS_ARE_COLUMNS). One
        writer fills both, so a gap cannot reach one channel and not the
        other.
        """
        client = self._client(_Sequence([TimeoutError("read timed out"),
                                         _Response(b"<html><body>gone</body></html>")]))
        self.assertEqual(client.titles("mzm8442"), ([], []))
        self.assertEqual(client.titles("sm280"), ([], []))
        self.assertEqual(sorted(client.problems), ["mzm8442", "sm280"])
        self.assertIn("TimeoutError", client.problems["mzm8442"])
        self.assertIn("без цитат", client.problems["sm280"])
        self.assertEqual(len(client.failures), len(client.problems))

    def test_a_page_that_answered_leaves_no_problem_recorded(self):
        client = self._client(_Sequence([_Response(PAGE.encode("windows-1251"))]))
        self.assertNotEqual(client.titles("mzm8442"), ([], []))
        self.assertEqual(client.problems, {})

    def test_a_titleless_page_is_cached_as_the_negative_answer_it_is(self):
        """The request SUCCEEDED -- the page simply carries no citation line.

        Re-asking buys nothing and costs a 0.6 s pause against a site that
        starts timing out, on the startup path of every non-offline run.
        The failure stays recorded on the cached round, so the gap in the
        identity anchor is as visible on the second run as on the first.
        """
        with tempfile.TemporaryDirectory() as tmp:
            opener = _Sequence([_Response(b"<html><body>redirected</body></html>"),
                                _Response(PAGE.encode("windows-1251"))])
            client = self._client(opener, Path(tmp))
            self.assertEqual(client.titles("mzm8442"), ([], []))
            self.assertEqual(client.titles("mzm8442"), ([], []))
        self.assertEqual(opener.calls, 1, "успешный ответ спрошен дважды")
        self.assertEqual(client.n_requests, 1)
        self.assertEqual(client.n_cache_hits, 1)
        self.assertEqual(len(client.failures), 2)

    def test_a_transport_failure_is_still_asked_again(self):
        """A cached blank would turn one timeout into a permanent verdict."""
        with tempfile.TemporaryDirectory() as tmp:
            opener = _Sequence([TimeoutError("read timed out"),
                                _Response(PAGE.encode("windows-1251"))])
            client = self._client(opener, Path(tmp))
            self.assertEqual(client.titles("mzm8442"), ([], []))
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
