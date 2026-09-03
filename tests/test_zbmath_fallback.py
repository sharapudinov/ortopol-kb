"""The two seed-metadata surfaces: the zbMATH fallback and the Math-Net
title anchor.

The zbMATH half is the one thing run 85 left that source doing.

Split from test_pg_load_citations.py (kb/CLAUDE.md FILE_SIZE) along the
line the module docstrings already draw: that file is about the crawl's
own arithmetic and encodings, this one about a single third-party surface
and the distinction it is built around -- "zbMATH does not have it" and
"we never found out" are different answers, and only the first may ever be
remembered as an absence.

No network: every opener is a stub, every sleep is captured, every cache is
a temporary directory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
from citations import inputs, seed_metadata
from citations.http_cache import DiskCache
from citations.zbmath_client import ZbmathClient, ZbmathUnavailable, abstract_of


class ZbmathAbstractTests(unittest.TestCase):
    def test_summary_precedes_review(self):
        record = {"editorial_contributions": [
            {"contribution_type": "review", "text": "R"},
            {"contribution_type": "summary", "text": "S"},
        ]}
        text, types = abstract_of(record)
        self.assertTrue(text.startswith("S"))
        self.assertEqual(types[0], "summary")

    def test_a_blank_contribution_is_named_by_neither_half(self):
        """The types are what the abstract is MADE of, so a contribution
        whose text is whitespace cannot appear among them: it does not
        reach the joined text, and evidence.abstract_source naming it would
        attribute the abstract to a source that contributed nothing.
        """
        record = {"editorial_contributions": [
            {"contribution_type": "summary", "text": "   \n  "},
            {"contribution_type": "review", "text": "R"},
        ]}
        self.assertEqual(abstract_of(record), ("R", ["review"]))

    def test_a_record_of_nothing_but_blanks_is_no_abstract(self):
        record = {"editorial_contributions": [
            {"contribution_type": "summary", "text": "  "},
            {"contribution_type": "review", "text": ""},
        ]}
        self.assertEqual(abstract_of(record), (None, []))

    def test_a_contribution_without_a_type_is_still_named(self):
        record = {"editorial_contributions": [{"text": "T"}]}
        self.assertEqual(abstract_of(record), ("T", ["unknown"]))

    def test_no_contribution_is_none_not_blank(self):
        self.assertEqual(abstract_of({"editorial_contributions": []}), (None, []))
        self.assertEqual(abstract_of(None), (None, []))



class ZbmathFailureVsAbsenceTests(unittest.TestCase):
    """"zbMATH does not have it" and "the request failed" are different
    answers, and the crawl's own discipline (a missing thing is a RECORDED
    decision, never indistinguishable from a bug) applies to both.
    """

    class _Response:
        """urlopen stand-in: a body and the headers a real answer carries."""

        status = 200

        def __init__(self, body):
            self._body = body.encode()
            self.headers = {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _client(self, opener, cache=None):
        return ZbmathClient(opener=opener, sleep=lambda _s: None, cache=cache)

    def test_404_is_a_legitimate_absence(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

        client = self._client(opener)
        self.assertIsNone(client.document("1234.56789"))
        self.assertEqual(client.failures, [])

    def test_a_record_without_contributions_is_also_an_absence(self):
        client = self._client(lambda request, timeout=None: self._Response('{"result": {}}'))
        self.assertEqual(abstract_of(client.document("1234.56789")), (None, []))
        self.assertEqual(client.failures, [])

    def test_rate_limit_is_a_failure_not_an_absence(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

        client = self._client(opener)
        with self.assertRaises(ZbmathUnavailable) as ctx:
            client.document("1234.56789")
        self.assertIn("429", str(ctx.exception))
        self.assertEqual(len(client.failures), 1)

    def test_network_error_is_a_failure_not_an_absence(self):
        def opener(request, timeout=None):
            raise TimeoutError("read timed out")

        client = self._client(opener)
        with self.assertRaises(ZbmathUnavailable):
            client.document("1234.56789")
        self.assertEqual(len(client.failures), 1)

    def test_unparseable_body_is_a_failure_too(self):
        client = self._client(lambda request, timeout=None: self._Response("<html>502</html>"))
        with self.assertRaises(ZbmathUnavailable):
            client.document("1234.56789")
        self.assertEqual(len(client.failures), 1)

    REVIEWED = ('{"result": {"editorial_contributions": '
                '[{"contribution_type": "review", "text": "разбор"}]}}')

    def test_a_cached_answer_costs_no_second_request(self):
        """These abstracts are static between runs, and the fallback runs on
        the startup path of every non-offline invocation: one sequential
        request per matched seed, with a pause, before anything happens.
        """
        calls = []

        def opener(request, timeout=None):
            calls.append(request.full_url)
            return self._Response(self.REVIEWED)

        with tempfile.TemporaryDirectory() as tmp:
            first = ZbmathClient(opener=opener, sleep=lambda _s: None,
                                 cache=DiskCache(Path(tmp)))
            self.assertEqual(abstract_of(first.document("1234.56789"))[0], "разбор")
            second = ZbmathClient(opener=opener, sleep=lambda _s: None,
                                  cache=DiskCache(Path(tmp)))
            self.assertEqual(abstract_of(second.document("1234.56789"))[0], "разбор")
        self.assertEqual(calls, ["https://api.zbmath.org/v1/document/1234.56789"])
        self.assertEqual((second.n_requests, second.n_cache_hits), (0, 1))

    def test_a_truncated_cache_entry_is_asked_again_and_named(self):
        """An entry cut short by a killed process is non-empty, so it is
        served as a hit. Parsed without a guard it raised a bare
        JSONDecodeError -- not a ZbmathUnavailable, so not in .failures and
        not caught by the caller: the one outcome this client exists to
        keep apart from "zbMATH has no record".
        """
        calls = []

        def opener(request, timeout=None):
            calls.append(request.full_url)
            return self._Response(self.REVIEWED)

        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / "1234.56789.json"
            entry.write_text('{"result": {"editorial_cont', encoding="utf-8")
            client = ZbmathClient(opener=opener, sleep=lambda _s: None,
                                  cache=DiskCache(Path(tmp)))
            self.assertEqual(abstract_of(client.document("1234.56789"))[0], "разбор")
            self.assertEqual(len(calls), 1, "битая запись не перезапрошена")
            self.assertEqual(json.loads(entry.read_text(encoding="utf-8"))
                             ["editorial_contributions"][0]["text"], "разбор",
                             "битая запись осталась в кэше")
        self.assertEqual(len(client.failures), 1,
                         "нечитаемая запись кэша не названа")
        self.assertIn("1234.56789", client.failures[0])

    def test_a_failure_is_never_cached(self):
        """A cached blank would turn one 429 into a permanent "no review" --
        the very confusion ZbmathUnavailable exists to prevent.
        """
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, None)

        with tempfile.TemporaryDirectory() as tmp:
            client = ZbmathClient(opener=opener, sleep=lambda _s: None,
                                  cache=DiskCache(Path(tmp)))
            with self.assertRaises(ZbmathUnavailable):
                client.document("1234.56789")
            self.assertEqual(list(Path(tmp).iterdir()), [])


class ZbmathAbstractsJournalTests(unittest.TestCase):
    """A failed fetch is journalled as action='error', so "no abstract" in
    the graph can always be told apart from "we never got an answer".
    """

    class _Writer:
        def __init__(self):
            self.steps = []
            self.calls = []

        def journal(self, steps):
            batch = list(steps)
            self.calls.append(batch)
            self.steps += batch
            return len(batch)

    def _run(self, document_side_effect, stored=None):
        writer = self._Writer()
        client = mock.Mock(n_requests=1, n_cache_hits=0, failures=[],
                           document=mock.Mock(side_effect=document_side_effect))
        with mock.patch.object(seed_metadata, "seed_matches",
                                return_value={"1997_sm280": "1234.56789"}), \
             mock.patch.object(seed_metadata, "stored_zbmath_abstracts",
                                return_value=stored or {}), \
             mock.patch.object(seed_metadata, "ZbmathClient", return_value=client):
            out = seed_metadata.zbmath_abstracts(
                {}, ["1997_sm280"], {"1997_sm280": "W1"}, cache=None, writer=writer,
                crawl_id="t", log=lambda *_: None,
            )
        return out, writer, client

    def test_an_abstract_already_in_the_graph_is_not_asked_for_again(self):
        """The stored one is used as it stands, provenance included: only
        rows whose evidence says the abstract came from zbMATH are read back
        (citations.inputs.stored_zbmath_abstracts), so nothing here can
        relabel an OpenAlex abstract as a zbMATH review.
        """
        out, writer, client = self._run(lambda _id: None,
                                        stored={"1997_sm280": "уже в графе"})
        self.assertEqual(out, {"1997_sm280": ("уже в графе", "1234.56789")})
        client.document.assert_not_called()
        self.assertEqual(writer.steps, [])

    def test_absence_writes_no_error_row(self):
        out, writer, _client = self._run(lambda _id: None)
        self.assertEqual(out, {})
        self.assertEqual([s["action"] for s in writer.steps], [])

    def test_failure_writes_an_error_row_naming_the_document(self):
        out, writer, _client = self._run(ZbmathUnavailable("1234.56789: HTTP 429"))
        self.assertEqual(out, {})
        self.assertEqual([s["action"] for s in writer.steps], ["error"])
        self.assertEqual(writer.steps[0]["frontier_key"], "1997_sm280")
        self.assertIn("429", writer.steps[0]["reason"])

    def test_the_journal_is_written_through_the_seam_either_way(self):
        """No branch on the writer: an empty batch is a no-op at both
        writers, and a `if writer is not None` here is the flag the seam
        exists to remove. The call happens; what it costs is the object's
        business.
        """
        _out, quiet, _client = self._run(lambda _id: None)
        self.assertEqual(quiet.calls, [[]])
        _out, loud, _client = self._run(ZbmathUnavailable("1234.56789: HTTP 429"))
        self.assertEqual(len(loud.calls), 1)
        self.assertEqual([s["action"] for s in loud.calls[0]], ["error"])

    def test_neither_the_writer_nor_the_run_may_be_forgotten(self):
        """The caller this module advertises has no command line, and an
        omitted seam must be a TypeError rather than a run whose failures
        vanish (or journal rows detached from their crawl).
        """
        for omit in ("writer", "crawl_id"):
            keywords = {"cache": None, "writer": self._Writer(), "crawl_id": "t",
                        "log": lambda *_: None}
            del keywords[omit]
            with self.subTest(omitted=omit), self.assertRaises(TypeError):
                seed_metadata.zbmath_abstracts({}, [], {}, **keywords)


class MathnetNamesTests(unittest.TestCase):
    """The identity anchor asks the database before it asks the site.

    Both titles of a seed are stored on its own citation.work row
    (external_ids->'titles', which twin_pass.seed_titles reads back), so a
    second crawl already knows them. Without that short-circuit the anchor
    re-walked every seed page on the startup path of EVERY crawl and every
    --calibrate -- one request and a 0.6 s pause per miss, and under
    --dry-run the cache persists nothing, so nothing was ever a hit twice.
    """

    def _run(self, stored=None, titles=(["Title"], [1989])):
        client = mock.Mock(n_requests=1, n_cache_hits=0, failures=[],
                           titles=mock.Mock(return_value=titles))
        with mock.patch.object(
                seed_metadata, "corpus_seed_documents",
                return_value=[("1997_sm280", "https://www.mathnet.ru/rus/sm280")]), \
             mock.patch.object(seed_metadata, "stored_mathnet_titles",
                               return_value=stored or {}), \
             mock.patch.object(seed_metadata, "MathnetClient", return_value=client):
            out = seed_metadata.mathnet_names({}, cache=None, log=lambda *_: None)
        return out, client

    def test_titles_already_in_the_graph_cost_no_request(self):
        out, client = self._run(stored={"1997_sm280": (["Уже в базе"], [1989])})
        self.assertEqual(out, {"1997_sm280": (["Уже в базе"], [1989])})
        client.titles.assert_not_called()

    def test_a_seed_without_stored_titles_is_still_fetched(self):
        out, client = self._run()
        client.titles.assert_called_once_with("sm280")
        self.assertEqual(out, {"1997_sm280": (["Title"], [1989])})

    def test_the_stored_read_takes_titles_and_years_off_the_work_row(self):
        answer = '[["1997_sm280", ["Рус", "Eng"], [1989, 1991]], ["x", [], []]]'
        with mock.patch.object(inputs, "scalar", return_value=answer):
            stored = inputs.stored_mathnet_titles({})
        self.assertEqual(stored, {"1997_sm280": (["Рус", "Eng"], [1989, 1991])})


if __name__ == "__main__":
    unittest.main()
