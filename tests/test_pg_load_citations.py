"""Offline unit tests for the citation snowball's building blocks.

No network and no database: abstract reconstruction, the 50-id batch cap,
node identity (the twin union), the frontier arithmetic, the zbMATH
abstract fallback, the rate-limit reaction and the CSV/vector encodings the
loader writes through.

The crawl itself and its idempotency against real SQL live in
test_citations_crawl.py.
"""
from __future__ import annotations

import unittest
import urllib.error
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import (
    DIMS,
    FakeClient,
    PlannedEmbedder,
    unit,
    work,
)
import pg_load_citations
from citations import frontier, registry
from citations.openalex_client import (
    OpenAlexClient,
    QuotaExhausted,
    batched,
    restore_abstract,
    short_id,
)
from citations.store import csv_rows, vector_literal
from citations.zbmath_client import ZbmathClient, ZbmathUnavailable, abstract_of

# Verbatim from research/citation-sources/data/openalex_works_A5066843289_p1.json
# (W2074536792). Kept whole rather than trimmed: a truncated index would let a
# reconstruction bug that only shows on repeated words pass unnoticed.
REAL_INVERTED_INDEX = {
    "We": [0], "estimate": [1], "the": [2, 14, 23, 30, 41], "order": [3, 31],
    "of": [4, 7, 16, 19, 29, 32, 34], "weighted": [5], "approximations": [6],
    "functions": [8], "and": [9, 37, 44], "their": [10, 45], "derivatives": [11, 39],
    "by": [12, 40], "using": [13], "means": [15, 43], "mixed": [17], "series": [18],
    "Legendre": [20], "polynomials.": [21], "As": [22], "main": [24], "result,": [25],
    "we": [26], "obtain": [27], "estimates": [28], "approximation": [33], "a": [35],
    "function": [36], "its": [38], "Vallé-Poussin": [42], "derivatives.": [46],
}
REAL_ABSTRACT = (
    "We estimate the order of weighted approximations of functions and their "
    "derivatives by using the means of mixed series of Legendre polynomials. As "
    "the main result, we obtain estimates of the order of approximation of a "
    "function and its derivatives by the Vallé-Poussin means and their derivatives."
)


class InvertedIndexTests(unittest.TestCase):
    def test_inverted_index_roundtrip(self):
        self.assertEqual(restore_abstract(REAL_INVERTED_INDEX), REAL_ABSTRACT)

    def test_absent_index_is_none_not_empty_string(self):
        self.assertIsNone(restore_abstract(None))
        self.assertIsNone(restore_abstract({}))
        self.assertIsNone(restore_abstract({"word": []}))

    def test_repeated_word_lands_in_every_position(self):
        self.assertEqual(restore_abstract({"a": [0, 2], "b": [1]}), "a b a")


class BatchingTests(unittest.TestCase):
    def test_batches_never_exceed_fifty_ids(self):
        ids = [f"W{i}" for i in range(137)]
        chunks = list(batched(ids))
        self.assertEqual([len(c) for c in chunks], [50, 50, 37])
        self.assertEqual([i for c in chunks for i in c], ids)

    def test_asking_for_more_than_fifty_is_refused(self):
        with self.assertRaises(ValueError):
            list(batched(["W1"], size=51))

    def test_client_never_sends_more_than_fifty(self):
        records = [work(f"W{i}") for i in range(120)]
        client = FakeClient(records)
        client.works_by_ids([r["id"] for r in records])
        self.assertTrue(client.id_batches)
        self.assertLessEqual(max(len(b) for b in client.id_batches), 50)

    def test_short_id_strips_the_url(self):
        self.assertEqual(short_id("https://openalex.org/W123"), "W123")
        self.assertEqual(short_id("W123"), "W123")
        self.assertEqual(short_id(None), "")


class IdentityTests(unittest.TestCase):
    def test_two_records_sharing_a_doi_are_one_node(self):
        table = registry.WorkRegistry()
        original = work("W1", doi="10.4213/sm723", title="Оригинал")
        twin = work("W2", doi="10.4213/SM723", title="Translation")
        first, is_new_first = table.add(original, kind="external-skeleton", depth=1)
        second, is_new_second = table.add(twin, kind="external-skeleton", depth=1)
        self.assertTrue(is_new_first)
        self.assertFalse(is_new_second, "двойник по DOI создал второй узел")
        self.assertIs(first, second)
        self.assertEqual(len(table), 1)
        self.assertEqual(first.openalex_ids(), ["W1", "W2"])
        self.assertIn("openalex:W2", first.external_ids()["aliases"])

    def test_doi_case_and_prefix_do_not_split_a_node(self):
        self.assertEqual(registry.normalize_doi("https://doi.org/10.4213/SM723."),
                         registry.normalize_doi("10.4213/sm723"))

    def test_mag_id_alone_merges_two_records(self):
        table = registry.WorkRegistry()
        table.add(work("W1", mag="99"), kind="external-skeleton", depth=1)
        _node, is_new = table.add(work("W2", mag="99"), kind="external-skeleton", depth=1)
        self.assertFalse(is_new)

    def test_unrelated_records_stay_separate(self):
        table = registry.WorkRegistry()
        table.add(work("W1", doi="10.1/a"), kind="external-skeleton", depth=1)
        table.add(work("W2", doi="10.1/b"), kind="external-skeleton", depth=1)
        self.assertEqual(len(table), 2)

    def test_our_document_is_not_demoted_by_a_later_sighting(self):
        table = registry.WorkRegistry()
        table.add(work("W1", doi="10.1/a"), kind="our-document", depth=0, document_id="2003_sm723")
        node, _ = table.add(work("W2", doi="10.1/a"), kind="external-skeleton", depth=1)
        self.assertEqual(node.kind, "our-document")
        self.assertEqual(node.document_id, "2003_sm723")

    def test_union_fills_missing_fields_from_the_second_record(self):
        table = registry.WorkRegistry()
        table.add(work("W1", doi="10.1/a", abstract=None), kind="external-skeleton", depth=1)
        node, _ = table.add(work("W2", doi="10.1/a", abstract={"Meixner": [0]}),
                            kind="external-skeleton", depth=1)
        self.assertEqual(node.abstract, "Meixner")
        self.assertEqual(node.abstract_source, "openalex")


class FrontierMathTests(unittest.TestCase):
    def test_centroid_of_one_seed_is_that_seed(self):
        self.assertAlmostEqual(frontier.cosine(frontier.centroid([unit(0)]), unit(0)), 1.0)

    def test_centroid_normalizes_before_averaging(self):
        # A long vector must not outvote a short one: both are directions.
        centre = frontier.centroid([unit(0, 100.0), unit(1, 0.01)])
        self.assertAlmostEqual(frontier.cosine(centre, unit(0)),
                               frontier.cosine(centre, unit(1)), places=9)

    def test_split_by_threshold_keeps_the_boundary(self):
        kept, dropped = frontier.split_by_threshold({"a": 0.7, "b": 0.5, "c": 0.5001}, 0.5001)
        self.assertEqual(kept, ["a", "c"])
        self.assertEqual(dropped, ["b"])

    def test_quantiles_are_observed_values(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for value in frontier.quantiles(values).values():
            self.assertIn(value, values)

    def test_histogram_counts_every_value_once(self):
        values = [0.1 * i for i in range(37)]
        self.assertEqual(sum(c for _l, _h, c in frontier.histogram(values, bins=7)), 37)

    def test_candidate_text_survives_a_missing_abstract(self):
        self.assertEqual(frontier.candidate_text("T", None), "T")
        self.assertEqual(frontier.candidate_text(None, None), "")


class ZbmathAbstractTests(unittest.TestCase):
    def test_summary_precedes_review(self):
        record = {"editorial_contributions": [
            {"contribution_type": "review", "text": "R"},
            {"contribution_type": "summary", "text": "S"},
        ]}
        text, types = abstract_of(record)
        self.assertTrue(text.startswith("S"))
        self.assertEqual(types[0], "summary")

    def test_no_contribution_is_none_not_blank(self):
        self.assertEqual(abstract_of({"editorial_contributions": []}), (None, []))
        self.assertEqual(abstract_of(None), (None, []))


class HeaderResponse:
    """Minimal urlopen stand-in: body plus headers, usable as a context manager."""

    def __init__(self, body: str, headers: dict, status: int = 200):
        self._body, self.headers, self.status = body.encode(), dict(headers), status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class RateLimitTests(unittest.TestCase):
    def test_rate_limit_headers_trigger_wait(self):
        slept = []
        client = OpenAlexClient(
            opener=lambda *_a, **_k: HeaderResponse(
                '{"results": [], "meta": {}}',
                {"x-ratelimit-remaining": "12", "x-ratelimit-reset": "45"},
            ),
            sleep=slept.append, pause=0.0,
        )
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertIn(45.0, slept, f"ожидалась пауза до сброса окна, паузы: {slept}")

    def test_plenty_of_quota_does_not_wait_for_the_reset(self):
        slept = []
        client = OpenAlexClient(
            opener=lambda *_a, **_k: HeaderResponse(
                '{"results": [], "meta": {}}',
                {"x-ratelimit-remaining": "900", "x-ratelimit-reset": "45"},
            ),
            sleep=slept.append, pause=0.0,
        )
        client.get_json("https://api.openalex.org/works?x=1")
        self.assertNotIn(45.0, slept)

    def test_a_reset_further_away_than_we_will_wait_raises_instead(self):
        client = OpenAlexClient(
            opener=lambda *_a, **_k: HeaderResponse(
                '{"results": [], "meta": {}}',
                {"x-ratelimit-remaining": "3", "x-ratelimit-reset": "83942"},
            ),
            sleep=lambda _s: None, pause=0.0, max_quota_wait=900.0,
        )
        with self.assertRaises(QuotaExhausted):
            client.get_json("https://api.openalex.org/works?x=1")

    def test_every_request_counts(self):
        client = OpenAlexClient(
            opener=lambda *_a, **_k: HeaderResponse('{"results": [], "meta": {}}', {}),
            sleep=lambda _s: None, pause=0.0,
        )
        client.get_json("https://api.openalex.org/works?x=1")
        client.get_json("https://api.openalex.org/works?x=2")
        self.assertEqual(client.n_requests, 2)


class PagedStreamingTests(unittest.TestCase):
    """Cursor pages reach the caller one at a time.

    A depth-2 batch set returned over 51000 works (crawl.py's own
    docstring); accumulating every page into one list before the caller
    sees anything held all of them at once, and the caller drops most of
    them on the next line.
    """

    class PagedOpener:
        """Three cursor pages of one work each, logging every fetch."""

        PAGES = [
            '{"results": [{"id": "https://openalex.org/W1"}], "meta": {"next_cursor": "c2"}}',
            '{"results": [{"id": "https://openalex.org/W2"}], "meta": {"next_cursor": "c3"}}',
            '{"results": [{"id": "https://openalex.org/W3"}], "meta": {}}',
        ]

        def __init__(self, log):
            self.log = log
            self.n = 0

        def __call__(self, _request, timeout=None):
            page = self.PAGES[self.n]
            self.n += 1
            self.log.append(f"fetch{self.n}")
            return HeaderResponse(page, {})

    def _client(self, log):
        return OpenAlexClient(opener=self.PagedOpener(log), sleep=lambda _s: None, pause=0.0)

    def test_nothing_is_fetched_before_the_first_record_is_asked_for(self):
        log = []
        pages = self._client(log)._paged(filter="cites:W0")
        self.assertEqual(log, [])
        next(iter(pages))
        self.assertEqual(log, ["fetch1"])

    def test_pages_interleave_with_consumption_instead_of_piling_up(self):
        log = []
        for record in self._client(log).citers_of(["W0"]):
            log.append("seen " + record["id"].rsplit("/", 1)[-1])
        self.assertEqual(log, ["fetch1", "seen W1", "fetch2", "seen W2",
                               "fetch3", "seen W3"])

    def test_works_by_ids_streams_the_same_way(self):
        log = []
        seen = []
        for record in self._client(log).works_by_ids(["W0"]):
            seen.append(record["id"].rsplit("/", 1)[-1])
            self.assertEqual(len(log), len(seen), "страница прочитана впрок")
        self.assertEqual(seen, ["W1", "W2", "W3"])


class EvidenceWeightTests(unittest.TestCase):
    """A node keeps the ids it extracted, not the list it extracted them
    from: node.records becomes citation.work.evidence, and referenced_works
    is by far the bulkiest field OpenAlex returns.
    """

    def test_absorb_keeps_the_ids_and_drops_the_list(self):
        node = registry.Node(key="W1", kind="external-skeleton", depth=1)
        record = work("W1", refs=["W2", "W3"])
        node.absorb(record)
        self.assertEqual(node.referenced_works, {"W2", "W3"})
        self.assertNotIn("referenced_works", node.records[0])
        self.assertEqual(node.records[0]["referenced_works_count"], 2,
                         "OpenAlex's own count stays -- hub_report counts by it")

    def test_the_callers_record_is_left_alone(self):
        record = work("W1", refs=["W2"])
        registry.Node(key="W1", kind="external-skeleton", depth=1).absorb(record)
        self.assertIn("referenced_works", record,
                      "edges.among_known reads referenced_works off the record")


class CsvEncodingTests(unittest.TestCase):
    def test_none_and_embedded_separators_survive(self):
        text = csv_rows([["a,b", None, 'quote"inside', "line\nbreak"]])
        self.assertIn('"a,b"', text)
        self.assertIn('"line\nbreak"', text)

    def test_vector_literal_is_pgvector_shaped(self):
        self.assertEqual(vector_literal([1.0, 0.5]), "[1.0,0.5]")
        self.assertIsNone(vector_literal(None))




class ZbmathFailureVsAbsenceTests(unittest.TestCase):
    """"zbMATH does not have it" and "the request failed" are different
    answers, and the crawl's own discipline (a missing thing is a RECORDED
    decision, never indistinguishable from a bug) applies to both.
    """

    class _Response:
        def __init__(self, body):
            self._body = body.encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _client(self, opener):
        return ZbmathClient(opener=opener, sleep=lambda _s: None)

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


class ZbmathAbstractsJournalTests(unittest.TestCase):
    """A failed fetch is journalled as action='error', so "no abstract" in
    the graph can always be told apart from "we never got an answer".
    """

    class _Writer:
        def __init__(self):
            self.steps = []

        def journal(self, steps):
            self.steps += list(steps)
            return len(steps)

    def _run(self, document_side_effect):
        writer = self._Writer()
        client = mock.Mock(n_requests=1, failures=[], document=mock.Mock(side_effect=document_side_effect))
        with mock.patch.object(pg_load_citations, "seed_matches",
                                return_value={"1997_sm280": "1234.56789"}), \
             mock.patch.object(pg_load_citations, "ZbmathClient", return_value=client):
            out = pg_load_citations.zbmath_abstracts(
                {}, ["1997_sm280"], {"1997_sm280": "W1"}, writer=writer, crawl_id="t",
            )
        return out, writer

    def test_absence_writes_no_error_row(self):
        out, writer = self._run(lambda _id: None)
        self.assertEqual(out, {})
        self.assertEqual([s["action"] for s in writer.steps], [])

    def test_failure_writes_an_error_row_naming_the_document(self):
        out, writer = self._run(ZbmathUnavailable("1234.56789: HTTP 429"))
        self.assertEqual(out, {})
        self.assertEqual([s["action"] for s in writer.steps], ["error"])
        self.assertEqual(writer.steps[0]["frontier_key"], "1997_sm280")
        self.assertIn("429", writer.steps[0]["reason"])


if __name__ == "__main__":
    unittest.main()
