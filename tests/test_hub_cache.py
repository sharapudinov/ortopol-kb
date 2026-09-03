"""Что проход по кэшу ответов OpenAlex знает о батчах «вверх».

Отделено от test_citations_form.py по ответственности (и по kb/CLAUDE.md
FILE_SIZE): тот файл — форма обхода и правило двойников, этот — читатель
кэша. Всё офлайн: страницы пишутся в каталог тут же, сети не требуется.

Числа в этих проверках — те же, что живут в замере: один meta.count на
БАТЧ, а не на страницу (первая версия ключевалась на url с курсором в
хвосте и опубликовала 3 392 521 обещанного цитирующего вместо 51 652).
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import _pathfix  # noqa: F401
from citations import hub_cache, http_cache


class _RecordingCache(http_cache.DiskCache):
    """A DiskCache that remembers WHAT was asked for and how much of it."""

    def __init__(self, directory):
        super().__init__(directory)
        self.reads: list[tuple[str, int | None]] = []

    def read(self, name, *, floor=0, limit=None):
        self.reads.append((name, limit))
        return super().read(name, floor=floor, limit=limit)


class BatchCountTests(unittest.TestCase):
    """One number per BATCH, not per page.

    The first version keyed on x_query.url, which carries the cursor in its
    tail: 8 batches came back as 253 distinct urls and the report published
    3 392 521 promised citers instead of 51 652.
    """

    def _page(self, directory, name, ids, count, cursor, oql=None):
        body = {"meta": {"count": count, "x_query": {
            "oql": oql or ("works where it cites (" + " or ".join(ids) + ")"),
            "url": f"/works?filter=referenced_works:{'|'.join(ids)}"
                   f"&per_page=200&cursor={cursor}"}}}
        (directory / name).write_text(json.dumps(body), encoding="utf-8")

    def test_pages_of_one_batch_are_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            self._page(directory, "b.json", ["W1", "W2"], 18904, "BBB")
            self._page(directory, "c.json", ["W3"], 21, "CCC")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)), [18904, 21])

    def _down_page(self, directory, name):
        (directory / name).write_text(json.dumps({"meta": {
            "count": 50, "x_query": {"oql": "works where openalex id is (W1)",
                                     "url": "/works?filter=ids.openalex:W1"}},
            "results": [{"id": "W1", "referenced_works": ["W%d" % i for i in range(500)]}]}),
            encoding="utf-8")

    def test_no_page_is_ever_read_or_decoded_whole(self):
        """The cache is 217 MiB of works with their referenced_works lists,
        and the report needs two fields per BATCH -- both of them inside
        meta, the first object of the body. So no page is read past its
        head, in either direction, and nothing bigger than meta is decoded.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            self._down_page(directory, "b.json")
            cache = _RecordingCache(directory)
            decoded = []
            real = json.loads

            def counting(text, *args, **kwargs):
                decoded.append(len(text))
                return real(text, *args, **kwargs)

            with mock.patch.object(hub_cache.json, "loads", counting):
                self.assertEqual(hub_cache.HubCacheReader(cache).batch_counts(), [18904])
            whole = [name for name, limit in cache.reads
                     if limit is None and not name.endswith(".meta.json")]
            self.assertEqual(whole, [], "страница прочитана целиком")
            page = (directory / "b.json").stat().st_size
            self.assertLess(max(decoded), page, "разобрано тело, а не meta")

    def test_a_meta_too_large_for_the_head_falls_back_to_the_whole_page(self):
        """The cheap path is not the only path: a page whose meta does not
        fit the head is still read, just expensively. Silence here would be
        a batch quietly missing from the measurement.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            body = {"results": [{"id": "W1", "padding": "x" * hub_cache.HEAD_BYTES}],
                    "meta": {"count": 777, "x_query": {
                        "oql": "works where it cites (W1)",
                        "url": "/works?filter=referenced_works:W1&cursor=AAA"}}}
            (directory / "a.json").write_text(json.dumps(body), encoding="utf-8")
            cache = _RecordingCache(directory)
            self.assertEqual(hub_cache.HubCacheReader(cache).batch_counts(), [777])
            self.assertIn(("a.json", None), cache.reads)

    def test_the_direction_is_the_filter_not_the_english_of_the_oql(self):
        """meta.x_query.oql is OpenAlex's rendered sentence for a human, and
        a reader sniffing it for the word "cites" is parsing a third party's
        presentation text: a rewording turns every page into not-a-batch and
        the measurement reports "nothing to measure" against a full cache.
        Both pages below lie in their oql; both are classified by filter.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1"], 18904, "AAA",
                       oql="произведения, ссылающиеся на (W1)")
            (directory / "b.json").write_text(json.dumps({"meta": {
                "count": 50, "x_query": {"oql": "works where it cites (W2)",
                                         "url": "/works?filter=ids.openalex:W2"}}}),
                encoding="utf-8")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)),
                             [18904])

    def test_a_sidecar_written_before_the_direction_existed_still_counts(self):
        """259 pages and their indexes are older than the field; a reader
        that needed the new key would report an empty cache.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1"], 18904, "AAA")
            (directory / "a.meta.json").write_text(json.dumps({
                "filter": "referenced_works:W1",
                "oql": "works where it cites (W1)", "count": 18904}), encoding="utf-8")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)),
                             [18904])

    def test_the_second_pass_reads_the_sidecar_not_the_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)), [18904])
            # The body is now unreadable: only a reader that still parses it
            # can notice, and the answer must not change.
            (directory / "a.json").write_text("{ not json at all", encoding="utf-8")
            # A NEXT pass has a memo of its own -- what makes it cheap is
            # the sidecar, and that is what this asks about.
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)), [18904])

    def test_a_second_pass_by_one_reader_reads_nothing_again(self):
        """Under --dry-run the cache is read-only, so no sidecar can be
        written and every pass would re-read (and re-parse) the whole page.
        What the reader has already read stays on the READER, so the saving
        lasts exactly as long as the object does.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            cache = http_cache.ReadOnlyCache(directory)
            reader = hub_cache.HubCacheReader(cache)
            self.assertEqual(reader.batch_counts(), [18904])
            first = cache.hits
            self.assertGreater(first, 0)
            self.assertEqual(reader.batch_counts(), [18904])
            self.assertEqual(cache.hits, first,
                             "страница прочитана второй раз")

    def test_two_readers_over_one_directory_share_nothing(self):
        """A DiskCache and a ReadOnlyCache over the same directory in one
        process used to share memo entries through a module-level dict
        keyed by the path, so what one read was served to the other --
        including after the page had been rewritten.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            first = hub_cache.HubCacheReader(http_cache.ReadOnlyCache(directory))
            self.assertEqual(first.batch_counts(), [18904])
            self._page(directory, "a.json", ["W1", "W2"], 21, "AAA")
            second = hub_cache.HubCacheReader(http_cache.ReadOnlyCache(directory))
            self.assertEqual(second.batch_counts(), [21],
                             "новый читатель получил чужую память")

    def test_a_sidecar_is_not_mistaken_for_a_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1", "W2"], 18904, "AAA")
            hub_cache.batch_counts(http_cache.DiskCache(directory))
            self.assertTrue((directory / "a.meta.json").is_file())
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)), [18904])

    def test_a_truncated_sidecar_is_recovered_from_the_page_and_rewritten(self):
        """Битый сайдкар — не «страницы нет».

        Сайдкар пишется в дерево данных и переживает обрыв процесса
        огрызком: непустым, а значит попаданием. Страница при этом цела, и
        путь голова/тело её разбирает — иначе полный кэш молча отчитался бы
        нулём (или NothingToMeasure), то есть ровно тем тихим нулём, против
        которого этот читатель и написан.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._page(directory, "a.json", ["W1"], 18904, "AAA")
            sidecar = directory / "a.meta.json"
            sidecar.write_text('{"filter": "referenced_wo', encoding="utf-8")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)),
                             [18904])
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["count"],
                             18904, "восстановленный сайдкар не переписан")

    def test_openalex_id_batches_are_not_counted_as_cites(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "d.json").write_text(json.dumps({"meta": {
                "count": 50, "x_query": {"oql": "works where openalex id is (W1)",
                                         "url": "/works?filter=ids.openalex:W1"}}}),
                encoding="utf-8")
            self.assertEqual(hub_cache.batch_counts(http_cache.DiskCache(directory)), [])


if __name__ == "__main__":
    unittest.main()
