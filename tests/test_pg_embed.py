"""How pg_embed.py spends its round trips, and which door it goes through.

Two batch sizes, not one: the database is asked for a PAGE of pending rows
(FETCH_BATCH) and ollama is asked for vectors a request at a time
(EMBED_BATCH). Using the ollama size for both meant a psql process, a temp
script and a fresh connection for every 16 rows -- twice, since the UPDATEs
went in a second invocation -- over a table that grows with every crawl.

Nothing here touches Postgres or ollama. The seams are pg_common's
(scalar/run_sql), pg_copy's staging copy and embed(): the module has no
Postgres path of its own to stub, which is the point -- one access path
means one env, and the `works` target writes into the citation schema.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_embed
from pg_common import FIELD_SEP, RECORD_SEP


class FakeDatabase:
    """A table of `total` pending rows, answering pg_embed's two reads and
    consuming its staged page.

    The cursor is honoured, not ignored: the loop pages by `r.id > after`,
    and a stub that answers from the head of the pending list whatever the
    query said would let a lost cursor pass unnoticed.
    """

    def __init__(self, total: int):
        self.pending = list(range(1, total + 1))
        self.selects: list[str] = []
        self.counts: list[str] = []
        self.copies: list[list] = []

    def scalar(self, env, sql, **kwargs):
        self.counts.append(sql)
        return f"{len(self.pending)}"

    def run_sql(self, env, sql, **kwargs):
        self.selects.append(sql)
        limit = int(sql.rsplit("limit", 1)[1].strip(" ;\n"))
        after = int(sql.split("r.id > ", 1)[1].split(" ", 1)[0])
        page = [row_id for row_id in self.pending if row_id > after][:limit]
        return mock.Mock(stdout="".join(
            f"{row_id}{FIELD_SEP}text {row_id}{RECORD_SEP}" for row_id in page))

    def copy_csv_rows(self, env, table_columns, rows, *, preamble="", epilogue=""):
        staged = list(rows)
        self.copies.append([table_columns, preamble, epilogue, staged])
        self.pending = [row for row in self.pending
                        if row not in {row_id for row_id, _vector in staged}]
        return mock.Mock(rows=len(staged), stdout=f"{len(staged)}\n")


class RoundTripsPerPageTests(unittest.TestCase):
    ENV = {"PGDATABASE": "ortopol"}

    def _run(self, total: int) -> tuple[FakeDatabase, list[int]]:
        database = FakeDatabase(total)
        chunks: list[int] = []

        def embed(texts, model, dims):
            for start in range(0, len(texts), pg_embed.EMBED_BATCH):
                chunks.append(len(texts[start:start + pg_embed.EMBED_BATCH]))
            return [[0.0] * dims for _ in texts]

        with mock.patch.object(pg_embed, "scalar", side_effect=database.scalar), \
             mock.patch.object(pg_embed, "run_sql", side_effect=database.run_sql), \
             mock.patch.object(pg_embed, "copy_csv_rows", side_effect=database.copy_csv_rows), \
             mock.patch.object(pg_embed, "embed", side_effect=embed), \
             mock.patch("builtins.print"):
            left = pg_embed.embed_target(self.ENV, "works", "bge-m3", 4)
        self.assertEqual(left, 0, "остаток посчитан не арифметикой")
        self.assertEqual(database.pending, [])
        return database, chunks

    def test_a_full_page_costs_one_select_and_one_copy(self):
        database, _ = self._run(pg_embed.FETCH_BATCH)
        self.assertEqual(len(database.copies), 1,
                         "вся страница обновляется одним вызовом psql")
        # Two selects: the page, and the one that finds nothing left.
        self.assertEqual(len(database.selects), 2)
        self.assertEqual(len(database.counts), 1)

    def test_the_page_is_the_fetch_batch_not_the_ollama_batch(self):
        database, _ = self._run(pg_embed.FETCH_BATCH)
        self.assertIn(f"limit {pg_embed.FETCH_BATCH}", database.selects[0])

    def test_ollama_is_still_asked_in_embed_batch_sized_requests(self):
        _, chunks = self._run(pg_embed.FETCH_BATCH)
        self.assertEqual(sum(chunks), pg_embed.FETCH_BATCH)
        self.assertTrue(all(size <= pg_embed.EMBED_BATCH for size in chunks), chunks)
        self.assertGreater(len(chunks), 1, "партии к ollama остаются мелкими")

    def test_more_than_one_page_is_more_than_one_round_trip(self):
        database, _ = self._run(pg_embed.FETCH_BATCH + 5)
        self.assertEqual(len(database.copies), 2)
        self.assertEqual(len(database.selects), 3)

    def test_three_pages_walk_forward_without_reading_a_row_twice(self):
        """Each page starts past the last id of the one before, so no row
        is offered to ollama twice and the text expression is never
        evaluated again for ground already covered.
        """
        database, _ = self._run(2 * pg_embed.FETCH_BATCH + 7)
        seen = [row_id for _table, _pre, _epi, staged in database.copies
                for row_id, _vector in staged]
        self.assertEqual(len(database.copies), 3)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(seen), len(set(seen)))
        cursors = [int(sql.split("r.id > ", 1)[1].split(" ", 1)[0])
                   for sql in database.selects]
        self.assertEqual(cursors, [0, pg_embed.FETCH_BATCH,
                                   2 * pg_embed.FETCH_BATCH,
                                   2 * pg_embed.FETCH_BATCH + 7])

    def test_the_target_is_counted_once_and_the_remainder_is_arithmetic(self):
        """embed_target() counts what is pending; the closing report reads
        that count's remainder instead of asking for it again. Three
        targets asked twice each was six full aggregate scans and six psql
        processes for three numbers.
        """
        database = FakeDatabase(pg_embed.FETCH_BATCH)
        with mock.patch.object(pg_embed, "scalar", side_effect=database.scalar), \
             mock.patch.object(pg_embed, "run_sql", side_effect=database.run_sql), \
             mock.patch.object(pg_embed, "copy_csv_rows",
                               side_effect=database.copy_csv_rows), \
             mock.patch.object(pg_embed, "embed",
                               side_effect=lambda texts, model, dims:
                               [[0.0] * dims for _ in texts]), \
             mock.patch("builtins.print"):
            left = pg_embed.embed_target(self.ENV, "works", "bge-m3", 4)
            gaps = pg_embed.missing_semantic_key(self.ENV, {"works": left})
        self.assertEqual(len(database.counts), len(pg_embed.TARGETS),
                         "цель посчитана дважды за прогон")
        self.assertNotIn("works", dict(gaps))

    def test_the_page_is_staged_and_consumed_in_one_script(self):
        """The same seam citations/store.py writes through: a TEMP table,
        one \\copy, and the UPDATE that reads it, in one session -- not 200
        concatenated statements each carrying 1024 numbers in a literal.
        """
        database, _ = self._run(pg_embed.FETCH_BATCH)
        table_columns, preamble, epilogue, staged = database.copies[0]
        self.assertEqual(table_columns, "stage_embedding (id, embedding)")
        self.assertIn("CREATE TEMP TABLE stage_embedding", preamble)
        self.assertIn("ON COMMIT DROP", preamble)
        self.assertIn("UPDATE citation.work", epilogue)
        self.assertIn("s.embedding::vector", epilogue)
        self.assertEqual(len(staged), pg_embed.FETCH_BATCH)
        self.assertEqual(staged[0], [1, "[0.0,0.0,0.0,0.0]"])


class OnePostgresPathTests(unittest.TestCase):
    """One door to the database, and it takes an env: a module that reads
    the model through pg_common while writing through a subprocess call of
    its own has two behaviours to keep in step and only one of them can be
    pointed at a --pgenv.
    """

    def test_the_module_holds_no_subprocess_call_of_its_own(self):
        self.assertFalse(hasattr(pg_embed, "psql"))
        source = Path(pg_embed.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("import tempfile", source)

    def test_every_read_takes_the_env_it_is_given(self):
        env = {"PGDATABASE": "elsewhere"}
        with mock.patch.object(pg_embed, "scalar", return_value="0") as scalar_mock:
            pg_embed.pending(env, "citation.work", "true")
        self.assertIs(scalar_mock.call_args[0][0], env)

    def test_the_cli_resolves_a_pgenv_the_way_the_others_do(self):
        with mock.patch.object(pg_embed, "load_pgenv",
                                return_value={"PGDATABASE": "x"}) as load_mock, \
             mock.patch.object(pg_embed, "resolve_target", return_value=("bge-m3", 4)), \
             mock.patch.object(pg_embed, "embed_target", return_value=0), \
             mock.patch.object(pg_embed, "missing_semantic_key", return_value=[]), \
             mock.patch("builtins.print"):
            self.assertEqual(pg_embed.main(["works", "--pgenv", "/tmp/other.pgenv"]), 0)
        self.assertEqual(str(load_mock.call_args[0][0]), "/tmp/other.pgenv")

    def test_an_unreachable_pgenv_is_a_message_not_a_traceback(self):
        with mock.patch.object(pg_embed, "load_pgenv",
                                side_effect=pg_embed.PostgresUnavailable("нет файла")), \
             mock.patch("sys.stderr"):
            self.assertEqual(pg_embed.main([]), 1)

    def test_an_unknown_target_is_refused_before_the_database_is_opened(self):
        with mock.patch.object(pg_embed, "load_pgenv") as load_mock, \
             mock.patch("builtins.print"):
            self.assertEqual(pg_embed.main(["nonesuch"]), 2)
        load_mock.assert_not_called()


class FetchPageTests(unittest.TestCase):
    """The text read here is third-party, and the separators psql prints
    are the only thing standing between one row and two.
    """

    def _page(self, records: list[tuple[str, str]]) -> list[tuple[int, str]]:
        stdout = "".join(f"{i}{FIELD_SEP}{t}{RECORD_SEP}" for i, t in records)
        with mock.patch.object(pg_embed, "run_sql",
                                return_value=mock.Mock(stdout=stdout)):
            return pg_embed.fetch_page({}, "corpus.pages", "body", "true")

    def test_the_page_starts_past_the_cursor_it_is_given(self):
        with mock.patch.object(pg_embed, "run_sql",
                                return_value=mock.Mock(stdout="")) as run_sql:
            pg_embed.fetch_page({}, "corpus.pages", "body", "true", 412)
        sql = run_sql.call_args[0][1]
        self.assertIn("r.id > 412", sql)
        self.assertIn("order by r.id", sql)

    def test_a_row_is_id_and_text(self):
        self.assertEqual(self._page([("7", "Чебышёв")]), [(7, "Чебышёв")])

    def test_a_newline_in_the_text_does_not_become_a_second_row(self):
        self.assertEqual(self._page([("7", "первая\nвторая")]),
                         [(7, "первая\nвторая")])

    def test_an_empty_text_carries_no_vector_and_is_skipped(self):
        self.assertEqual(self._page([("7", "   "), ("8", "есть")]), [(8, "есть")])


class BlankRowsAreExcludedBySqlTests(unittest.TestCase):
    """A row whose rendered text is empty is never updated, so it stays at
    the head of `order by id` and is re-selected on every iteration -- and
    once a whole page of them accumulates, the loop reads the empty page as
    "no work left" and stops with later rows unembedded. Discarding them in
    Python was both of those bugs; the fetch must not ask for them.
    """

    def _sql(self, table: str, text_expr: str, content_pred: str) -> str:
        with mock.patch.object(pg_embed, "run_sql",
                                return_value=mock.Mock(stdout="")) as run_sql:
            pg_embed.fetch_page({}, table, text_expr, content_pred)
        return run_sql.call_args[0][1]

    def test_the_fetch_asks_the_database_to_skip_blank_text(self):
        sql = self._sql("corpus.pages", "body", "btrim(body) <> ''")
        self.assertIn("btrim(t.txt) <> ''", sql)

    def test_every_target_gets_the_same_exclusion_whatever_its_predicate(self):
        """`runs` is the case the classification cannot cover on its own:
        its content predicate is literally "true" (measurements.run has no
        body column), so a row with question/verdict/rules_out/arbiter all
        empty is pending forever.
        """
        for name, (table, text_expr, content_pred) in pg_embed.TARGETS.items():
            with self.subTest(target=name):
                sql = self._sql(table, text_expr, content_pred)
                self.assertIn("btrim(t.txt) <> ''", sql)

    def test_the_text_is_built_once_and_the_filter_reads_that_value(self):
        """Interpolated twice, the expression was evaluated twice for every
        scanned row -- in full for the filter and truncated for the
        projection -- over a pending set the loop re-reads on every
        iteration. The LATERAL computes it once and the filter reads the
        computed column.
        """
        for name, (table, text_expr, content_pred) in pg_embed.TARGETS.items():
            with self.subTest(target=name):
                sql = self._sql(table, text_expr, content_pred)
                self.assertEqual(sql.count(text_expr), 1, sql)

    def test_the_pending_count_still_names_what_has_no_key(self):
        """The count is not narrowed with it: a row that can never carry a
        vector is exactly what the closing "БЕЗ СЕМАНТИЧЕСКОГО КЛЮЧА" line
        reports, and hiding it from the count would hide the report.
        """
        with mock.patch.object(pg_embed, "scalar", return_value="3") as scalar:
            pg_embed.pending({}, "measurements.run", "true")
        self.assertNotIn("btrim", scalar.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
