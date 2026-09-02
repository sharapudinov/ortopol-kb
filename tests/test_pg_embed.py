"""How pg_embed.py spends its round trips.

Two batch sizes, not one: the database is asked for a PAGE of pending rows
(FETCH_BATCH) and ollama is asked for vectors a request at a time
(EMBED_BATCH). Using the ollama size for both meant a psql process, a temp
script and a fresh connection for every 16 rows -- twice, since the UPDATEs
went in a second invocation -- over a table that grows with every crawl.

Nothing here touches Postgres or ollama: psql() and embed() are the two
seams, and both are stubbed so the round trips can be counted.
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401

import pg_embed


class FakeDatabase:
    """A table of `total` pending rows, answering pg_embed's two statements."""

    def __init__(self, total: int):
        self.pending = list(range(1, total + 1))
        self.calls: list[str] = []

    def psql(self, sql: str, tuples_only: bool = True) -> str:
        self.calls.append(sql)
        if sql.lstrip().lower().startswith("select count(*)"):
            return f"{len(self.pending)}\n"
        if sql.lstrip().lower().startswith("select id"):
            limit = int(sql.rsplit("limit", 1)[1].strip(" ;\n"))
            self.page = self.pending[:limit]
            return "".join(f"{row_id}|text {row_id}\n" for row_id in self.page)
        if "update" in sql.lower():
            updated = {int(line.split("where id = ")[1].rstrip(";"))
                       for line in sql.splitlines() if "update" in line.lower()}
            self.pending = [row for row in self.pending if row not in updated]
            return ""
        raise AssertionError(f"unexpected statement: {sql[:80]}")

    @property
    def selects(self) -> list[str]:
        return [c for c in self.calls if c.lstrip().lower().startswith("select id")]

    @property
    def updates(self) -> list[str]:
        return [c for c in self.calls if "update" in c.lower()]


class RoundTripsPerPageTests(unittest.TestCase):
    def _run(self, total: int) -> tuple[FakeDatabase, list[int]]:
        database = FakeDatabase(total)
        chunks: list[int] = []

        def embed(texts, model, dims):
            for start in range(0, len(texts), pg_embed.EMBED_BATCH):
                chunks.append(len(texts[start:start + pg_embed.EMBED_BATCH]))
            return [[0.0] * dims for _ in texts]

        with mock.patch.object(pg_embed, "psql", side_effect=database.psql), \
             mock.patch.object(pg_embed, "embed", side_effect=embed), \
             mock.patch("builtins.print"):
            done = pg_embed.embed_target("works", "bge-m3", 4)
        self.assertEqual(done, total)
        return database, chunks

    def test_a_full_page_costs_one_select_and_one_update(self):
        database, _ = self._run(pg_embed.FETCH_BATCH)
        self.assertEqual(len(database.updates), 1,
                         "вся страница обновляется одним вызовом psql")
        # Two selects: the page, and the one that finds nothing left.
        self.assertEqual(len(database.selects), 2)
        self.assertLessEqual(len(database.calls), 4,
                             "страница в 200 строк — 4 вызова psql, не 26")

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
        self.assertEqual(len(database.updates), 2)
        self.assertEqual(len(database.selects), 3)


if __name__ == "__main__":
    unittest.main()
