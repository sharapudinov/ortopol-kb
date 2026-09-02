"""The one rule for citation.work's embedded text, in both dialects.

citation.work.embedding has two producers -- the crawl (Python) and
pg_embed.py (SQL) -- and no per-row column recording which text or which
model made a vector. A disagreement between them is therefore invisible:
the cosine it produces is plausible, not wrong-looking. The live class
below holds the two spellings to each other over the real table, including
the rows that carry newlines and tabs; the pure class holds the Python side
to the rule itself.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import pg_embed
import pg_embedding_text
import pg_search
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql
from pg_embedding_text import MAX_CHARS, WORKS_TEXT_SQL, works_text


class WorksTextRuleTests(unittest.TestCase):
    def test_the_two_parts_are_joined_by_one_space(self):
        self.assertEqual(works_text("Title", "Abstract"), "Title Abstract")

    def test_a_missing_part_leaves_no_stray_space(self):
        self.assertEqual(works_text("Title", None), "Title")
        self.assertEqual(works_text(None, "Abstract"), "Abstract")
        self.assertEqual(works_text("", ""), "")

    def test_newlines_and_tabs_do_not_survive_the_edges(self):
        self.assertEqual(works_text("  Title\nsecond line\t", "\tAbstract\r\n"),
                         "Title second line Abstract")

    def test_the_cut_is_at_max_chars(self):
        self.assertEqual(len(works_text("x" * 5000, "y" * 5000)), MAX_CHARS)

    def test_the_sql_expression_is_not_pre_truncated(self):
        """pg_embed.py wraps every target's expression in left(..., N); a
        left() baked in here would be applied twice and read as the rule.
        """
        self.assertNotIn("left(", WORKS_TEXT_SQL)


class BothProducersAgreeTests(unittest.TestCase):
    """The SQL side is only checkable against a database, and the rows that
    matter are the real ones: titles carrying newlines exist in the live
    table (measured: 5 of 438), and they are exactly where a hand-written
    second spelling drifts.
    """

    @classmethod
    def setUpClass(cls):
        try:
            env = load_pgenv(default_corpus_dir() / ".pgenv")
        except (PostgresUnavailable, RuntimeError) as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(env):
            raise unittest.SkipTest("Postgres not reachable")
        cls.env = env

    def _rows(self) -> list[dict]:
        out = run_sql(
            self.env,
            "SELECT coalesce(json_agg(row), '[]'::json) FROM ("
            "  SELECT json_build_object("
            "    'key', key, 'title', title, 'abstract', abstract,"
            f"   'sql', left({WORKS_TEXT_SQL}, {MAX_CHARS})) AS row"
            "  FROM citation.work ORDER BY key) t;",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        return json.loads(out)

    def test_the_sql_expression_and_works_text_agree_on_every_row(self):
        rows = self._rows()
        self.assertGreater(len(rows), 20, "таблица пуста — сравнивать нечего")
        differing = [row["key"] for row in rows
                     if works_text(row["title"], row["abstract"]) != row["sql"]]
        self.assertEqual(differing, [], f"разошлись на {len(differing)} строках")

    def test_the_rows_with_newlines_are_among_them(self):
        """Otherwise the agreement is only over the easy rows."""
        rows = self._rows()
        tricky = [row for row in rows
                  if any(ch in (row["title"] or "") + (row["abstract"] or "")
                         for ch in "\n\r\t")]
        self.assertTrue(tricky, "в таблице нет строк с переводами строк")
        for row in tricky:
            self.assertEqual(works_text(row["title"], row["abstract"]), row["sql"],
                             row["key"])


class SingleContractTests(unittest.TestCase):
    """Neither producer is allowed to hold its own copy of the rule."""

    def test_pg_embed_takes_the_works_expression_from_the_shared_module(self):
        self.assertIs(pg_embed.TARGETS["works"][1], WORKS_TEXT_SQL)

    def test_the_crawl_takes_the_same_rule(self):
        from citations import frontier

        self.assertEqual(frontier.candidate_text("A\nB", " C "),
                         works_text("A\nB", " C "))
        self.assertIs(frontier.MAX_CHARS, pg_embedding_text.MAX_CHARS)

    def test_pg_embed_asks_ollama_through_the_shared_seam(self):
        """One column, two writers, one request. The local implementation
        checked the width and not the count, so a short answer from ollama
        silently updated fewer rows than it was given texts for.
        """
        with mock.patch.object(pg_embed, "embed_batch",
                               return_value=[[0.0]]) as seam:
            self.assertEqual(pg_embed.embed(["text"], "bge-m3", 1024), [[0.0]])
        seam.assert_called_once_with("bge-m3", 1024, ["text"])
        self.assertIs(pg_embed.embed_batch, pg_search.embed_batch)
        self.assertIs(pg_embed.EMBED_BATCH, pg_search.EMBED_BATCH)

    def test_pg_embed_holds_no_http_client_of_its_own(self):
        source = Path(pg_embed.__file__).read_text(encoding="utf-8")
        self.assertFalse(hasattr(pg_embed, "OLLAMA"))
        self.assertFalse(hasattr(pg_embed, "BATCH"))
        imported = {alias.name.split(".")[0]
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in getattr(node, "names", [])}
        self.assertNotIn("urllib", imported, "второй HTTP-клиент к ollama")
        self.assertNotIn("api/embed", source)

    def test_pg_embed_resolves_the_model_it_does_not_fix_one(self):
        """The constants it kept were a second resolution of the pair the
        rest of the repository reads out of corpus.embedding_model.
        """
        self.assertFalse(hasattr(pg_embed, "MODEL"))
        self.assertFalse(hasattr(pg_embed, "DIMS"))
        self.assertIs(pg_embed.resolve_model, pg_search.resolve_model)


if __name__ == "__main__":
    unittest.main()
