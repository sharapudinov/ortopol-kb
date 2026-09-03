"""The one rule for citation.work's embedded text, in both dialects.

citation.work.embedding has two producers -- the crawl (Python) and
pg_embed.py (SQL) -- and no per-row column recording which text or which
model made a vector. A disagreement between them is therefore invisible:
the cosine it produces is plausible, not wrong-looking. The live class
below holds the two spellings to each other over a FIXTURE it seeds
itself -- the awkward rows spelled out, so the comparison is the same on
every instance and an empty corpus is no reason for it to have nothing to
say. A second live class repeats it over the real table, where it is a
bonus: real titles carry real newlines, but how many is data, and a corpus
without them is not a defect. The pure class holds the Python side to the
rule itself.
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
from pg_common import (PostgresUnavailable, check_postgres_available, load_pgenv,
                       run_sql, sql_literal)
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


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except (PostgresUnavailable, RuntimeError) as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


# Every shape the two spellings can disagree over, named rather than hoped
# for: a newline and a carriage return inside a title (pg_embed.py parses
# its rows record by record, and psql's own separators are not the only
# thing str.splitlines() breaks on), tabs and spaces at both edges, each
# part missing in turn, both missing, and a pair long enough to be cut.
FIXTURE = [
    ("f:plain", "Discrete Chebyshev polynomials", "An estimate on a grid."),
    ("f:newline", "Title\nsecond line", "Abstract\r\nsecond line"),
    ("f:tabs", "\tTitle\t", "  Abstract  "),
    ("f:no-abstract", "Title only", None),
    ("f:no-title", None, "Abstract only"),
    ("f:neither", None, None),
    ("f:blank", "   ", "\t\n "),
    ("f:long", "ч" * (MAX_CHARS - 10), "щ" * 100),
]


class BothProducersAgreeOnAFixtureTests(unittest.TestCase):
    """The SQL side is only checkable against a database -- but not against
    the operator's corpus. The rows are seeded here, inside a transaction
    that is rolled back, so the comparison says the same thing on a fresh
    instance as on this one, and a disagreement names the shape that caused
    it instead of a key nobody can look up.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    def _fixture_rows(self) -> list[dict]:
        values = ",\n".join(
            f"({sql_literal(key)}, "
            f"{'NULL' if title is None else sql_literal(title)}, "
            f"{'NULL' if abstract is None else sql_literal(abstract)})"
            for key, title, abstract in FIXTURE)
        out = run_sql(
            self.env,
            "BEGIN;\n"
            "CREATE TEMP TABLE fixture_work (key TEXT, title TEXT, abstract TEXT)\n"
            "  ON COMMIT DROP;\n"
            f"INSERT INTO fixture_work (key, title, abstract) VALUES\n{values};\n"
            "SELECT coalesce(json_agg(row ORDER BY key), '[]'::json) FROM ("
            "  SELECT key, json_build_object("
            f"   'key', key, 'sql', left({WORKS_TEXT_SQL}, {MAX_CHARS})) AS row"
            "  FROM fixture_work) t;\n"
            "ROLLBACK;\n",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        return json.loads([line for line in out.splitlines() if line.strip()][-1])

    def test_the_sql_expression_and_works_text_agree_on_every_shape(self):
        produced = {row["key"]: row["sql"] for row in self._fixture_rows()}
        self.assertEqual(len(produced), len(FIXTURE), "фикстура прочитана не вся")
        for key, title, abstract in FIXTURE:
            with self.subTest(shape=key):
                self.assertEqual(produced[key], works_text(title, abstract))

    def test_the_cut_is_the_same_cut_on_both_sides(self):
        produced = {row["key"]: row["sql"] for row in self._fixture_rows()}
        self.assertEqual(len(produced["f:long"]), MAX_CHARS)


class BothProducersAgreeOnTheCorpusTests(unittest.TestCase):
    """The same comparison over the real table, as a bonus over the
    fixture: real titles carry real newlines (measured: 5 of 438), and
    those are exactly where a hand-written second spelling drifts. HOW MANY
    such rows exist is data, though, so an instance without them skips
    rather than fails -- the contract itself is held by the fixture above.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

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
        if not rows:
            raise unittest.SkipTest("citation.work пуста: сверять нечего")
        differing = [row["key"] for row in rows
                     if works_text(row["title"], row["abstract"]) != row["sql"]]
        self.assertEqual(differing, [], f"разошлись на {len(differing)} строках")

    def test_the_rows_with_newlines_are_among_them(self):
        """Otherwise the agreement is only over the easy rows -- but a
        corpus that happens to carry none is not a defect.
        """
        tricky = [row for row in self._rows()
                  if any(ch in (row["title"] or "") + (row["abstract"] or "")
                         for ch in "\n\r\t")]
        if not tricky:
            raise unittest.SkipTest("в таблице нет строк с переводами строк")
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
        # Including the cut: a candidate long enough to be truncated is
        # truncated by the shared rule, not by a length this module keeps.
        long_title = "ч" * (MAX_CHARS + 100)
        cut = frontier.candidate_text(long_title, "abstract")
        self.assertEqual(cut, works_text(long_title, "abstract"))
        self.assertEqual(len(cut), pg_embedding_text.MAX_CHARS)

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

    def test_the_declared_default_is_spliced_through_sql_literal(self):
        """The declaration an empty corpus.embedding_model provokes is a
        write, and the one place this repository quotes a value into a
        statement is pg_common.sql_literal() -- the module docstring it
        carries names the f-string as the alternative it exists to replace.
        """
        with mock.patch.object(pg_embed, "resolve_model", return_value=None), \
             mock.patch.object(pg_embed, "run_sql") as run_mock, \
             mock.patch("builtins.print"):
            self.assertEqual(pg_embed.resolve_target({}),
                             (pg_embed.DEFAULT_MODEL, pg_embed.DEFAULT_DIMS))
        statement = run_mock.call_args.args[1]
        self.assertIn(sql_literal(pg_embed.DEFAULT_MODEL), statement)
        self.assertNotIn(f" '{pg_embed.DEFAULT_MODEL}'", statement,
                         "значение вклеено f-строкой, а не sql_literal()")

    def test_pg_embed_resolves_the_model_it_does_not_fix_one(self):
        """The constants it kept were a second resolution of the pair the
        rest of the repository reads out of corpus.embedding_model.
        """
        self.assertFalse(hasattr(pg_embed, "MODEL"))
        self.assertFalse(hasattr(pg_embed, "DIMS"))
        self.assertIs(pg_embed.resolve_model, pg_search.resolve_model)


if __name__ == "__main__":
    unittest.main()
