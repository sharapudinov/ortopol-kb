"""Integration test for exact-phrase full-text search over Postgres.

Skips (rather than fails) if Postgres isn't reachable, so the rest of the
suite (which is storage-agnostic) still runs in an environment without the
corpus database configured.
"""
import unittest

import _pathfix  # noqa: F401
from build_corpus import build_corpus, write_manifest
from paths import default_corpus_dir, default_pdf_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql
from pg_load import load_corpus
from pg_search import search


class FullTextSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = default_corpus_dir()
        cls.pgenv_path = cls.corpus_dir / ".pgenv"
        try:
            cls.env = load_pgenv(cls.pgenv_path)
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable at configured host/port")

        extractions = build_corpus(default_pdf_dir(), cls.corpus_dir)
        write_manifest(extractions, cls.corpus_dir)
        load_corpus(cls.corpus_dir, cls.pgenv_path)

    def test_full_text_search_finds_known_phrase(self):
        hits = search("повторные средние", self.env)
        self.assertIn("2019_smj3104", {doc_id for doc_id, _, _ in hits})

    def test_full_text_search_finds_second_phrase(self):
        hits = search("прилипания", self.env)
        self.assertIn("2015_demr1", {doc_id for doc_id, _, _ in hits})

    def test_formerly_unreadable_documents_are_present(self):
        # Раньше здесь проверялось «нечитаемых ровно 4». Число устарело: все четыре
        # транскрибированы с отрисовки. Инвариант за ним остаётся — ни один документ
        # не исчезает из индекса молча, иначе запрос о покрытии соврёт.
        result = run_sql(
            self.env,
            "SELECT d.id FROM corpus.documents d WHERE d.id IN "
            "('1996_sm105','1997_sm280','2000_sm480','2003_sm723') "
            "AND EXISTS (SELECT 1 FROM corpus.pages p WHERE p.document_id = d.id) "
            "ORDER BY d.id;",
            extra_args=["-t", "-A"],
        )
        ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        self.assertEqual(len(ids), 4, "документ из бывших нечитаемых потерял страницы")
