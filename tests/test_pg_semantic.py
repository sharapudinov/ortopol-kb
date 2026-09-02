"""Integration tests for the semantic half of corpus search.

Skips rather than fails when Postgres or the embedding service is unreachable,
so the storage-agnostic part of the suite still runs elsewhere.
"""
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import pg_embed
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql
from pg_embed import TARGETS
from pg_search import embed_query, search


class SemanticSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = default_corpus_dir()
        try:
            cls.env = load_pgenv(cls.corpus_dir / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def _scalar(self, sql):
        return run_sql(self.env, sql, extra_args=["-t", "-A"]).stdout.strip()

    def test_every_page_carries_a_semantic_key(self):
        # A record without an embedding is findable only by someone who already
        # knows the exact word to search for — which is the case retrieval exists
        # to solve. Pages with a body must therefore all be embedded.
        missing = self._scalar(
            "SELECT count(*) FROM corpus.pages WHERE embedding IS NULL;")
        self.assertEqual(missing, "0", "страницы без вектора")

    def test_measurement_runs_carry_a_semantic_key_too(self):
        # Same rule for results, not just for sources: a spike verdict that is
        # not semantically findable gets re-derived by the next spike.
        missing = self._scalar(
            "SELECT count(*) FROM measurements.run WHERE embedding IS NULL;")
        self.assertEqual(missing, "0", "прогоны без вектора")

    def test_embedding_model_is_recorded(self):
        # Mixing models is otherwise undetectable: cosine distance between two
        # unrelated vector spaces computes fine and means nothing.
        row = self._scalar(
            "SELECT model || ' ' || dims FROM corpus.embedding_model WHERE id = 1;")
        self.assertTrue(row, "модель эмбеддингов не записана в базе")

    def test_stored_dimension_matches_the_recorded_model(self):
        mismatched = self._scalar(
            "SELECT count(*) FROM corpus.pages p, corpus.embedding_model m "
            "WHERE m.id = 1 AND p.embedding IS NOT NULL "
            "AND vector_dims(p.embedding) <> m.dims;")
        self.assertEqual(mismatched, "0", "размерность вектора не совпадает с записанной")

    def test_vector_finds_what_fulltext_cannot(self):
        # The whole justification for stage 4. A query phrased in ordinary words,
        # sharing no term with the corpus wording: full-text must miss it, vector
        # must not. If this ever passes for full-text too, the query stopped being
        # a fair test and needs rewording, not the assertion relaxing.
        query = "потеря точности вблизи концов промежутка"
        if embed_query(query, self.env) is None:
            self.skipTest("сервис эмбеддингов недоступен")
        self.assertEqual(search(query, self.env, mode="fulltext"), [])
        self.assertTrue(search(query, self.env, mode="vector"))

    def test_degraded_documents_are_indexed_not_discarded(self):
        # They were dropped entirely over ~0.3% of words missing an ff/fi ligature,
        # which cost the index a source for the Sobolev-operator question.
        pages = self._scalar(
            "SELECT count(*) FROM corpus.pages p JOIN corpus.documents d "
            "ON d.id = p.document_id WHERE d.extraction_state = 'degraded';")
        self.assertGreater(int(pages), 0, "degraded-документы снова выпали из индекса")

    def test_no_document_is_indexed_with_zero_pages(self):
        # Раньше проверялось «нечитаемых ровно 4» — число устарело после транскрипции.
        # Инвариант тот же и сформулирован прямо: документ без страниц делает отчёт
        # о покрытии ложным, потому что снаружи он выглядит присутствующим.
        empty = self._scalar(
            "SELECT count(*) FROM corpus.documents d WHERE NOT EXISTS "
            "(SELECT 1 FROM corpus.pages p WHERE p.document_id = d.id);")
        self.assertEqual(empty, "0", "документ в индексе без единой страницы")


class WorksTargetRegistrationTests(unittest.TestCase):
    """No live database needed: TARGETS is a plain module-level dict.

    Registration (EXTENDING.md § "Процедура E" step 2) is itself the check
    that citation.work is a findable record kind, so this asserts the
    dict entry directly rather than round-tripping through Postgres.
    """

    def test_works_target_registered_with_content_predicate(self):
        self.assertIn("works", TARGETS, "citation.work не зарегистрирован в TARGETS")
        table, text_expr, content_pred = TARGETS["works"]
        self.assertEqual(table, "citation.work")
        # A record with no title carries no semantic content — same rule as
        # corpus.pages's "btrim(body) <> ''" — so the predicate must exclude it,
        # not embed an empty/whitespace string.
        self.assertIn("title", content_pred)
        self.assertIn("title", text_expr)
        self.assertIn("abstract", text_expr)


class PendingEmbeddingIndexTests(unittest.TestCase):
    """The добор reads "rows with no vector" over and over; nothing served
    that predicate, and the crawl makes almost every row a non-NULL one.
    """

    INDEX = "work_pending_embedding_idx"

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except (PostgresUnavailable, RuntimeError) as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def test_the_schema_declares_it_idempotently(self):
        schema = (Path(pg_embed.__file__).resolve().parent
                  / "pg_schema_citation.sql").read_text(encoding="utf-8")
        self.assertIn(f"CREATE INDEX IF NOT EXISTS {self.INDEX}", schema)
        self.assertIn("WHERE embedding IS NULL", schema)

    def test_the_pending_read_uses_it_on_the_live_table(self):
        """One pending row, inside a transaction that is rolled back: the
        plan is read on the real table rather than on a claim about it.
        """
        table, text_expr, content_pred = TARGETS["works"]
        plan = run_sql(
            self.env,
            "BEGIN;\n"
            "INSERT INTO citation.work (key, title, source, kind) VALUES "
            "('test:pending-embedding-plan', 'Chebyshev pending row', "
            "'manual', 'external-skeleton');\n"
            "EXPLAIN (COSTS OFF) SELECT id, left(" + text_expr + ", 100) "
            f"FROM {table} WHERE embedding IS NULL AND ({content_pred}) "
            "ORDER BY id LIMIT 16;\n"
            "ROLLBACK;",
            extra_args=["-t", "-A"],
        ).stdout
        self.assertIn(self.INDEX, plan, plan)

    def test_the_rolled_back_row_left_nothing_behind(self):
        left = run_sql(
            self.env,
            "SELECT count(*) FROM citation.work "
            "WHERE key = 'test:pending-embedding-plan';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(left, "0")


if __name__ == "__main__":
    unittest.main()
