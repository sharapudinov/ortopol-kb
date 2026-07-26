"""Покрытие корпуса: индекс против реального числа страниц в PDF.

До появления этого теста «63 документа проиндексированы» звучало как «корпус
покрыт», хотя 92 страницы отсутствовали целиком, а частичная потеря страниц
ВНУТРИ документа не ловилась вообще ничем. `pdfinfo` — независимый и дешёвый
арбитр: он читает структуру PDF, а не текстовый слой, поэтому не разделяет
слепых пятен с извлекателем.
"""
import re
import subprocess
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
from paths import default_corpus_dir, default_pdf_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"


class CorpusCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")
        cls.pdf_dir = default_pdf_dir()
        rows = run_sql(
            cls.env,
            f"SELECT filename, pages_count, extraction_state FROM corpus.documents "
            f"ORDER BY filename;",
            extra_args=["-t", "-A", "-F", FIELD_SEP],
        ).stdout
        cls.docs = [r.split(FIELD_SEP) for r in rows.splitlines() if r.strip()]

    @staticmethod
    def _pdf_pages(path: Path) -> int | None:
        out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        return int(m.group(1)) if m else None

    def test_every_pdf_has_a_document_row(self):
        pdfs = {p.name for p in self.pdf_dir.glob("*.pdf")}
        indexed = {d[0] for d in self.docs}
        self.assertEqual(pdfs - indexed, set(), "PDF без строки в corpus.documents")

    def test_page_counts_match_pdfinfo(self):
        # Ловит и полное отсутствие документа, и молчаливую потерю страниц внутри
        # него — второе опаснее, потому что снаружи выглядит как успех.
        mismatches = []
        for filename, pages_count, state in self.docs:
            real = self._pdf_pages(self.pdf_dir / filename)
            if real is None:
                continue
            if real != int(pages_count):
                mismatches.append(f"{filename}: PDF {real}, индекс {pages_count} ({state})")
        self.assertEqual(mismatches, [], "\n".join(mismatches))

    def test_coverage_is_total(self):
        # Сумма и по индексу, и по источнику — ТОЛЬКО для документов, чей файл
        # pdfinfo умеет прочитать. Иначе арифметика лжёт в обе стороны: djvu и
        # metadata раздували «индекс» (110.6% покрытия — тоже провал проверки,
        # только с другим знаком). Полнота по ВСЕМ типам — corpus_completeness.py.
        indexed = real = 0
        for filename, pages_count, _state in self.docs:
            n = self._pdf_pages(self.pdf_dir / filename)
            if n is None:
                continue
            indexed += int(pages_count)
            real += n
        self.assertEqual(indexed, real, f"покрытие {100 * indexed / real:.1f}%, ожидалось 100%")


class TranscribedDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def _scalar(self, sql: str) -> str:
        return run_sql(self.env, sql, extra_args=["-t", "-A"]).stdout.strip()

    def test_transcribed_state_is_distinct_from_clean(self):
        # Транскрипция — чтение изображения моделью, извлечение — механическое.
        # Слить их в один статус значит потерять различие безвозвратно: ошибка
        # в индексе формулы выглядит ровно как верный индекс.
        # 4 статьи с нечитаемым текстовым слоем + монография (заменён ненадёжный OCR-слой)
        n = self._scalar(
            "SELECT count(*) FROM corpus.documents WHERE extraction_state = 'transcribed';")
        self.assertEqual(n, "5")

    def test_transcribed_documents_have_pages(self):
        empty = self._scalar(
            "SELECT count(*) FROM corpus.documents d WHERE d.extraction_state = 'transcribed' "
            "AND NOT EXISTS (SELECT 1 FROM corpus.pages p WHERE p.document_id = d.id);")
        self.assertEqual(empty, "0")

    def test_transcribed_note_states_the_limitation(self):
        # Ограничение должно быть в БАЗЕ, а не только в отчёте: тот, кто найдёт
        # страницу поиском, увидит note, а отчёт читать не будет.
        bad = self._scalar(
            "SELECT count(*) FROM corpus.documents WHERE extraction_state = 'transcribed' "
            "AND note NOT ILIKE '%transcribed%';")
        self.assertEqual(bad, "0")

    def test_transcribed_pages_are_searchable(self):
        hit = self._scalar(
            "SELECT count(*) FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id "
            "WHERE d.extraction_state = 'transcribed' "
            "AND p.tsv @@ phraseto_tsquery('russian', 'смешанные ряды');")
        self.assertGreater(int(hit), 0, "транскрибированные страницы не находятся полнотекстом")

    def test_transcribed_pages_carry_semantic_key(self):
        missing = self._scalar(
            "SELECT count(*) FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id "
            "WHERE d.extraction_state = 'transcribed' AND p.embedding IS NULL;")
        self.assertEqual(missing, "0")


if __name__ == "__main__":
    unittest.main()


class CompletenessScriptTests(unittest.TestCase):
    """Полнота корпуса — предикат, а не впечатление: скрипт перечисляет КАЖДЫЙ файл
    под theory/, падает на неклассифицированном и сверяет страницы с источником."""

    def test_completeness_script_passes(self):
        import subprocess as sp
        r = sp.run(["python3", str(Path(__file__).resolve().parents[1] / "corpus_completeness.py")],
                   capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"полнота не доказана:\n{r.stdout}\n{r.stderr}")


class AuxSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.env = load_pgenv(default_corpus_dir() / ".pgenv")
        except PostgresUnavailable as exc:
            raise unittest.SkipTest(f"Postgres not configured: {exc}")
        if not check_postgres_available(cls.env):
            raise unittest.SkipTest("Postgres not reachable")

    def _scalar(self, sql):
        return run_sql(self.env, sql, extra_args=["-t", "-A"]).stdout.strip()

    def test_monograph_is_indexed_with_all_pages(self):
        # Была загружена OCR-слоем (state 'ocr'), затем полностью транскрибирована.
        row = self._scalar(
            "SELECT extraction_state || ':' || pages_count FROM corpus.documents "
            "WHERE id = 'Sharapudinov_I_I_Mnogochleny__ortogonal_nye_na_setkah_Mahachkala';")
        self.assertEqual(row, "transcribed:234", "монография не транскрибирована целиком")

    def test_monograph_is_searchable(self):
        # Глава 5 монографии — многочлены Кравчука; OCR-слой обязан находиться стеммингом.
        hits = self._scalar(
            "SELECT count(*) FROM corpus.pages p "
            "WHERE p.document_id = 'Sharapudinov_I_I_Mnogochleny__ortogonal_nye_na_setkah_Mahachkala' "
            "AND p.tsv @@ to_tsquery('russian', 'Кравчука');")
        self.assertGreater(int(hits), 0)

    def test_every_document_carries_its_source_blob(self):
        # База самодостаточна: пакет (задача 031) разворачивается без theory/,
        # и сверка транскрипции с изображением возможна на любой машине.
        # Целостность хеша диск<->база проверяет corpus_completeness.py.
        bad = self._scalar(
            "SELECT count(*) FROM corpus.documents "
            "WHERE source_blob IS NULL OR length(source_blob) = 0 "
            "OR source_sha256 IS NULL OR source_sha256 = '';")
        self.assertEqual(bad, "0", "документ без исходника-блоба или без хеша")

    def test_metadata_files_are_indexed(self):
        ids = self._scalar(
            "SELECT string_agg(id, ',' ORDER BY id) FROM corpus.documents "
            "WHERE extraction_state = 'metadata';")
        self.assertEqual(ids, "INDEX,THEMES")

    def test_ordinary_reload_preserves_aux_documents(self):
        # Тот же класс дефекта ловили дважды: обычная перезагрузка не имеет права
        # трогать документы чужих конвейеров (transcribed/ocr/metadata).
        from pg_load import load_corpus
        load_corpus(default_corpus_dir(), default_corpus_dir() / ".pgenv")
        n = self._scalar(
            "SELECT count(*) FROM corpus.documents "
            "WHERE extraction_state IN ('transcribed', 'ocr', 'metadata');")
        self.assertEqual(n, "7", "перезагрузка уничтожила aux-документы (4+1+2)")
