#!/usr/bin/env python3
"""Заполняет embedding векторами через локальную ollama.

Идемпотентен: берёт только строки, где embedding IS NULL, поэтому прерванный
прогон продолжается с того же места, а повторный запуск ничего не пересчитывает.

Модель НЕ константа этого файла: она читается из corpus.embedding_model тем же
pg_search.resolve_model(), которым её читают все остальные (поиск, обход
цитирований, смок-проверки пакета). Смешение моделей даёт правдоподобное число,
а не ошибку, и поймать его нечем — колонки модели у строки нет. Пустая таблица
означает «модель ещё не объявлена»: тогда объявляет её этот прогон, вслух и
своими умолчаниями.

Цель `works` — ВТОРОЙ писатель citation.work.embedding, и это добор, а не
конвейер: обход (pg_load_citations.py) пишет вектор кандидата сразу, как только
посчитал его score, а сюда попадают строки, у которых вектора нет — например
после ручной правки title. Текст для вектора и модель обязаны совпасть с
обходовыми, поэтому текст берётся из pg_embedding_text (одно правило в двух
диалектах), а модель — из той же таблицы. Сам запрос к ollama — тоже общий:
pg_search.embed_batch, где живут адрес, размер партии и обе проверки ответа
(размерность и КОЛИЧЕСТВО векторов).

Никаких зависимостей: Postgres через psql — драйвера Postgres в системе нет,
и ради одного скрипта он не заводится.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pg_embedding_text import MAX_CHARS, WORKS_TEXT_SQL
from pg_search import EMBED_BATCH, embed_batch, resolve_model

# Объявляются, только если corpus.embedding_model пуста — см. resolve_target().
DEFAULT_MODEL = "bge-m3"
DEFAULT_DIMS = 1024


def psql(sql: str, tuples_only: bool = True) -> str:
    """Выполняет SQL через psql. Возвращает stdout."""
    args = ["psql", "-v", "ON_ERROR_STOP=1", "-q"]
    if tuples_only:
        args += ["-tA"]
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as fh:
        fh.write(sql)
        path = fh.name
    try:
        r = subprocess.run(args + ["-f", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"psql: {r.stderr.strip()}")
        return r.stdout
    finally:
        Path(path).unlink(missing_ok=True)


def resolve_target(env: dict[str, str]) -> tuple[str, int]:
    """(model, dims) для этого прогона — из corpus.embedding_model.

    Тот же читатель, что у pg_search.embed_query() и у обхода: одна таблица
    решает, какой моделью посчитаны ВСЕ векторы базы. Пустая таблица — это не
    «возьми что хочешь», а «ещё никто не объявил»; тогда прогон объявляет
    умолчания вслух, и следующий читатель получит уже их.
    """
    resolved = resolve_model(env)
    if resolved is not None:
        return resolved
    print(f"corpus.embedding_model пуста — объявляю {DEFAULT_MODEL}/{DEFAULT_DIMS}; "
          "все дальнейшие читатели прочтут эту пару")
    psql("insert into corpus.embedding_model (id, model, dims) values "
         f"(1, '{DEFAULT_MODEL}', {DEFAULT_DIMS}) on conflict (id) do nothing;",
         tuples_only=False)
    return DEFAULT_MODEL, DEFAULT_DIMS


def embed(texts: list[str], model: str, dims: int) -> list[list[float]]:
    """Векторы для текстов, через общий шов запроса (pg_search.embed_batch).

    Своего HTTP здесь нет намеренно: у citation.work.embedding два писателя,
    и запрос к ollama у них обязан быть одним. Второй экземпляр проверял
    только размерность — и молча писал меньше строк, чем собирался, когда
    ollama возвращала меньше векторов, чем текстов (zip обрезает по
    короткому). Обе проверки, адрес и размер партии живут в pg_search.
    """
    return embed_batch(model, dims, texts)


# Каждый вид записи, который может стать результатом, обязан нести семантический
# ключ — иначе он находится только тем, кто уже знает нужное слово. Добавление
# нового вида записи означает добавление строки СЮДА, а не отдельного скрипта.
TARGETS = {
    # имя: (таблица, выражение текста для вектора, предикат «есть содержание»)
    # Предикат нужен, потому что пустая страница семантического содержания не несёт
    # и ключ ей не положен, а у measurements.run колонки body нет вовсе.
    "pages": ("corpus.pages",
              "replace(replace(body, E'\\n', ' '), E'\\r', ' ')",
              "btrim(body) <> ''"),
    "runs": (
        "measurements.run",
        # Смысл прогона — в вопросе, вердикте и в том, что он ИСКЛЮЧАЕТ.
        # Переводы строк внутри полей вычищаются, как у pages: парсер ниже
        # построчный, и многострочный вердикт иначе рвёт разбор id|text.
        "replace(replace(coalesce(question,'')||' '||coalesce(verdict,'')||' '"
        "||coalesce(rules_out,'')||' '||coalesce(arbiter,''), E'\n', ' '), E'\r', ' ')",
        "true",
    ),
    "works": (
        "citation.work",
        # Скелет из внешнего источника несёт заголовок и аннотацию — вместе
        # они и есть смысл записи для фильтра фронтира (косинус к центроиду
        # семян). Выражение НЕ пишется здесь: у этой колонки два писателя, и
        # правило текста одно на обоих (pg_embedding_text.WORKS_TEXT_SQL и
        # его питоновский двойник works_text). Предикат ниже — как у pages:
        # запись без непустого заголовка семантического содержания не несёт.
        WORKS_TEXT_SQL,
        "btrim(coalesce(title,'')) <> ''",
    ),
}


def pending(table: str, content_pred: str = "true") -> int:
    return int(psql(
        f"select count(*) from {table} "
        f"where embedding is null and ({content_pred});").strip())


def missing_semantic_key() -> list[tuple[str, int]]:
    """Записи-результаты без семантического ключа. Пустой список — инвариант держится."""
    return [
        (name, n)
        for name, (table, _, pred) in TARGETS.items()
        if (n := pending(table, pred)) > 0
    ]


def embed_target(name: str, model: str, dims: int) -> int:
    table, text_expr, content_pred = TARGETS[name]
    total = pending(table, content_pred)
    if total == 0:
        print(f"{name}: все записи уже несут семантический ключ")
        return 0
    print(f"{name}: к обсчёту {total}, модель {model}, партиями по {EMBED_BATCH}")

    done, started = 0, time.monotonic()
    while True:
        rows = psql(
            f"select id, left({text_expr}, {MAX_CHARS}) from {table} "
            f"where embedding is null and ({content_pred}) "
            f"order by id limit {EMBED_BATCH};"
        ).strip()
        if not rows:
            break
        pairs = []
        for line in rows.split("\n"):
            pid, _, text = line.partition("|")
            if text.strip():
                pairs.append((int(pid), text))
        if not pairs:
            # Все оставшиеся записи с пустым текстом — вектор им не из чего строить.
            break

        vecs = embed([t for _, t in pairs], model, dims)
        updates = "\n".join(
            f"update {table} set embedding = '{json.dumps(v)}' where id = {pid};"
            for (pid, _), v in zip(pairs, vecs)
        )
        psql("begin;\n" + updates + "\ncommit;")

        done += len(pairs)
        rate = done / max(time.monotonic() - started, 1e-9)
        print(f"  {name}: {done}/{total}  ({rate:.1f} зап/с)", flush=True)
    return done


def main() -> int:
    which = sys.argv[1:] or list(TARGETS)
    for name in which:
        if name not in TARGETS:
            print(f"неизвестная цель: {name}; известны {list(TARGETS)}")
            return 2
    # Один раз на прогон: таблица несёт ровно одну строку (CHECK (id = 1)),
    # и модель не может смениться посреди обсчёта.
    model, dims = resolve_target(dict(os.environ))
    for name in which:
        embed_target(name, model, dims)

    gaps = missing_semantic_key()
    if gaps:
        # Не падаем: пустой текст встречается легитимно. Но молчать нельзя —
        # запись без ключа находится только тем, кто знает точное слово.
        print("БЕЗ СЕМАНТИЧЕСКОГО КЛЮЧА:", ", ".join(f"{n}: {c}" for n, c in gaps))
    else:
        print("все записи-результаты несут семантический ключ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
