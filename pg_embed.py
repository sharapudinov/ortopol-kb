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
и ради одного скрипта он не заводится. Ровно один путь к базе: pg_common
(run_sql/scalar/copy_csv_rows) с ЯВНЫМ env, и `--pgenv`, как у
pg_load_citations.py и pg_graph.py. Свой subprocess-вызов рядом с ними брал
подключение из окружения молча, и цель `works` — писатель citation.work —
никуда, кроме базы по умолчанию, направлена быть не могла.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from paths import default_corpus_dir
from pg_common import (
    FIELD_SEP,
    PostgresUnavailable,
    ROW_ARGS,
    load_pgenv,
    run_sql,
    scalar,
    split_records,
    sql_literal,
    vector_literal,
)
from pg_copy import copy_csv_rows
from pg_embedding_text import MAX_CHARS, WORKS_TEXT_SQL
from pg_search import EMBED_BATCH, embed_batch, resolve_model

# Объявляются, только если corpus.embedding_model пуста — см. resolve_target().
DEFAULT_MODEL = "bge-m3"
DEFAULT_DIMS = 1024

# Сколько строк берётся у БАЗЫ за один заход. Размер партии к ollama
# (EMBED_BATCH = 16) для этого не годится: у psql на каждый вызов процесс,
# временный скрипт и новое соединение, и цель `works` растёт с каждым
# обходом. Та же пара размеров, что у citations/frontier.vectors_for
# («~22 round trips вместо ~267»): страница из базы крупная, запрос к
# ollama мелкий, и UPDATE'ы всей страницы уходят одним вызовом.
FETCH_BATCH = 200


# Куда уходит посчитанная страница: во временную таблицу одним \copy, и
# UPDATE читает её же в том же скрипте (pg_copy.copy_csv_rows). Раньше это
# был склеенный текст из 200 отдельных UPDATE'ов, каждый с вектором в 1024
# числа внутри строкового литерала — тот самый способ, ради отказа от
# которого citations/store.py и завёл этот шов.
_STAGE_DDL = """
CREATE TEMP TABLE stage_embedding (id BIGINT, embedding TEXT) ON COMMIT DROP;
"""


def _update_sql(table: str) -> str:
    return f"""
WITH updated AS (
UPDATE {table} t SET embedding = s.embedding::vector
FROM stage_embedding s WHERE t.id = s.id
RETURNING 1)
SELECT count(*) FROM updated;
"""


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
    run_sql(env, "insert into corpus.embedding_model (id, model, dims) values "
                 f"(1, {sql_literal(DEFAULT_MODEL)}, {int(DEFAULT_DIMS)}) "
                 "on conflict (id) do nothing;")
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


def pending(env: dict[str, str], table: str, content_pred: str = "true") -> int:
    return int(scalar(env, f"select count(*) from {table} "
                           f"where embedding is null and ({content_pred});"))


def missing_semantic_key(env: dict[str, str]) -> list[tuple[str, int]]:
    """Записи-результаты без семантического ключа. Пустой список — инвариант держится."""
    return [
        (name, n)
        for name, (table, _, pred) in TARGETS.items()
        if (n := pending(env, table, pred)) > 0
    ]


def fetch_page(env: dict[str, str], table: str, text_expr: str,
               content_pred: str) -> list[tuple[int, str]]:
    """Страница ожидающих строк как (id, текст), обрезанный до MAX_CHARS.

    Разбор общий (ROW_ARGS/split_records), а не построчный по '|': текст
    здесь — чужой (тело страницы, заголовок работы), а str.splitlines()
    считает границей строки не только \n (pg_common's own comment на
    FIELD_SEP/RECORD_SEP). Разделитель в тексте больше не рвёт строку на
    две записи.
    """
    out = run_sql(
        env,
        f"select id, left({text_expr}, {MAX_CHARS}) from {table} "
        f"where embedding is null and ({content_pred}) "
        f"order by id limit {FETCH_BATCH};",
        extra_args=ROW_ARGS,
    ).stdout
    pairs = []
    for record in split_records(out):
        row_id, _, text = record.partition(FIELD_SEP)
        if text.strip():
            pairs.append((int(row_id), text))
    return pairs


def embed_target(env: dict[str, str], name: str, model: str, dims: int) -> int:
    table, text_expr, content_pred = TARGETS[name]
    total = pending(env, table, content_pred)
    if total == 0:
        print(f"{name}: все записи уже несут семантический ключ")
        return 0
    print(f"{name}: к обсчёту {total}, модель {model}, страницами по {FETCH_BATCH}, "
          f"партиями к ollama по {EMBED_BATCH}")

    done, started = 0, time.monotonic()
    while True:
        pairs = fetch_page(env, table, text_expr, content_pred)
        if not pairs:
            # Страниц не осталось, либо у оставшихся записей пустой текст —
            # вектор им не из чего строить.
            break

        # Вся страница — один \copy плюс один UPDATE: embed() уже разбивает
        # тексты на партии по EMBED_BATCH внутри себя (pg_search.embed_batch).
        vecs = embed([t for _, t in pairs], model, dims)
        copy_csv_rows(
            env,
            "stage_embedding (id, embedding)",
            ([pid, vector_literal(vector)] for (pid, _), vector in zip(pairs, vecs)),
            preamble=_STAGE_DDL,
            epilogue=_update_sql(table),
        )

        done += len(pairs)
        rate = done / max(time.monotonic() - started, 1e-9)
        print(f"  {name}: {done}/{total}  ({rate:.1f} зап/с)", flush=True)
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", default=None,
                        help=f"что считать; по умолчанию всё: {list(TARGETS)}")
    parser.add_argument("--pgenv", type=Path, default=None,
                        help="файл с доступом к базе (по умолчанию corpus/.pgenv)")
    args = parser.parse_args(argv)

    which = args.targets or list(TARGETS)
    for name in which:
        if name not in TARGETS:
            print(f"неизвестная цель: {name}; известны {list(TARGETS)}")
            return 2
    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres недоступен: {exc}", file=sys.stderr)
        return 1
    # Один раз на прогон: таблица несёт ровно одну строку (CHECK (id = 1)),
    # и модель не может смениться посреди обсчёта.
    model, dims = resolve_target(env)
    for name in which:
        embed_target(env, name, model, dims)

    gaps = missing_semantic_key(env)
    if gaps:
        # Не падаем: пустой текст встречается легитимно. Но молчать нельзя —
        # запись без ключа находится только тем, кто знает точное слово.
        print("БЕЗ СЕМАНТИЧЕСКОГО КЛЮЧА:", ", ".join(f"{n}: {c}" for n, c in gaps))
    else:
        print("все записи-результаты несут семантический ключ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
