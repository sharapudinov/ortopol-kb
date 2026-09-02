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

Реестр целей и учёт остатка — рядом, в pg_embed_targets.py (kb/CLAUDE.md
FILE_SIZE): здесь — ОДИН проход по цели, там — какие цели вообще бывают.

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
    split_records,
    sql_literal,
    vector_literal,
)
from pg_copy import copy_csv_rows
from pg_embedding_text import MAX_CHARS
from pg_embed_targets import TARGETS, missing_semantic_key, pending
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


def fetch_page(env: dict[str, str], table: str, text_expr: str,
               content_pred: str, after: int = 0) -> list[tuple[int, str]]:
    """Страница ожидающих строк как (id, текст), обрезанный до MAX_CHARS,
    начиная сразу ЗА строкой `after`.

    Разбор общий (ROW_ARGS/split_records), а не построчный по '|': текст
    здесь — чужой (тело страницы, заголовок работы), а str.splitlines()
    считает границей строки не только \n (pg_common's own comment на
    FIELD_SEP/RECORD_SEP). Разделитель в тексте больше не рвёт строку на
    две записи.

    Пустой текст отсекает САМ запрос, а не разбор после него: такую строку
    цикл не обновляет никогда, и раньше она оставалась в голове выборки и
    перечитывалась на каждой итерации, а страница, целиком из таких строк
    состоящая, читалась как «работы больше нет». Предикат содержания
    (TARGETS) для этого не годится: у цели `runs` он «true» — у
    measurements.run нет колонки body, и пустой прогон отличим только по
    самому выражению текста.

    Курсор `after` — по первичному ключу, а не OFFSET: страница берётся
    `r.id > after`, поэтому выражение текста вычисляется ТОЛЬКО для строк
    после курсора. Без курсора каждая итерация пересчитывала ожидающих
    сначала, а отсечение пустого текста в WHERE сделало выражение
    квалификатором, считаемым до LIMIT: у цели `pages` это склейка целого
    тела страницы, индекса под `embedding is null` у corpus.pages нет, и
    цена прогона росла квадратично. Фильтр и порядок — по r.id, колонке
    таблицы, чтобы упорядоченный обход индекса остался доступен планировщику.

    Выражение текста подставляется ОДИН раз, в LATERAL, и фильтр читает уже
    вычисленное значение: две подстановки склеивали тело каждой строки
    дважды. Фильтр по ОБРЕЗАННОМУ тексту: вектор считается именно с него, и
    разбор ниже отбрасывал ровно этот случай.
    """
    out = run_sql(
        env,
        f"select t.id, t.txt from {table} r "
        f"cross join lateral (select r.id, left({text_expr}, {MAX_CHARS}) as txt) t "
        f"where r.embedding is null and ({content_pred}) and r.id > {int(after)} "
        f"and btrim(t.txt) <> '' "
        f"order by r.id limit {FETCH_BATCH};",
        extra_args=ROW_ARGS,
    ).stdout
    pairs = []
    for record in split_records(out):
        row_id, _, text = record.partition(FIELD_SEP)
        if text.strip():
            pairs.append((int(row_id), text))
    return pairs


def embed_target(env: dict[str, str], name: str, model: str, dims: int) -> int:
    """Обсчитывает цель и возвращает ОСТАТОК: сколько записей так и осталось
    без ключа.

    Остаток — арифметика, а не второй count(*): ожидало total, посчитано
    done, разница — строки с пустым после обрезки текстом, вектора у них не
    будет. Это же число печатает закрывающая строка прогона.
    """
    table, text_expr, content_pred = TARGETS[name]
    total = pending(env, table, content_pred)
    if total == 0:
        print(f"{name}: все записи уже несут семантический ключ")
        return 0
    print(f"{name}: к обсчёту {total}, модель {model}, страницами по {FETCH_BATCH}, "
          f"партиями к ollama по {EMBED_BATCH}")

    done, after, started = 0, 0, time.monotonic()
    while True:
        pairs = fetch_page(env, table, text_expr, content_pred, after)
        if not pairs:
            # Строк с непустым текстом и без вектора за курсором нет.
            # Пустые запрос уже не возвращает (fetch_page), а курсор идёт
            # только вперёд, поэтому пустая страница — конец работы.
            break
        after = pairs[-1][0]

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
    return total - done


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
    remaining = {name: embed_target(env, name, model, dims) for name in which}

    gaps = missing_semantic_key(env, which, remaining)
    if gaps:
        # Не падаем: пустой текст встречается легитимно. Но молчать нельзя —
        # запись без ключа находится только тем, кто знает точное слово.
        print("БЕЗ СЕМАНТИЧЕСКОГО КЛЮЧА:", ", ".join(f"{n}: {c}" for n, c in gaps))
    else:
        print("все записи-результаты несут семантический ключ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
