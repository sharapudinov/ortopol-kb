#!/usr/bin/env python3
"""Какие виды записей обязаны нести семантический ключ — и сколько их ещё без него.

Отделено от pg_embed.py по ответственности (и по kb/CLAUDE.md FILE_SIZE):
тот модуль — ОДИН проход по цели (страница из базы, партия к ollama,
staged UPDATE), этот — реестр самих целей и учёт остатка. Реестр читают и
те, кому проход не нужен вовсе: «все ли записи-результаты находимы» — это
вопрос к базе, а не к обсчёту.

Выражение текста для цели `works` НЕ живёт здесь: у citation.work.embedding
два писателя, и правило текста одно на обоих (pg_embedding_text).
"""
from __future__ import annotations

from pg_common import scalar
from pg_embedding_text import WORKS_TEXT_SQL


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


def missing_semantic_key(env: dict[str, str], which=None,
                         known: dict[str, int] | None = None) -> list[tuple[str, int]]:
    """Записи-результаты без семантического ключа. Пустой список — инвариант держится.

    `which` — цели, о которых спрашивают; прогон отвечает за те, что тронул.
    Обход всего реестра означал count(*) по КАЖДОЙ чужой таблице: у цели
    `pages` предикат содержания — btrim(body) по corpus.pages, индекса под
    `embedding is null` там нет, и добор одной цели заканчивался полным
    проходом по самой большой таблице базы ради числа, которого он не менял.
    `known` — остатки, уже посчитанные обсчётом этого прогона (embed_target
    возвращает свой): цель, только что прошедшая обсчёт, знает свой остаток
    арифметикой, а отдельный count(*) по ней — второй полный агрегатный
    проход с предикатом содержания за уже известным числом.
    """
    known = known or {}
    out = []
    for name in (TARGETS if which is None else which):
        table, _, pred = TARGETS[name]
        n = known[name] if name in known else pending(env, table, pred)
        if n > 0:
            out.append((name, n))
    return out
