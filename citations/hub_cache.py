#!/usr/bin/env python3
"""Что кэш ответов OpenAlex помнит о батчах «вверх».

Отделено от hub_report.py по ответственности (и по kb/CLAUDE.md FILE_SIZE):
тот модуль считает из БАЗЫ и пишет отчёт, этот читает КЭШ ОТВЕТОВ в дереве
данных. Общего у них ничего, кроме потребителя: замеру нужны оба счёта, и
именно их независимость делает вывод вердикта возможным.

Кэш приходит объектом (citations/http_cache.py), а не путём: проход
дописывает к страницам сайдкары, то есть ПИШЕТ в дерево данных, и под
--dry-run эту запись снимает ReadOnlyCache — тем же способом, что
DryRunWriter снимает запись в citation.*. Сайдкар у страницы один и тот же
(openalex_client.sidecar_name), поэтому и писатель у него один: клиент и
этот проход зовут cache.write(), а не write_text() мимо шва.

Сети не требует: батчи уже скачаны, и повторный проход по кэшу
воспроизводит числа при исчерпанной квоте.
"""
from __future__ import annotations

import json

from . import openalex_client

# Как OpenAlex формулирует запрос «кто цитирует эти 50» в meta.x_query.oql.
# Проверено на всём кэше: маркер стоит ровно у 253 страниц батчей cites: и
# ни у одной из 6 страниц направления «вниз» (openalex id).
CITES_MARKER = "works where it cites"
# Сколько байт головы страницы читается на префильтр. meta идёт первым
# объектом тела, x_query — вторым полем meta, так что маркер лежит в первых
# сотнях байт; запас на порядки. Если в голове нет даже x_query, страница
# устроена иначе, и решает уже разбор целиком, а не догадка.
HEAD_BYTES = 65536


def batch_note(cache, name: str) -> dict | None:
    """{filter, oql, count} страницы кэша: из сайдкара, иначе разбором.

    Сайдкар пишет сам клиент рядом со страницей
    (openalex_client.page_index), но кэш долговечен и не стирается: 259
    страниц лежат с тех пор, когда сайдкаров не было. Для них — один проход:
    префильтр по СЫРОМУ тексту головы (страница батча «вниз» весит десятки
    мегабайт, и разбирать её ради двух полей нечего), затем разбор и запись
    сайдкара, чтобы следующий прогон читал килобайты.
    """
    sidecar = openalex_client.sidecar_name(name)
    stored = cache.read(sidecar)
    if stored is not None:
        try:
            return json.loads(stored)
        except ValueError:
            return None
    head = cache.read(name, limit=HEAD_BYTES)
    if head is None:
        return None
    if CITES_MARKER not in head and '"x_query"' in head:
        return None
    body = cache.read(name)
    if body is None:
        return None
    try:
        note = openalex_client.page_index(json.loads(body))
    except ValueError:
        return None
    cache.write(sidecar, json.dumps(note, ensure_ascii=False))
    return note


def batch_counts(cache) -> list[int]:
    """meta.count каждого батча `cites:` из кэша — по одному числу на батч."""
    seen: dict[str, int] = {}
    for name in sorted(cache.names()):
        if not name.endswith(".json") or name.endswith(openalex_client.SIDECAR_SUFFIX):
            continue
        note = batch_note(cache, name)
        if note is None:
            continue
        oql = note.get("oql") or ""
        if "cites" not in oql or "openalex id" in oql:
            continue
        seen.setdefault(note.get("filter") or "", note.get("count") or 0)
    return sorted(seen.values(), reverse=True)
