#!/usr/bin/env python3
"""Что кэш ответов OpenAlex помнит о батчах «вверх».

Отделено от hub_report.py по ответственности (и по kb/CLAUDE.md FILE_SIZE):
тот модуль считает из БАЗЫ и пишет отчёт, этот читает КАТАЛОГ КЭША в дереве
данных. Общего у них ничего, кроме потребителя: замеру нужны оба счёта, и
именно их независимость делает вывод вердикта возможным.

Сети не требует: батчи уже скачаны, и повторный проход по кэшу
воспроизводит числа при исчерпанной квоте.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def batch_note(path: Path) -> dict | None:
    """{filter, oql, count} страницы кэша: из сайдкара, иначе разбором.

    Сайдкар пишет сам клиент рядом со страницей
    (openalex_client.page_index), но кэш долговечен и не стирается: 259
    страниц лежат с тех пор, когда сайдкаров не было. Для них — один проход:
    префильтр по СЫРОМУ тексту головы (страница батча «вниз» весит десятки
    мегабайт, и разбирать её ради двух полей нечего), затем разбор и запись
    сайдкара, чтобы следующий прогон читал килобайты.
    """
    sidecar = path.with_name(openalex_client.sidecar_name(path.name))
    try:
        if sidecar.is_file():
            return json.loads(sidecar.read_text(encoding="utf-8"))
        with path.open(encoding="utf-8") as handle:
            head = handle.read(HEAD_BYTES)
        if CITES_MARKER not in head and '"x_query"' in head:
            return None
        note = openalex_client.page_index(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None
    try:
        sidecar.write_text(json.dumps(note, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return note


def batch_counts(cache_dir: Path) -> list[int]:
    """meta.count каждого батча `cites:` из кэша — по одному числу на батч."""
    seen: dict[str, int] = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        if path.name.endswith(openalex_client.SIDECAR_SUFFIX):
            continue
        note = batch_note(path)
        if note is None:
            continue
        oql = note.get("oql") or ""
        if "cites" not in oql or "openalex id" in oql:
            continue
        seen.setdefault(note.get("filter") or "", note.get("count") or 0)
    return sorted(seen.values(), reverse=True)
