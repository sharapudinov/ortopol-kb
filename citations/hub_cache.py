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


class HubCacheReader:
    """Проход по кэшу ответов, помнящий прочитанные страницы.

    Память — на объекте, а не в модуле. Сайдкар удешевляет СЛЕДУЮЩИЙ
    прогон, но под --dry-run кэш read-only и писать сайдкар некуда
    (DRY_RUN_WRITES_NOTHING) — тогда единственное, что спасает от
    повторного разбора десятков мегабайт, это память процесса. Модульный
    словарь давал её всем сразу: два объекта Cache над одним каталогом
    (DiskCache и ReadOnlyCache в одном процессе — ровно то, что делает
    прогон тестов) делили записи, а перезапись страницы ничего не
    сбрасывала. Читатель держит свою память ровно столько, сколько живёт
    сам, и подмена режима снова становится свойством того, что построено.

    Кэш приходит объектом (citations/http_cache.py), а не путём: проход
    дописывает к страницам сайдкары, то есть ПИШЕТ в дерево данных, и под
    --dry-run эту запись снимает ReadOnlyCache.
    """

    def __init__(self, cache):
        self.cache = cache
        self._notes: dict[str, dict | None] = {}

    def batch_note(self, name: str) -> dict | None:
        """{filter, oql, count} страницы кэша: из памяти читателя, из
        сайдкара, иначе разбором.

        Сайдкар пишет сам клиент рядом со страницей
        (openalex_client.page_index), но кэш долговечен и не стирается: 259
        страниц лежат с тех пор, когда сайдкаров не было. Для них — один
        проход: префильтр по СЫРОМУ тексту головы (страница батча «вниз»
        весит десятки мегабайт, и разбирать её ради двух полей нечего),
        затем разбор и запись сайдкара, чтобы следующий прогон читал
        килобайты.
        """
        if name not in self._notes:
            self._notes[name] = self._read_note(name)
        return self._notes[name]

    def _read_note(self, name: str) -> dict | None:
        sidecar = openalex_client.sidecar_name(name)
        stored = self.cache.read(sidecar)
        if stored is not None:
            try:
                return json.loads(stored)
            except ValueError:
                return None
        head = self.cache.read(name, limit=HEAD_BYTES)
        if head is None:
            return None
        if CITES_MARKER not in head and '"x_query"' in head:
            return None
        body = self.cache.read(name)
        if body is None:
            return None
        try:
            note = openalex_client.page_index(json.loads(body))
        except ValueError:
            return None
        self.cache.write(sidecar, json.dumps(note, ensure_ascii=False))
        return note

    def batch_counts(self) -> list[int]:
        """meta.count каждого батча `cites:` из кэша — по одному числу на батч."""
        seen: dict[str, int] = {}
        for name in sorted(self.cache.names()):
            if not name.endswith(".json") or name.endswith(openalex_client.SIDECAR_SUFFIX):
                continue
            note = self.batch_note(name)
            if note is None:
                continue
            oql = note.get("oql") or ""
            if "cites" not in oql or "openalex id" in oql:
                continue
            seen.setdefault(note.get("filter") or "", note.get("count") or 0)
        return sorted(seen.values(), reverse=True)


def batch_counts(cache) -> list[int]:
    """Один проход по кэшу. Читатель, которому нужна память между
    проходами, держит HubCacheReader сам — память живёт столько же, сколько
    объект, а не столько, сколько процесс.
    """
    return HubCacheReader(cache).batch_counts()
