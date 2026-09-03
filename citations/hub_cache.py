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
(openalex_records.sidecar_name), поэтому и писатель у него один: клиент и
этот проход зовут cache.write(), а не write_text() мимо шва.

Сети не требует: батчи уже скачаны, и повторный проход по кэшу
воспроизводит числа при исчерпанной квоте.
"""
from __future__ import annotations

import json

from citation_vocab import Relation
from . import openalex_records

# Сколько байт головы страницы читается. meta идёт ПЕРВЫМ объектом тела и
# весит сотни байт, а всё, что нужно замеру, лежит в нём — значит целиком
# читать страницу (десятки мегабайт у батча «вниз») не нужно ни разу.
# Запас на порядки; если meta в голову не уложилась, читается тело.
HEAD_BYTES = 65536


def meta_in_head(head: str) -> dict | None:
    """Объект meta, вырезанный из головы страницы скобочным курсором.

    Всё, что нужно замеру, лежит в meta (openalex_records.page_index), а тело
    страницы батча «вниз» — десятки мегабайт, которые json.loads разворачивает
    в полный граф объектов ради двух чисел. Курсор считает скобки, пропуская
    строки и экранирование, поэтому вырезанный кусок — валидный JSON.

    None, если meta в голову не уложилась или оказалась не объектом: это не
    ошибка, а «здесь дешёвым путём не вышло», и читатель идёт длинным.
    """
    start = head.find('"meta"')
    if start < 0:
        return None
    start = head.find("{", start + len('"meta"'))
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for position in range(start, len(head)):
        symbol = head[position]
        if in_string:
            if escaped:
                escaped = False
            elif symbol == "\\":
                escaped = True
            elif symbol == '"':
                in_string = False
            continue
        if symbol == '"':
            in_string = True
        elif symbol == "{":
            depth += 1
        elif symbol == "}":
            depth -= 1
            if depth == 0:
                try:
                    meta = json.loads(head[start:position + 1])
                except ValueError:
                    return None
                # Голова страницы начинается с meta, но если бы кэш принёс
                # страницу другой формы, первым "meta" мог оказаться чужой
                # вложенный ключ. Тогда — длинный путь, а не тихий ноль.
                usable = isinstance(meta, dict) and ("x_query" in meta or "count" in meta)
                return meta if usable else None
    return None


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
        """{filter, direction, oql, count} страницы кэша: из памяти
        читателя, из сайдкара, иначе разбором головы.

        Сайдкар пишет сам клиент рядом со страницей
        (openalex_records.page_index), но кэш долговечен и не стирается: 259
        страниц лежат с тех пор, когда сайдкаров не было. Для них — один
        проход по ГОЛОВЕ (meta идёт первым объектом тела) и запись
        сайдкара, чтобы следующий прогон читал килобайты.
        """
        if name not in self._notes:
            self._notes[name] = self._read_note(name)
        return self._notes[name]

    def _read_note(self, name: str) -> dict | None:
        sidecar = openalex_records.sidecar_name(name)
        stored = self.cache.read(sidecar)
        if stored is not None:
            try:
                return json.loads(stored)
            except ValueError:
                # Битый сайдкар — не «страницы нет»: сама страница на месте
                # и читается тем же путём, что у страницы без сайдкара
                # (внизу он же и перепишется). Обрыв записи оставляет
                # непустой огрызок, то есть попадание, и ранний None
                # отчитался бы тихим нулём о полном кэше — ровно тем, против
                # чего этот читатель и написан.
                pass
        head = self.cache.read(name, limit=HEAD_BYTES)
        if head is None:
            return None
        note = self._note_from(meta_in_head(head))
        if note is None:
            # meta в голову не уложилась — страница устроена иначе, чем все
            # 259 в кэше, и тогда её читают целиком. Единственный путь, на
            # котором тело батча «вниз» становится объектом.
            note = self._note_from(self._whole_meta(name))
        if note is None:
            return None
        self.cache.write(sidecar, json.dumps(note, ensure_ascii=False))
        return note

    @staticmethod
    def _note_from(meta) -> dict | None:
        return None if meta is None else openalex_records.page_index({"meta": meta})

    def _whole_meta(self, name: str):
        body = self.cache.read(name)
        if body is None:
            return None
        try:
            return (json.loads(body) or {}).get("meta")
        except ValueError:
            return None

    def batch_counts(self) -> list[int]:
        """meta.count каждого батча «вверх» — по одному числу на батч.

        Направление берётся из поля страницы (openalex_records.note_direction,
        выведено из `filter=`), а не из английской фразы x_query.oql: oql —
        текст ЧУЖОЙ витрины, и его переформулировка обнулила бы весь замер
        молча, отчитавшись «нечего мерить» о полном кэше.
        """
        seen: dict[str, int] = {}
        for name in sorted(self.cache.names()):
            if not name.endswith(".json") or name.endswith(openalex_records.SIDECAR_SUFFIX):
                continue
            note = self.batch_note(name)
            if note is None or openalex_records.note_direction(note) != Relation.CITES:
                continue
            seen.setdefault(note.get("filter") or "", note.get("count") or 0)
        return sorted(seen.values(), reverse=True)


def batch_counts(cache) -> list[int]:
    """Один проход по кэшу. Читатель, которому нужна память между
    проходами, держит HubCacheReader сам — память живёт столько же, сколько
    объект, а не столько, сколько процесс.
    """
    return HubCacheReader(cache).batch_counts()
