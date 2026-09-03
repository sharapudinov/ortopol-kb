#!/usr/bin/env python3
"""Что два спайк-режима ГОВОРЯТ о записанном, и с каким кодом выходят.

Вторая половина шва, объявленного в spike_runs.py: тот модуль возвращает
ЗАПИСАННОЕ (CalibrationRecord / HubRecord) либо поднимает NothingToMeasure и
не печатает ни строки, а здесь живёт всё остальное — печать, stderr и коды
возврата.
Отдельный модуль от pg_load_citations.py по размеру (kb/CLAUDE.md
FILE_SIZE) и ровно по этому шву: тела двух других режимов — обхода и
склейки двойников — остались у командной строки, потому что говорят они о
графе, а не о measurements.

Ни писателя, ни кэша здесь не строят: оба приходят объектами (см.
DRY_RUN_WRITES_NOTHING) — у канала записи нет умолчания, а «какой режим
какой объект даёт» сказано один раз: писатели в
pg_load_citations.writers_for, все кэши — в pg_load_citations.main.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import calibration, hub_report
from .spike_runs import NothingToMeasure, record_calibration, record_hub_report


def do_hub_report(cache, tree_root: Path, writer, hub_cap: int) -> int:
    """Замер цены расширения вверх: что записано и что об этом сказано.

    Кэш приходит объектом, как писатель, и строит его pg_load_citations.main
    вместе с остальными — вместе с проверкой каталога, которую нельзя делать
    позже: рабочий кэш создаёт свой каталог при создании, и после этого
    «кэша ответов нет» сказать уже нечему. Само чтение идёт через объект
    (citations/http_cache.py), поэтому под --dry-run проход не дописывает
    к страницам ни одного сайдкара.

    Параметрами, а не `args`: у режима нет причин знать форму командной
    строки, и argparse.Namespace здесь означал бы, что вызвать замер без
    неё нельзя (сосед record_hub_report так и написан).
    """
    try:
        record = record_hub_report(cache, tree_root, writer, hub_cap)
    except NothingToMeasure as exc:
        print(f"{exc} (записей в кэше: {len(cache.names())})", file=sys.stderr)
        return 1
    if writer.dry:
        print("--dry-run: ничего не записано. meta.count батчей cites: "
              + ", ".join(str(c) for c in record.counts)
              + f" (сумма {sum(record.counts)}); записались бы прогон "
              f"{hub_report.SPIKE} и отчёт {record.report}. Статистику узлов "
              "читают из заполненной таблицы, поэтому её здесь нет.")
        return 0
    total = sum(int(row[1]) for row in record.rows)
    print(f"run {record.run_id} ({hub_report.SPIKE}); узлов depth-1: {total}; "
          f"отчёт: {record.report}")
    for row in record.rows:
        print("  " + " | ".join(row))
    return 0


def do_calibrate(snowball, client, tree_root: Path, writer) -> int:
    """Калибровка порога: тот же порядок — записать, потом рассказать."""
    try:
        record = record_calibration(snowball, tree_root, writer)
    except NothingToMeasure as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"запросов OpenAlex: {client.n_requests} (из кэша: {client.n_cache_hits})")
    print(calibration.boundary_line(record.tau_hint))
    if writer.dry:
        print(f"--dry-run: ни строки прогона, ни строк порога "
              f"({record.written} записалось бы), ни отчёта {record.report} — "
              "ничего не записано")
    else:
        print(f"run {record.run_id} ({calibration.SPIKE}); строк порога: "
              f"{record.written}; отчёт: {record.report}")
    return 0
