#!/usr/bin/env python3
"""Что два спайк-режима ГОВОРЯТ о записанном, и с каким кодом выходят.

Вторая половина шва, объявленного в spike_runs.py: тот модуль возвращает
ЗАПИСАННОЕ (CalibrationRecord / HubRecord) либо поднимает NothingToMeasure и
не печатает ни строки, а здесь живёт всё остальное — печать, stderr, коды
возврата и проверка входа, которую нельзя делать позже (каталог кэша).
Отдельный модуль от pg_load_citations.py по размеру (kb/CLAUDE.md
FILE_SIZE) и ровно по этому шву: тела двух других режимов — обхода и
склейки двойников — остались у командной строки, потому что говорят они о
графе, а не о measurements.

Ни писателя, ни кэша здесь не строят: оба приходят объектами (см.
DRY_RUN_WRITES_NOTHING) — у канала записи нет умолчания, а «какой режим
какой объект даёт» сказано один раз, в pg_load_citations.writers_for.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import calibration, hub_report
from .http_cache import cache_for
from .spike_runs import NothingToMeasure, record_calibration, record_hub_report


def do_hub_report(args, tree_root: Path, writer) -> int:
    """Замер цены расширения вверх: что записано и что об этом сказано.

    Каталог кэша проверяется ЗДЕСЬ и до того, как объект кэша построен:
    рабочий кэш создаёт свой каталог при создании, и после этого «кэша
    ответов нет» сказать было бы уже нечему — режим померил бы пустоту в
    только что созданном пустом каталоге. Само чтение идёт через объект
    (citations/http_cache.py), поэтому под --dry-run проход не дописывает
    к страницам ни одного сайдкара.
    """
    cache_path = Path(args.cache_dir)
    if not cache_path.is_dir():
        print(f"кэша ответов нет: {cache_path} — замер читает батчи cites: из него, "
              "и пустой каталог это не «ноль батчей», а «нечего мерить»; укажите "
              "--cache-dir того прогона, цену которого меряем", file=sys.stderr)
        return 1
    cache = cache_for(cache_path, read_only=args.dry_run)
    try:
        record = record_hub_report(cache, tree_root, writer, args.hub_cap)
    except NothingToMeasure as exc:
        print(f"{exc} (кэш {cache_path})", file=sys.stderr)
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
