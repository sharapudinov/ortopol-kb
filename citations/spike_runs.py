#!/usr/bin/env python3
"""Два спайк-режима загрузчика и ЕДИНСТВЕННЫЙ шов, через который они пишут.

citations/store.py делает это для графа: Writer / PostgresWriter /
DryRunWriter — один шов, и обещание «--dry-run: в базу ничего не записано»
держится конструкцией, а не аккуратностью. Здесь то же самое для схемы
measurements: калибровка порога и замер цены расширения вверх ходят в базу
и на диск только через объект-писатель, поэтому под --dry-run не остаётся
ни строки прогона, ни строк данных, ни перезаписанного отчёта.

Почему отдельный шов, а не тот же: у графа и у замера разные контракты.
Store пишет долговременный граф (upsert, сохранённые kind и эмбеддинги),
здесь — результат исследования по процедуре D EXTENDING: идемпотентность
по имени спайка, вердикт остаётся оркестратору, отчёт — файл в дереве
данных.

Формулировки замеров и рендер отчётов — в calibration.py и hub_report.py;
здесь только порядок записи.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pg_common import run_sql

from . import calibration, hub_report, threshold_store


class MeasurementsWriter:
    """Живая база и диск: то, что режим записывает на самом деле."""

    dry = False

    def __init__(self, env):
        self.env = env

    def ddl(self, sql: str) -> None:
        run_sql(self.env, sql)

    def upsert_run(self, spike: str, fields: dict) -> int:
        return threshold_store.upsert_run(self.env, spike, fields)

    def update_run_fields(self, spike: str, fields: dict) -> None:
        threshold_store.update_run_fields(self.env, spike, fields)

    def threshold_rows(self, run_id: int, rows) -> int:
        return threshold_store.insert_threshold_rows(self.env, run_id, rows)

    def populate(self, sql: str, run_id: int) -> None:
        run_sql(self.env, sql, variables={"run": str(int(run_id))})

    def report(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class DryRunMeasurementsWriter:
    """Тот же контракт, не трогающий ни базу, ни диск.

    Возвращает 0 вместо id прогона: под --dry-run строки прогона нет, и
    номер, которого никто не создавал, печатать нельзя. Вызовы копятся в
    .calls, чтобы режим мог сказать, что именно он записал бы.
    """

    dry = True

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def ddl(self, sql: str) -> None:
        self.calls.append(("ddl", sql))

    def upsert_run(self, spike: str, fields: dict) -> int:
        self.calls.append(("upsert_run", spike))
        return 0

    def update_run_fields(self, spike: str, fields: dict) -> None:
        self.calls.append(("update_run_fields", spike))

    def threshold_rows(self, run_id: int, rows) -> int:
        rows = list(rows)
        self.calls.append(("threshold_rows", len(rows)))
        return len(rows)

    def populate(self, sql: str, run_id: int) -> None:
        self.calls.append(("populate", run_id))

    def report(self, path: Path, text: str) -> None:
        self.calls.append(("report", str(path)))


def record_calibration(snowball, client, data_root: Path, writer) -> int:
    """Распределение score всех кандидатов depth-1 -> measurements + отчёт."""
    rows = snowball.calibrate()
    if not rows:
        print("кандидатов depth-1 нет — калибровать нечего", file=sys.stderr)
        return 1
    tau_hint = calibration.suggest_tau(rows)
    writer.ddl(threshold_store.THRESHOLD_DDL)
    run_id = writer.upsert_run(calibration.SPIKE, calibration.run_fields(rows, tau_hint))
    written = writer.threshold_rows(run_id, rows)
    report = data_root / calibration.REPORT_PATH
    writer.report(report, calibration.carry_over_verdict(
        calibration.calibration_report(rows, tau_hint, snowball.candidate_refs), report))
    print(f"запросов OpenAlex: {client.n_requests} (из кэша: {client.n_cache_hits})")
    print(f"рекомендация τ (не вердикт): {tau_hint:.4f}")
    if writer.dry:
        print(f"--dry-run: ни строки прогона, ни строк порога ({written} записалось бы), "
              f"ни отчёта {report} — ничего не записано")
    else:
        print(f"run {run_id} ({calibration.SPIKE}); строк порога: {written}; отчёт: {report}")
    return 0


def record_hub_report(env, cache_dir, data_root: Path, writer, hub_cap: int) -> int:
    """Отрицательный результат про цену расширения вверх. Сети не требует."""
    counts = hub_report.batch_counts(Path(cache_dir))
    if writer.dry:
        print("--dry-run: ничего не записано. meta.count батчей cites: "
              + ", ".join(str(c) for c in counts) + f" (сумма {sum(counts)}). "
              "Таблица узлов не заполнялась, поэтому ни её статистик, ни отчёта "
              "в этом режиме нет: они читаются из записанного.")
        return 0
    writer.ddl(hub_report.DDL)
    run_id = writer.upsert_run(hub_report.SPIKE, hub_report.run_fields(counts, []))
    writer.populate(hub_report.POPULATE, run_id)
    rows = hub_report.stats(env, run_id)
    # verify_query называет ожидаемые числа, а они известны только после
    # заполнения таблицы. Правка НА МЕСТЕ: перезапись строки прогона унесла
    # бы каскадом только что записанные строки, и заполнять пришлось бы
    # второй раз (см. threshold_store.update_run_fields).
    writer.update_run_fields(
        hub_report.SPIKE,
        {"verify_query": hub_report.run_fields(counts, rows)["verify_query"]})
    report = data_root / hub_report.REPORT_PATH
    writer.report(report, hub_report.report(counts, rows, hub_report.worst_nodes(env, run_id),
                                            run_id, hub_cap))
    total = sum(int(r[1]) for r in rows)
    print(f"run {run_id} ({hub_report.SPIKE}); узлов depth-1: {total}; отчёт: {report}")
    for row in rows:
        print("  " + " | ".join(row))
    return 0
