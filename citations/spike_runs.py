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
здесь только порядок записи. Ни печати, ни кодов возврата: режим возвращает
ЗАПИСАННОЕ (CalibrationRecord / HubRecord) либо поднимает NothingToMeasure,
а что из этого сказать человеку и с каким кодом выйти — дело CLI
(pg_load_citations.py), у которого и живут тела остальных двух режимов.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from pg_common import run_sql

from . import calibration, hub_cache, hub_report, threshold_store


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


class NothingToMeasure(RuntimeError):
    """The input a mode measures is empty, so there is no measurement.

    A domain error, not an exit code: "мерить нечего" is a fact about the
    input, and whether that fact ends the process (and on which stream it is
    said) is the CLI's decision, the same division store.py keeps between a
    writer and the loader that drives it. Refusing rather than recording is
    the point -- an empty input once put a run row into measurements whose
    verify_query "confirmed" numbers nobody had observed.
    """


class CalibrationRecord(NamedTuple):
    """What the calibration wrote, for whoever has to report it."""

    run_id: int
    tau_hint: float | None
    written: int
    report: Path


class HubRecord(NamedTuple):
    """What the hub measurement wrote. Under a dry-run writer only `counts`
    is a measurement: nothing was populated, so there are no statistics to
    read back and no report to name.
    """

    counts: list[int]
    run_id: int
    rows: list
    report: Path | None


def record_calibration(snowball, data_root: Path, writer) -> CalibrationRecord:
    """Распределение score всех кандидатов depth-1 -> measurements + отчёт."""
    rows = snowball.calibrate()
    if not rows:
        raise NothingToMeasure("кандидатов depth-1 нет — калибровать нечего")
    tau_hint = calibration.suggest_tau(rows)
    writer.ddl(threshold_store.THRESHOLD_DDL)
    run_id = writer.upsert_run(calibration.SPIKE, calibration.run_fields(rows))
    written = writer.threshold_rows(run_id, rows)
    report = data_root / calibration.REPORT_PATH
    writer.report(report, calibration.carry_over_sections(
        calibration.calibration_report(rows, tau_hint, snowball.candidate_refs), report))
    return CalibrationRecord(run_id, tau_hint, written, report)


def record_hub_report(env, cache, data_root: Path, writer, hub_cap: int) -> HubRecord:
    """Отрицательный результат про цену расширения вверх. Сети не требует.

    Отказ вместо записи, как в record_calibration: пустой вход — это не
    «замерили ноль», а «мерить было нечего». batch_counts() отдаёт пустой
    список на пустом или чужом кэше, а режим ходит в кэш по умолчанию
    (paths.default_cache_dir()) — так что прогон, писавший страницы в
    scratch, легко читается отсюда пустым.

    Кэш приходит объектом (citations/http_cache.py), а не путём: сайдкары,
    которые проход дописывает к страницам, — записи в дерево данных, и под
    --dry-run их не делает ReadOnlyCache, а не аккуратность этого модуля.
    """
    counts = hub_cache.batch_counts(cache)
    if not counts:
        raise NothingToMeasure(
            "в кэше нет ни одного батча cites: — мерить нечего; это кэш "
            "другого прогона либо страницы направления «вниз»")
    if writer.dry:
        return HubRecord(counts, 0, [], None)
    writer.ddl(hub_report.DDL)
    run_id = writer.upsert_run(hub_report.SPIKE,
                               hub_report.run_fields(counts, [], hub_cap))
    writer.populate(hub_report.POPULATE, run_id)
    rows = hub_report.stats(env, run_id, hub_cap)
    # verify_query называет ожидаемые числа, а они известны только после
    # заполнения таблицы. Правка НА МЕСТЕ: перезапись строки прогона унесла
    # бы каскадом только что записанные строки, и заполнять пришлось бы
    # второй раз (см. threshold_store.update_run_fields).
    writer.update_run_fields(
        hub_report.SPIKE,
        {"verify_query": hub_report.run_fields(counts, rows, hub_cap)["verify_query"]})
    report = data_root / hub_report.REPORT_PATH
    writer.report(report, hub_report.report(counts, rows, hub_report.worst_nodes(env, run_id),
                                            run_id, hub_cap))
    return HubRecord(counts, run_id, rows, report)
