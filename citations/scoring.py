#!/usr/bin/env python3
"""Векторная арифметика фильтра и честный рендер распределения.

Отделено от frontier.py по ответственности (и по kb/CLAUDE.md FILE_SIZE):
там шов — эмбеддер ollama и чтение векторов из Postgres, здесь ЧИСТАЯ
математика, у которой нет ни одной зависимости, кроме `math`. Разделение
уже было названо потребителем: citations/calibration.py импортировал модуль
как `frontier as frontier_math`, то есть просил половину и говорил об этом
вслух.

Порог tau здесь НЕ выбирается и умолчания не имеет нигде в пакете. Его
меряют: pg_load_citations.py --calibrate считает score каждого кандидата
depth-1, пишет распределение в measurements.citation_frontier_threshold, а
вердикт о том, где проходит граница, выносит оркестратор. Квантили и
гистограмма внизу существуют, чтобы показать распределение честно, а не
чтобы выбрать за него число.
"""
from __future__ import annotations

import math

# Счёт кандидата, которого НЕ мерили: эмбеддить было нечего (нет заголовка —
# citations/candidates.py). Значение вне области косинуса снизу, поэтому
# «ниже любого tau» получается само, а «этот кандидат не сравнивался» видно
# по числу — и journal.drop() называет причину словами по этому же признаку.
# Одно объявление на обе стороны: тот, кто ставит счёт, и тот, кто о нём
# рассказывает, — иначе сравнение с -1.0 живёт в двух модулях.
NO_TEXT_SCORE = -1.0


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def centroid(vectors: list[list[float]]) -> list[float]:
    """Mean of the L2-normalized seed vectors, itself normalized.

    Normalizing before averaging is what makes the centroid a direction
    rather than a length-weighted average: without it a seed with a long
    abstract would pull the centre harder than a seed with a short one, for
    no reason anyone would defend.
    """
    if not vectors:
        raise ValueError("центроид пустого множества семян не определён")
    unit = [l2_normalize(v) for v in vectors]
    size = len(unit[0])
    if any(len(v) != size for v in unit):
        raise ValueError("векторы семян разной длины")
    mean = [sum(v[i] for v in unit) / len(unit) for i in range(size)]
    return l2_normalize(mean)


def cosine(a: list[float], b: list[float]) -> float:
    """Косинус между двумя произвольными векторами: нормируются оба."""
    if len(a) != len(b):
        raise ValueError(f"разная размерность: {len(a)} и {len(b)}")
    na, nb = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(na, nb))


def cosine_unit(a: list[float], unit_b: list[float]) -> float:
    """То же число, когда вторая сторона УЖЕ единичная.

    Ровно случай фильтра: центроид возвращается из centroid() нормированным
    и не меняется весь обход, а cosine() нормировал бы его заново на каждого
    кандидата — проход по 1024 числам за суммой квадратов, второй за
    делением и свежий список на выброс, тысячи раз за уровень.
    """
    if len(a) != len(unit_b):
        raise ValueError(f"разная размерность: {len(a)} и {len(unit_b)}")
    norm = math.sqrt(sum(v * v for v in a))
    if norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, unit_b)) / norm


def split_by_threshold(scored: dict[str, float], tau: float) -> tuple[list[str], list[str]]:
    """(kept, dropped) keys. `>= tau` keeps: the boundary belongs to the
    side the calibration recommended, and a candidate scoring exactly the
    recommended number is by construction one we said we wanted."""
    kept = sorted(k for k, s in scored.items() if s >= tau)
    dropped = sorted(k for k, s in scored.items() if s < tau)
    return kept, dropped


def quantiles(values: list[float], points=(0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[float, float]:
    """Nearest-rank quantiles -- every reported number is an observed score,
    not an interpolation between two of them."""
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
        out[point] = ordered[index]
    return out


def histogram(values: list[float], bins: int = 20) -> list[tuple[float, float, int]]:
    """[(low, high, count)] over [min, max]; the last bin is closed."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [(low, high, len(values))]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return [(low + i * width, low + (i + 1) * width, counts[i]) for i in range(bins)]
