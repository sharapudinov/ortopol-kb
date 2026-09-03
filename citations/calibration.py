#!/usr/bin/env python3
"""Что калибровка порога печатает и записывает: отчёт и строка прогона.

Отделено от pg_load_citations.py, потому что это другая ответственность:
загрузчик ходит в сеть и пишет граф, а здесь — только формулировки замера
(EXTENDING процедура D) и рендер распределения в текст.

Ни рекомендации, ни вердикта этот модуль не пишет: и то и другое — чтение
распределения человеком, привязанное к одному прогону. Обе секции живут в
самом отчёте и ПЕРЕНОСЯТСЯ в следующую регенерацию как есть
(carry_over_sections) — та же сделка, что у загрузчиков с transcribed-
страницами: перезапуск инструмента не уничтожает работу, которую этот
инструмент не делал. Из чисел здесь считается только одно, и оно посчитано,
а не записано: suggest_tau() — где в данных есть пустая корзина.
"""
from __future__ import annotations

from citation_vocab import Relation
from . import scoring
from .scoring import keeps

SPIKE = "research/citation-frontier-threshold"
REPORT_PATH = "research/citation-frontier/threshold.md"
# The sections the orchestrator and the executor write INTO the generated
# report by hand, in the order they appear. Regenerating carries everything
# from the first of them onward across unchanged; nothing above it survives,
# because that is exactly what the new measurement recomputed.
RECOMMENDATION_HEADING = "## Рекомендация исполнителя"
VERDICT_HEADING = "## Вердикт"
CARRIED_HEADINGS = (RECOMMENDATION_HEADING, VERDICT_HEADING)

# Ширина корзины, в которой ищется разрыв. 0.02 — примерно шаг, на котором
# 390 кандидатов depth-1 перестают быть непрерывными: уже — дыры от
# разреженности, шире — разрыв тонет в соседях.
BIN_WIDTH = 0.02


def suggest_tau(rows, width: float = BIN_WIDTH) -> float | None:
    """Середина ближайшей к медиане ПУСТОЙ корзины слева от неё, либо None.

    Читает ровно то, что в данных есть: место, где распределение
    прерывается. Корзины шириной `width` выкладываются от минимума; поиск
    идёт от медианы влево и останавливается на первой пустой — справа от
    медианы разрыв отделял бы работы от работ, а вопрос порога стоит о
    нижнем крае.

    None — честный ответ «границы в данных нет»: одномодальное распределение
    без разрыва не указывает точку, и любая «точка перегиба» была бы
    придумана. Вердикт в обоих случаях за оркестратором.
    """
    scores = sorted(r["score"] for r in rows)
    if len(scores) < 2 or width <= 0:
        return None
    low, median = scores[0], scores[len(scores) // 2]
    counts: dict[int, int] = {}
    for score in scores:
        counts[int((score - low) / width)] = counts.get(int((score - low) / width), 0) + 1
    for index in range(int((median - low) / width) - 1, -1, -1):
        if not counts.get(index):
            return low + (index + 0.5) * width
    return None


def cost_table(rows, refs, taus=(0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70)) -> list[str]:
    """Во что обойдётся depth-2 при каждом τ: узлы фронтира и запросы.

    Ссылки считаются ПО ОБЪЕДИНЕНИЮ, а не суммой по узлам: суммой выходит
    15177 против 4262 различных работ — завышение в 3.5 раза, потому что
    соседи по цитированию ссылаются на одно и то же. Цену обхода определяет
    число РАЗНЫХ работ, которые придётся дозапросить: по 50 идентификаторов
    на запрос вниз, два курсорных запроса на батч из 50 вверх.
    """
    lines = ["| τ | оставлено на depth-1 | различных ссылок | ≈ запросов depth-2 |",
             "|---|---|---|---|"]
    for tau in taus:
        kept = [r for r in rows if keeps(r["score"], tau)]
        union: set[str] = set()
        for row in kept:
            union |= refs.get(row["candidate_key"], set())
        down = -(-len(union) // 50)
        up = 2 * (-(-len(kept) // 50))
        lines.append(f"| {tau:.2f} | {len(kept)} | {len(union)} | {down + up} |")
    return lines


def title_table(rows) -> list[str]:
    lines = ["| score | год | связь | реферат | заголовок |", "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: -r["score"]):
        title = (row["title"] or "—").replace("|", "/")[:100]
        has = "да" if row.get("has_abstract") else "нет"
        lines.append(f"| {row['score']:.4f} | {row['year'] or '—'} | {row['relation']} "
                     f"| {has} | {title} |")
    return lines


def calibration_report(rows, tau_hint: float | None, refs=None) -> str:
    scores = [r["score"] for r in rows]
    with_abstract = sorted(r["score"] for r in rows if r.get("has_abstract"))
    without = sorted(r["score"] for r in rows if not r.get("has_abstract"))
    lines = [
        "# Порог релевантности снежного кома: распределение depth-1",
        "",
        f"Кандидатов depth-1: {len(rows)} "
        f"(цитирующие {sum(1 for r in rows if r['relation'] == Relation.CITES)}, "
        "процитированные "
        f"{sum(1 for r in rows if r['relation'] == Relation.REFERENCED)}).",
        "Score = косинус между эмбеддингом `title + abstract` кандидата и центроидом",
        "нормированных эмбеддингов 56 семян; модель — та же, что у `corpus.pages`",
        "(`corpus.embedding_model`). Данные построчно:",
        f"`measurements.citation_frontier_threshold` (run по спайку `{SPIKE}`).",
        "**Вердикт здесь не пишется** — его выносит основная сессия.",
        "",
        "## Квантили",
        "",
        "| квантиль | score |",
        "|---|---|",
    ]
    for point, value in sorted(scoring.quantiles(scores).items()):
        lines.append(f"| {point:.2f} | {value:.4f} |")
    lines += ["", f"min {min(scores):.4f}, max {max(scores):.4f}, "
                  f"среднее {sum(scores) / len(scores):.4f}", "", "## Гистограмма", "", "```"]
    for low, high, count in scoring.histogram(scores, bins=20):
        lines.append(f"{low:6.3f}..{high:6.3f} {'#' * min(count, 60):<60} {count}")
    lines += ["```", "", "## Реферат сдвигает score", ""]
    if with_abstract and without:
        lines += [
            "| подмножество | n | p25 | медиана | p75 | среднее |",
            "|---|---|---|---|---|---|",
        ]
        for name, values in (("с рефератом", with_abstract), ("без реферата", without)):
            lines.append(
                f"| {name} | {len(values)} | {values[len(values) // 4]:.4f} | "
                f"{values[len(values) // 2]:.4f} | {values[3 * len(values) // 4]:.4f} | "
                f"{sum(values) / len(values):.4f} |")
        lines += ["", "Реферата в OpenAlex нет у старых русских работ и у монографий "
                      "(run 85: 30 из 56), поэтому высокий порог отсекает их за неполноту "
                      "метаданных, а не за содержание."]
    lines += ["", "## Цена depth-2 при разном τ", ""] + cost_table(rows, refs or {})
    lines += ["", "## Десять нижних кандидатов (что именно отсекает низкий порог)", ""]
    lines += title_table(sorted(rows, key=lambda r: r["score"])[:10])
    lines += ["", "## Разрыв в распределении", "", boundary_line(tau_hint), ""]
    if tau_hint is not None:
        lines += [f"Десять заголовков вокруг неё (τ = {tau_hint:.2f}):", ""]
        lines += title_table(sorted(rows, key=lambda r: abs(r["score"] - tau_hint))[:10])
    lines += ["", "Рекомендация исполнителя и вердикт оркестратора ниже написаны "
                  "руками и переносятся сюда из предыдущей версии отчёта как есть: "
                  "генератор их не пишет и не правит.", ""]
    return "\n".join(lines) + "\n"


def boundary_line(tau_hint: float | None) -> str:
    """Что распределение говорит о границе — включая «ничего»."""
    if tau_hint is None:
        return (f"Пустых корзин шириной {BIN_WIDTH} слева от медианы нет: "
                "границы в данных нет, распределение её не показывает.")
    low, high = tau_hint - BIN_WIDTH / 2, tau_hint + BIN_WIDTH / 2
    return (f"Ближайшая к медиане пустая корзина слева: {low:.4f}…{high:.4f} — "
            f"ни одного кандидата. Её середина: τ = {tau_hint:.4f}. Это то, что "
            "данные показывают сами; вердикт — за оркестратором.")


def carry_over_sections(new_text: str, previous_path) -> str:
    """Re-attach the hand-written tail of the report to a freshly generated
    one: the executor's recommendation and the orchestrator's verdict.

    Without this, a second --calibrate silently deletes sections this code
    never wrote. Nothing else of the old file survives: the facts above them
    are exactly what the new measurement recomputed.
    """
    if not previous_path.is_file():
        return new_text
    previous = previous_path.read_text(encoding="utf-8")
    found = [previous.index(h) for h in CARRIED_HEADINGS if h in previous
             and h not in new_text]
    if not found:
        return new_text
    return new_text.rstrip("\n") + "\n\n" + previous[min(found):]


def run_fields(rows) -> dict:
    scores = sorted(r["score"] for r in rows)
    return {
        "question": (
            "Где на распределении косинуса к центроиду семян проходит граница "
            "релевантного фронтира графа цитирований ИИШ: какое τ отделяет работы, "
            "которым место в скелете графа, от случайных соседей по цитированию?"
        ),
        "arbiter": (
            "Распределение score всех кандидатов depth-1 (одна строка на кандидата "
            "в measurements.citation_frontier_threshold): гистограмма и квантили плюс "
            "ручная выборка 10 узлов вокруг кандидата на порог с заголовками — "
            "порог принимается по тому, что читается в заголовках у границы, а не по "
            "виду кривой. Эмбеддинг — та же модель, что у corpus.pages "
            "(corpus.embedding_model), иначе дистанция считается, а результат мусор."
        ),
        "reproduce": (
            "cd kb && set -a; . ../corpus/.pgenv; set +a && "
            "python3 pg_load_citations.py --calibrate   "
            "# семена = 56 матчей OpenAlex из run 85, кандидаты = citers+references "
            "семян. Таблица создаётся самим прогоном (citations/threshold_store.py, "
            "THRESHOLD_DDL) и несёт колонки run_id, candidate_key, depth, relation, "
            "score, title, year, has_abstract, n_references — последние две "
            "добавлены после первой калибровки идемпотентным ADD COLUMN IF NOT "
            "EXISTS, поэтому пересоздание с нуля даёт ту же форму: has_abstract "
            "держит вывод «без реферата медиана ниже» (0.6506 против 0.6893), "
            "n_references — цену раскрытия узла на следующей глубине. "
            "ЧИСЛА НЕ ВЕЧНЫ: OpenAlex живой индекс, cited_by_count растёт"
        ),
        "verify_query": (
            "SELECT count(*), round(min(score)::numeric,4) AS min, "
            "round(percentile_cont(0.5) WITHIN GROUP (ORDER BY score)::numeric,4) AS p50, "
            "round(percentile_cont(0.9) WITHIN GROUP (ORDER BY score)::numeric,4) AS p90, "
            "round(max(score)::numeric,4) AS max "
            "FROM measurements.citation_frontier_threshold "
            "WHERE run_id = (SELECT id FROM measurements.run WHERE spike = "
            f"'{SPIKE}');  -- ждать: {len(scores)} строк, "
            f"min {scores[0]:.4f}, max {scores[-1]:.4f}"
        ),
        "rules_out": (
            "Исключает выбор τ «на глаз»: порог теперь цитата запроса к "
            "measurements.citation_frontier_threshold, а не число в коде. "
            "Исключает и допущение «релевантность видна по одному разрыву в "
            "распределении» ровно в той мере, в какой гистограмма его не "
            "показывает — что именно она показывает, записано в "
            "research/citation-frontier/threshold.md."
        ),
        "source_url": "https://api.openalex.org/works",
        "family": ["all-families"],
        "area": ["citation-graph", "relevance", "frontier"],
        "varied": ["threshold"],
    }
