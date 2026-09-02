#!/usr/bin/env python3
"""Что калибровка порога печатает и записывает: отчёт и строка прогона.

Отделено от pg_load_citations.py, потому что это другая ответственность:
загрузчик ходит в сеть и пишет граф, а здесь — только формулировки замера
(EXTENDING процедура D) и рендер распределения в текст.

RECOMMENDATION — прочтение распределения ИСПОЛНИТЕЛЕМ, константа, чтобы
отчёт пересоздавался байт в байт. Вердикт (само число τ, принятое к обходу)
выносит оркестратор; здесь только рекомендация и факты под ней.
"""
from __future__ import annotations

from . import frontier as frontier_math

SPIKE = "research/citation-frontier-threshold"
REPORT_PATH = "research/citation-frontier/threshold.md"
# The orchestrator writes the verdict INTO the generated report, by hand.
# Regenerating the report must therefore carry that section over instead of
# overwriting it -- the same bargain the loaders strike with transcribed
# pages: a rerun of a tool never destroys work the tool did not produce.
VERDICT_HEADING = "## Вердикт"

RECOMMENDED_TAU = 0.50
# The executor's reading of the distribution below, kept as a constant so the
# generated report is reproducible byte for byte. The VERDICT is the
# orchestrator's; this is the recommendation the facts in the report support.
RECOMMENDATION = """\
**τ = 0.50.**

1. **Это единственная пустая корзина гистограммы.** Распределение
   одномодальное и без плато, кроме одного места: 0.484…0.504 — ноль
   кандидатов. Слева от неё шесть записей и все шесть работами не являются:
   `Preface` (0.4245), `Bibliography` (0.4381), `Rellich Inequality` (0.4445),
   `Navigation System For Elderly Care Applications` (0.4515), `Maximal
   Operator` (0.4647), `Index` (0.4736) — OpenAlex индексирует главы и
   служебные разделы сборников отдельными works. Справа, начиная с 0.5182,
   идут работы. Порог в пустой корзине отделяет «не работа» от «работа», и
   это единственная граница на всей кривой, которую данные ставят сами.
2. **Всё, что выше, — по делу, и чем ниже смотришь, тем дороже ошибка.**
   Первые за границей: «Moment Functions in Image Analysis» (0.5182 — прямая
   цель продукта «моменты изображений»), «Collected Works» Чебышёва (0.5259),
   «Global asymptotics of the Hahn polynomials» (0.5274), «Mean Convergence of
   Expansions in Laguerre and Hermite Series» (0.5338), «Transverse limits in
   the Askey tableau» (0.5470), «Nonnegativity of a discrete Poisson kernel
   for the Hahn polynomials» (0.5693), «Limit relationships between Chebyshev
   and Hahn polynomials» (0.5891), «Asymptotic analysis of the Krawtchouk
   polynomials by the WKB method» (0.5916). Порог 0.60 выбрасывает последние
   две, порог 0.65 — весь этот список: ровно ту литературу, ради которой
   существуют задачи 010 (граница Кравчука) и M1 (асимптотики).
3. **Высокий порог фильтрует полноту метаданных, а не релевантность.**
   Измерено на этих же 390 кандидатах: у 173 без реферата медиана 0.6506, у
   217 с рефератом — 0.6893. Реферата в OpenAlex нет как раз у старых русских
   работ и у монографий — у слоя, который и в run 85 был опознан хуже
   прочих. Порог 0.65 срезал бы их не за содержание, а за то, что источник о
   них знает меньше.
4. **Цена не аргумент за высокий порог.** depth-2 при τ=0.50 — ≈102 запроса
   против ≈59 при τ=0.65, при окне квоты 1000 (на 2026-09-02 остаток ≈280).

**Вывод важнее самого числа: на depth-1 фильтр почти ничего не делает** —
384 кандидата из 390. Окрестность цитирования ИИШ по построению чистая:
работа, которая цитирует ИИШ или процитирована им, почти никогда не бывает
не по теме, и косинус к центроиду семян здесь отделяет не «своё от чужого», а
«работу от оглавления». Ценность порога проверится на depth-2, где популяция
кандидатов другая (соседи соседей); тот же τ там может вести себя иначе, и
это стоит перемерить отдельно, а не считать установленным.
"""


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
        kept = [r for r in rows if r["score"] >= tau]
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


def calibration_report(rows, tau_hint: float, refs=None) -> str:
    scores = [r["score"] for r in rows]
    with_abstract = sorted(r["score"] for r in rows if r.get("has_abstract"))
    without = sorted(r["score"] for r in rows if not r.get("has_abstract"))
    lines = [
        "# Порог релевантности снежного кома: распределение depth-1",
        "",
        f"Кандидатов depth-1: {len(rows)} "
        f"(цитирующие {sum(1 for r in rows if r['relation'] == 'cites')}, "
        f"процитированные {sum(1 for r in rows if r['relation'] == 'referenced')}).",
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
    for point, value in sorted(frontier_math.quantiles(scores).items()):
        lines.append(f"| {point:.2f} | {value:.4f} |")
    lines += ["", f"min {min(scores):.4f}, max {max(scores):.4f}, "
                  f"среднее {sum(scores) / len(scores):.4f}", "", "## Гистограмма", "", "```"]
    for low, high, count in frontier_math.histogram(scores, bins=20):
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
    lines += ["", f"## Десять заголовков вокруг рекомендуемой границы τ = {tau_hint:.2f}", ""]
    lines += title_table(sorted(rows, key=lambda r: abs(r["score"] - tau_hint))[:10])
    lines += ["", "## Рекомендация исполнителя (вердикт — за оркестратором)", "",
              RECOMMENDATION]
    return "\n".join(lines) + "\n"


def carry_over_verdict(new_text: str, previous_path) -> str:
    """Re-attach the orchestrator's verdict to a freshly generated report.

    Without this, a second --calibrate silently deletes a section this code
    never wrote. Nothing else of the old file survives: the facts above the
    verdict are exactly what the new measurement recomputed.
    """
    if not previous_path.is_file():
        return new_text
    previous = previous_path.read_text(encoding="utf-8")
    if VERDICT_HEADING not in previous or VERDICT_HEADING in new_text:
        return new_text
    return new_text.rstrip("\n") + "\n\n" + previous[previous.index(VERDICT_HEADING):]


def run_fields(rows, tau_hint: float) -> dict:
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


def suggest_tau(_rows) -> float:
    """Число, которое исполнитель предлагает, прочитав распределение.

    Формулы здесь нет намеренно: распределение одномодальное и без разрыва,
    так что любая «точка перегиба» была бы придумана. Обоснование — в
    RECOMMENDATION, вердикт — за оркестратором.
    """
    return RECOMMENDED_TAU


