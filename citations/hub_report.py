#!/usr/bin/env python3
"""Отрицательный результат: сколько цитирующих затягивает расширение вверх.

Замер, ради которого он существует (EXTENDING процедура D). Первая попытка
depth-2 от всех 382 узлов depth-1 не записала НИ ОДНОГО узла: батчи
`filter=cites:` вернули десятки тысяч работ, 253 страницы по 200, и окно
квоты в 1000 запросов кончилось внутри фазы «вверх». Причина не в объёме
вообще, а в том, ОТКУДА пришёл узел: направление «вниз» приводит классику
(справочники, учебники, обзоры), у классики цитирующих тысячи, и эти тысячи
— про область, а не про ИИШ.

Всё считается из уже имеющегося: citation.work (evidence несёт сырые записи
OpenAlex с cited_by_count и referenced_works_count) и кэш ответов в
corpus/cache/openalex (meta.count батчей). Сети не нужно — прогон
воспроизводится при исчерпанной квоте.

Вердикта здесь нет: его пишет основная сессия.
"""
from __future__ import annotations

import json
from pathlib import Path

from pg_common import run_sql, scalar

SPIKE = "research/citation-frontier-hub-expansion"
REPORT_PATH = "research/citation-frontier/hub-expansion.md"

DDL = """
CREATE TABLE IF NOT EXISTS measurements.citation_hub_expansion (
    run_id          BIGINT NOT NULL REFERENCES measurements.run(id) ON DELETE CASCADE,
    work_key        TEXT NOT NULL,
    -- Как узел попал в граф: 'cites' (цитирует фронтир) или 'referenced'
    -- (процитирован фронтиром). Это и есть варьируемая величина замера.
    relation        TEXT NOT NULL,
    cited_by_count  BIGINT NOT NULL,
    n_references    INTEGER NOT NULL,
    PRIMARY KEY (run_id, work_key)
);
CREATE INDEX IF NOT EXISTS citation_hub_expansion_relation
    ON measurements.citation_hub_expansion (run_id, relation);
"""

# Узлы depth-1 берутся по журналу, а не «все external-skeleton»: журнал —
# единственное место, где записана ГЛУБИНА, и замер обязан опираться на неё,
# а не на предположение «сейчас в базе только depth-1».
POPULATE = """
INSERT INTO measurements.citation_hub_expansion
    (run_id, work_key, relation, cited_by_count, n_references)
SELECT :run, w.key,
       coalesce(w.evidence->>'relation', 'unknown'),
       (SELECT coalesce(sum((r->>'cited_by_count')::bigint), 0)
          FROM jsonb_array_elements(w.evidence->'records') r),
       -- referenced_works_count, не length(referenced_works): сам список
       -- в evidence не хранится (registry.Node.absorb), а счётчик OpenAlex
       -- присылает рядом. Сверено на живой базе: 438 работ, 15634 ссылки
       -- обоими способами, ноль расхождений.
       (SELECT coalesce(sum((r->>'referenced_works_count')::bigint), 0)
          FROM jsonb_array_elements(w.evidence->'records') r)
FROM citation.work w
WHERE w.key IN (SELECT DISTINCT split_part(reason, 'node=', 2)
                FROM citation.crawl_step WHERE action = 'keep' AND depth = 1)
ON CONFLICT (run_id, work_key) DO UPDATE SET
    relation       = EXCLUDED.relation,
    cited_by_count = EXCLUDED.cited_by_count,
    n_references   = EXCLUDED.n_references;
"""

VERIFY_QUERY = (
    "SELECT relation, count(*) AS nodes, sum(cited_by_count) AS citers_total, "
    "max(cited_by_count) AS worst, "
    "count(*) FILTER (WHERE cited_by_count > 1000) AS over_cap, "
    "sum(n_references) AS refs_total "
    "FROM measurements.citation_hub_expansion "
    "WHERE run_id = (SELECT id FROM measurements.run WHERE spike = "
    f"'{SPIKE}') GROUP BY relation ORDER BY citers_total DESC;"
)


def batch_filter(x_query: dict) -> str:
    """Ключ батча — значение `filter=`, а НЕ полный x_query.url.

    Наблюдено: url несёт курсор в хвосте, поэтому у 8 батчей оказалось 253
    различных url — по одному на страницу, — и наивная дедупликация по url
    сложила один и тот же meta.count 95 раз (3 392 521 вместо 51 652).
    Значение filter= у всех страниц батча одно.
    """
    url = x_query.get("url") or ""
    if "filter=" not in url:
        return url or (x_query.get("oql") or "")
    return url.split("filter=", 1)[1].split("&", 1)[0]


def batch_counts(cache_dir: Path) -> list[int]:
    """meta.count каждого батча `cites:` из кэша — по одному числу на батч."""
    seen: dict[str, int] = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        meta = body.get("meta") or {}
        query = meta.get("x_query") or {}
        oql = query.get("oql") or ""
        if "cites" not in oql or "openalex id" in oql:
            continue
        seen.setdefault(batch_filter(query), meta.get("count") or 0)
    return sorted(seen.values(), reverse=True)


def stats(env, run_id: int) -> list[list[str]]:
    out = run_sql(
        env,
        "SELECT relation, count(*), sum(cited_by_count), max(cited_by_count), "
        "count(*) FILTER (WHERE cited_by_count > 1000), sum(n_references) "
        "FROM measurements.citation_hub_expansion WHERE run_id = :run "
        "GROUP BY relation ORDER BY sum(cited_by_count) DESC;",
        variables={"run": str(int(run_id))},
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    return [line.split("\x1f") for line in out.split("\n") if line]


def worst_nodes(env, run_id: int, limit: int = 10) -> list[list[str]]:
    out = run_sql(
        env,
        "SELECT h.relation, h.cited_by_count, coalesce(w.year::text, '-'), "
        "left(coalesce(w.title, ''), 70) "
        "FROM measurements.citation_hub_expansion h JOIN citation.work w ON w.key = h.work_key "
        f"WHERE h.run_id = :run ORDER BY h.cited_by_count DESC LIMIT {int(limit)};",
        variables={"run": str(int(run_id))},
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    return [line.split("\x1f") for line in out.split("\n") if line]


def run_fields(counts: list[int], rows: list[list[str]]) -> dict:
    total = sum(counts)
    return {
        "question": (
            "Сколько цитирующих затягивает расширение «вверх» от узлов depth-1 "
            "графа цитирований ИИШ и зависит ли это от типа входящей связи: "
            "узел пришёл как цитирующий фронтир ('cites') или как процитированный "
            "фронтиром ('referenced')?"
        ),
        "arbiter": (
            "Два независимых счёта. (1) meta.count батчей filter=cites: из кэша "
            "ответов OpenAlex (corpus/cache/openalex) — сколько работ источник "
            "обещал отдать на батч из 50 узлов depth-1: "
            + ", ".join(str(c) for c in counts) + f" (сумма {total}). "
            "(2) cited_by_count каждого из узлов depth-1 по типу связи — "
            "measurements.citation_hub_expansion, одна строка на узел, значения "
            "из сырых записей OpenAlex в citation.work.evidence. Счета независимы: "
            "первый — обещание источника, второй — сумма по нашим же записям."
        ),
        "reproduce": (
            "cd kb && set -a; . ../corpus/.pgenv; set +a && "
            "python3 pg_load_citations.py --hub-report   "
            "# СЕТИ НЕ ТРЕБУЕТ: считает из citation.work (evidence) и из кэша "
            "corpus/cache/openalex, поэтому воспроизводится при исчерпанной квоте. "
            "Узлы depth-1 определяются по журналу citation.crawl_step "
            "(action='keep', depth=1), а не предположением «в базе только depth-1»"
        ),
        "verify_query": VERIFY_QUERY + "  -- ждать: "
        + "; ".join(f"{r[0]} {r[1]} узлов, Σ cited_by {r[2]}, макс {r[3]}, "
                    f"за капом {r[4]}" for r in rows),
        "rules_out": (
            "Исключает перенос оценки survey §8 на depth-2: 4.603 цитирующих на "
            "узел и «31 запрос вверх» сняты на ЦИТИРУЮЩИХ пяти ключевых работ, а "
            "не на классике, которую приводит направление «вниз»; по факту батчи "
            f"cites: обещали {total} работ вместо ожидавшихся сотен, и окно квоты "
            "в 1000 запросов кончилось, не записав ни одного узла depth-2. "
            "Исключает и версию «дело в объёме графа, нужен порог повыше»: τ "
            "фильтрует РЕЛЕВАНТНОСТЬ кандидата, а стоимость создаёт цитируемость "
            "уже оставленного узла — величины разные, и порог по первой не "
            "управляет второй. Что НЕ исключено и не измерено: полезность самих "
            "цитирующих у hub-узлов — они не скачивались, утверждение здесь только "
            "о цене и об источнике цены."
        ),
        "source_url": "https://api.openalex.org/works",
        "family": ["all-families"],
        "area": ["citation-graph", "crawl-cost", "frontier"],
        "varied": ["relation"],
    }


def report(counts: list[int], rows: list[list[str]], worst: list[list[str]],
           run_id: int, cap: int) -> str:
    lines = [
        "# Расширение «вверх»: во что обходится и откуда цена",
        "",
        "Отрицательный результат замера 038.5. Первая попытка depth-2 от всех 382 "
        "узлов depth-1 **не записала ни одного узла**: батчи `filter=cites:` вернули "
        f"обещание в **{sum(counts)}** работ, скачано 253 страницы по 200, окно квоты "
        "OpenAlex (1000 запросов) кончилось внутри фазы «вверх».",
        "",
        f"Данные построчно: `measurements.citation_hub_expansion` (run {run_id}, "
        "строка на каждый узел depth-1). **Вердикт здесь не пишется.**",
        "",
        "## meta.count батчей `cites:` (обещание источника, из кэша)",
        "",
        "```",
        "  ".join(str(c) for c in counts),
        "```",
        "",
        "## Цитируемость узлов depth-1 по типу входящей связи",
        "",
        "| связь | узлов | Σ cited_by | максимум | за капом 1000 | Σ ссылок |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Десять самых цитируемых узлов depth-1", "",
              "| связь | cited_by | год | заголовок |", "|---|---|---|---|"]
    for row in worst:
        lines.append("| " + " | ".join(c.replace("|", "/") for c in row) + " |")
    lines += [
        "",
        "## Что из этого следует для формы обхода",
        "",
        f"Принято (оркестратор, 2026-09-02): на depth ≥ 2 расширяются только узлы со "
        f"связью `cites`; узлы `referenced` — листья. Плюс кап: узел с "
        f"`cited_by_count > {cap}` не спрашивается вверх (вниз — можно, ссылки уже в "
        "записи), с журнальной строкой `action='hub-skip'`.",
        "",
        "Почему именно так, а не «поднять τ»: порог отбирает кандидата по "
        "релевантности, а цену создаёт цитируемость уже оставленного узла. Это "
        "разные величины, и первая второй не управляет.",
    ]
    return "\n".join(lines) + "\n"
