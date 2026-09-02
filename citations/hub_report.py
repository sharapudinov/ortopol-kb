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
corpus/cache/openalex (meta.count батчей — их считает hub_cache.py). Сети не
нужно — прогон воспроизводится при исчерпанной квоте.

Вердикта здесь нет: его пишет основная сессия.
"""
from __future__ import annotations

from citation_vocab import CrawlAction, Relation
from pg_common import run_sql
from pg_common import FIELD_SEP, ROW_ARGS, split_records

SPIKE = "research/citation-frontier-hub-expansion"
REPORT_PATH = "research/citation-frontier/hub-expansion.md"

DDL = """
CREATE TABLE IF NOT EXISTS measurements.citation_hub_expansion (
    run_id          BIGINT NOT NULL REFERENCES measurements.run(id) ON DELETE CASCADE,
    work_key        TEXT NOT NULL,
    -- Как узел попал в граф: цитирует фронтир или процитирован им
    -- (citation_vocab.Relation). Это и есть варьируемая величина замера.
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
# а не на предположение «сейчас в базе только depth-1». Узел — это node_key,
# отдельная колонка: кандидат и узел не одно и то же (два кандидата
# сливаются в один узел), и разбор прозы reason этого не знал.
#
# relation берётся ОТТУДА ЖЕ — из колонки решения, а не из
# citation.work.evidence с запасным 'unknown'. Evidence лепит
# registry.Node.absorb из сырых записей источника; классифицировал же узел
# обход, и его ответ лежит в журнале. DISTINCT ON (node_key) по возрастанию
# id берёт первую строку keep для узла: реестр присваивает relation при
# СОЗДАНИИ узла (registry.add), последующие записи его не меняют.
#
# Отдельного индекса под этот срез НЕТ, и это измерено, а не забыто.
# EXPLAIN (ANALYZE, BUFFERS) на живой базе, журнал наращён до depth-2-
# размера внутри откатываемой транзакции (100 тыс. и 400 тыс. строк),
# сравнение с CREATE INDEX ... (depth, node_key) WHERE action = 'keep':
#
#   ~100k строк: чтение журнала 1558 буферов -> 287 с индексом,
#                весь POPULATE 9.9 мс -> 8.3 мс;
#   ~400k строк: 548 -> 287, 9.3 мс -> 9.0 мс.
#
# Выигрыш не растёт с журналом, а СЖИМАЕТСЯ: последовательного скана здесь
# нет и без нового индекса — планировщик берёт node_key IS NOT NULL
# bitmap-скном по crawl_step_node_key_idx, а на 400 тыс. переключает план
# сам. Основная цена статична и лежит не в журнале: 382 поиска по
# citation.work (1146 буферов) и разворот evidence (730). Индекс — это
# запись на КАЖДУЮ строку keep всякого обхода ради 0.3 мс у замера,
# который запускают вручную; не добавлен.
#
# evidence — самая объёмная колонка схемы, и две скалярных подзапроса по
# одному и тому же массиву разворачивали его ДВАЖДЫ на каждую работу. Один
# LEFT JOIN LATERAL разворачивает его один раз и отдаёт обе суммы. LEFT и
# LATERAL, а не подзапрос в SELECT: у работы без evidence
# jsonb_array_elements не даёт ни строки, агрегат по пустому множеству —
# одна строка с NULL, и coalesce снаружи делает из неё 0 (раньше это же
# делал coalesce внутри каждого подзапроса).
POPULATE = f"""
INSERT INTO measurements.citation_hub_expansion
    (run_id, work_key, relation, cited_by_count, n_references)
SELECT :run, w.key, j.relation,
       coalesce(agg.cited_by, 0),
       -- referenced_works_count, не length(referenced_works): сам список
       -- в evidence не хранится (registry.Node.absorb), а счётчик OpenAlex
       -- присылает рядом. Сверено на живой базе: 438 работ, 15634 ссылки
       -- обоими способами, ноль расхождений.
       coalesce(agg.refs, 0)
FROM citation.work w
JOIN (SELECT DISTINCT ON (node_key) node_key, relation
        FROM citation.crawl_step
       WHERE action = '{CrawlAction.KEEP}' AND depth = 1
         AND node_key IS NOT NULL AND relation IS NOT NULL
       ORDER BY node_key, id) j ON j.node_key = w.key
LEFT JOIN LATERAL (
    SELECT sum((r->>'cited_by_count')::bigint)          AS cited_by,
           sum((r->>'referenced_works_count')::bigint)  AS refs
    FROM jsonb_array_elements(w.evidence->'records') r
) agg ON true
ON CONFLICT (run_id, work_key) DO UPDATE SET
    relation       = EXCLUDED.relation,
    cited_by_count = EXCLUDED.cited_by_count,
    n_references   = EXCLUDED.n_references;
"""

# Кап — величина ПРОГОНА (--hub-cap), а не константа замера: он входит в
# счёт «за капом», и записанный запрос-контракт обязан считать по тому же
# числу, что посчитал сам замер. Литерал в запросе означал бы отчёт, чей
# заголовок называет одно число, колонка под ним измерена по другому, а
# verify_query воспроизводит третье.
def verify_query(cap: int) -> str:
    """Запрос, которым проверяется вывод замера, при капе этого прогона."""
    return (
        "SELECT relation, count(*) AS nodes, sum(cited_by_count) AS citers_total, "
        "max(cited_by_count) AS worst, "
        f"count(*) FILTER (WHERE cited_by_count > {int(cap)}) AS over_cap, "
        "sum(n_references) AS refs_total "
        "FROM measurements.citation_hub_expansion "
        "WHERE run_id = (SELECT id FROM measurements.run WHERE spike = "
        f"'{SPIKE}') GROUP BY relation ORDER BY citers_total DESC;"
    )


def stats(env, run_id: int, cap: int) -> list[list[str]]:
    """Агрегат замера: по одной строке на тип связи, «за капом» — по `cap`.

    Кап приходит психал-переменной, а не подстановкой в строку: число здесь
    из командной строки, и запрос собирается тем же способом, что и run_id.
    """
    out = run_sql(
        env,
        "SELECT relation, count(*), sum(cited_by_count), max(cited_by_count), "
        "count(*) FILTER (WHERE cited_by_count > :cap), sum(n_references) "
        "FROM measurements.citation_hub_expansion WHERE run_id = :run "
        "GROUP BY relation ORDER BY sum(cited_by_count) DESC;",
        variables={"run": str(int(run_id)), "cap": str(int(cap))},
        extra_args=ROW_ARGS,
    ).stdout
    return [record.split(FIELD_SEP) for record in split_records(out)]


def worst_nodes(env, run_id: int, limit: int = 10) -> list[list[str]]:
    out = run_sql(
        env,
        "SELECT h.relation, h.cited_by_count, coalesce(w.year::text, '-'), "
        "left(coalesce(w.title, ''), 70) "
        "FROM measurements.citation_hub_expansion h JOIN citation.work w ON w.key = h.work_key "
        f"WHERE h.run_id = :run ORDER BY h.cited_by_count DESC LIMIT {int(limit)};",
        variables={"run": str(int(run_id))},
        extra_args=ROW_ARGS,
    ).stdout
    return [record.split(FIELD_SEP) for record in split_records(out)]


def run_fields(counts: list[int], rows: list[list[str]], cap: int) -> dict:
    total = sum(counts)
    return {
        "question": (
            "Сколько цитирующих затягивает расширение «вверх» от узлов depth-1 "
            "графа цитирований ИИШ и зависит ли это от типа входящей связи: "
            f"узел пришёл как цитирующий фронтир ({Relation.CITES}) или как "
            f"процитированный фронтиром ({Relation.REFERENCED})?"
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
            f"python3 pg_load_citations.py --hub-report --hub-cap {int(cap)}   "
            "# СЕТИ НЕ ТРЕБУЕТ: считает из citation.work (evidence) и из кэша "
            "corpus/cache/openalex, поэтому воспроизводится при исчерпанной квоте. "
            "Узлы depth-1 определяются по журналу citation.crawl_step "
            f"(action='{CrawlAction.KEEP}', depth=1), а не предположением "
            "«в базе только depth-1»"
        ),
        "verify_query": verify_query(cap) + "  -- ждать: "
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
        f"| связь | узлов | Σ cited_by | максимум | за капом {cap} | Σ ссылок |",
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
        f"записи), с журнальной строкой `action='{CrawlAction.HUB_SKIP}'`.",
        "",
        "Почему именно так, а не «поднять τ»: порог отбирает кандидата по "
        "релевантности, а цену создаёт цитируемость уже оставленного узла. Это "
        "разные величины, и первая второй не управляет.",
    ]
    return "\n".join(lines) + "\n"
