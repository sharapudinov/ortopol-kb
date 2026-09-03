#!/usr/bin/env python3
"""Снежный ком графа цитирований: семена -> BFS по OpenAlex -> citation.*.

Семена — работы ИИШ, опознанные в OpenAlex прогоном 85 (measurements.
citation_source_coverage). Документ без матча строки work НЕ получает:
вместо неё в журнал ложится crawl_step(action='seed-missing') — «в источнике
нет» это решение, а не пробел.

Порог релевантности τ обязателен и не имеет умолчания. Сначала его меряют:

    set -a; . ../corpus/.pgenv; set +a
    python3 pg_load_citations.py --calibrate --cache-dir <scratch>
    # -> measurements.citation_frontier_threshold + research/citation-frontier/
    #    threshold.md; вердикт (число τ) пишет оркестратор
    python3 pg_load_citations.py --tau 0.62 --depth 2 --cache-dir <scratch>

Кэш ответов (--cache-dir) — расходный: квота OpenAlex окном 1000 запросов
измерена наполовину съеденной, а калибровка и обход спрашивают одни и те же
страницы depth-1. Долговременное свидетельство — citation.work.evidence в
базе, не файлы кэша.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from citations import frontier, twin_pass
from citations.crawl import HUB_CAP, Snowball
from citations.http_cache import cache_for
from citations.openalex_client import OpenAlexClient, OpenAlexError, QuotaExhausted
from citations.spike_cli import do_calibrate, do_hub_report
from citations.spike_runs import DryRunMeasurementsWriter, MeasurementsWriter
from citations.inputs import (
    COVERAGE_RUN,
    corpus_document_ids,
    fresh_keys,
    known_embeddings,
    seed_matches,
)
from citations.seed_metadata import mathnet_names, zbmath_abstracts
from citation_vocab import CrawlAction
from citations.dry_store import DryRunWriter
from citations.store import PostgresWriter
from citations.vector_cache import VectorMemo
from paths import (
    data_root,
    default_cache_dir,
    default_corpus_dir,
    default_embedding_cache_dir,
    default_mathnet_cache_dir,
    default_zbmath_cache_dir,
)
from pg_common import PostgresUnavailable, load_pgenv
from pg_search import resolve_model
from pg_graph_common import citation_schema_exists, init_schema


def build_client(args, cache) -> OpenAlexClient:
    """The crawl's HTTP client, with the cache main() chose for the run."""
    return OpenAlexClient(cache=cache,
                          quota_floor=args.quota_floor,
                          max_quota_wait=args.max_quota_wait)


def writers_for(args, env):
    """Оба писателя run'а: графовый (citations/store.py) и measurements
    (citations/spike_runs.py). ЕДИНСТВЕННОЕ место, где режим командной
    строки превращается в объект.

    Шов держит обещание --dry-run конструкцией, а не аккуратностью, — но
    само правило «какой флаг какой объект даёт» жило в трёх местах и уже
    разошлось: обход строил графового писателя по `--dry-run OR
    --calibrate`, а склейка двойников — по одному `--dry-run`, то есть
    четвёртый режим, которому нельзя писать в citation.*, достаточно было
    добавить в одну из формул. Разница между двумя писателями настоящая
    (калибровка НЕ пишет граф, но прогон в measurements пишет — она за ним
    и затевается), и она заявлена здесь одним выражением на каждого.
    """
    graph_dry = args.dry_run or args.calibrate
    return (DryRunWriter() if graph_dry else PostgresWriter(env),
            DryRunMeasurementsWriter() if args.dry_run else MeasurementsWriter(env))


def do_merge_twins(env, crawl_id: str, writer) -> int:
    """Склейка двойников: писатель приходит объектом, как во все остальные
    режимы, и всё сказанное после спрашивается у НЕГО, а не у флага.

    Перепроекция графа — тоже запись в Postgres, поэтому она метод писателя
    (writer.project()) и вызывается БЕЗУСЛОВНО: `if not writer.dry` вернул
    бы флаг ровно туда, откуда шов его убрал. Перепись kind — чтение из
    таблицы, которую сухой писатель не заполнял, поэтому она тоже метод
    (writer.census()), как hub_stats() у шва measurements.
    """
    merged = twin_pass.merge_twins(env, crawl_id, writer)
    for item in merged:
        print(f"  {item['key']} -> {item['document_id']} [{item['rule']}] "
              f"(семя {item['seed_key']}): {item['title'][:64]}")
    print(f"склеено двойников наших работ: {len(merged)}")
    print(writer.census())
    outcome = writer.project()
    print(outcome.report)
    return outcome.code


def do_crawl(snowball: Snowball, client, writer, depth_limit: int) -> int:
    """Обход и отчёт о нём. Записал ли он что-нибудь — знает писатель.

    У флага ответ расходится с делом на первом же режиме, который строит
    DryRunWriter без --dry-run (--calibrate уже такой): печаталась бы приёмка
    живого прогона, а проекция — «верная» для графа, в который ничего не
    писали. Поэтому и перепроекция, и её вердикт приходят из
    writer.project(): режим не спрашивают, у него получают объект.
    """
    summary = snowball.run(depth_limit)
    for depth in sorted(summary):
        print(f"  depth {depth}: " + ", ".join(f"{k}={v}" for k, v in summary[depth].items()))
    print(f"запросов OpenAlex: {client.n_requests} (из кэша: {client.n_cache_hits})")
    outcome = writer.project()
    print(outcome.report)
    return outcome.code


def _journal_error(writer, crawl_id: str, depth: int, exc: Exception) -> None:
    """Одна строка журнала об оборванном прогоне -- через тот же шов, что и
    всё остальное, поэтому под --dry-run её тоже никто не пишет."""
    writer.journal([{"crawl_id": crawl_id, "depth": depth,
                     "action": CrawlAction.ERROR, "reason": str(exc)}])


def main(argv: list[str] | None = None) -> int:
    """Флаги, а не подкоманды: форма CLI записана в провенансе.

    Поля `reproduce` прогонов 89 и 93 (measurements.run) цитируют именно
    флаговую форму — это команда, которой воспроизводится замер, и
    переписать её задним числом нельзя. Взаимоисключение режимов и
    принадлежность --tau обходу выражены группой и валидацией ниже; это
    ровно то, что подкоманды дали бы структурно, без потери провенанса.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tau", type=float, default=None,
                        help="порог косинуса; обязателен для обхода, умолчания нет")
    parser.add_argument("--depth", type=int, default=2)
    # Четыре режима — альтернативы, и это заявлено парсеру, а не спрятано в
    # порядке if'ов ниже: `--hub-report --calibrate` раньше молча выполнял
    # первый и выбрасывал второй запрос, отчитавшись успехом о замере,
    # которого не просили. Обход — режим по умолчанию, у него флага нет.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true",
                      help="посчитать score всех кандидатов depth-1 и записать распределение")
    mode.add_argument("--merge-twins", action="store_true",
                      help="склеить с корпусом переводы наших же работ (сети не требует)")
    mode.add_argument("--hub-report", action="store_true",
                      help="замер цены расширения вверх по типу связи (сети не требует)")
    parser.add_argument("--hub-cap", type=int, default=HUB_CAP,
                        help="узел с cited_by_count больше этого не спрашивается вверх")
    parser.add_argument("--crawl-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="не раскрывать узлы, скачанные свежее --fresh-days")
    parser.add_argument("--fresh-days", type=int, default=7)
    parser.add_argument("--cache-dir", default=str(default_cache_dir()),
                        help="кэш ответов OpenAlex; по умолчанию corpus/cache/openalex "
                             "в дереве данных — стёртый кэш стоит суток квоты")
    parser.add_argument("--quota-floor", type=int, default=30)
    parser.add_argument("--max-quota-wait", type=float, default=900.0)
    parser.add_argument("--pgenv", type=Path, default=None)
    args = parser.parse_args(argv)

    # --tau принадлежит одному режиму — обходу. --calibrate его ИЗМЕРЯЕТ, а
    # два офлайновых режима считают по уже записанному, и требовать порог с
    # них значило бы просить число, которое они не читают.
    offline = args.merge_twins or args.hub_report
    if not offline and not args.calibrate and args.tau is None:
        parser.error("--tau обязателен для обхода: умолчания нет и не будет "
                     "(порог измеряется, а не выбирается) — сначала --calibrate; "
                     "режимам --merge-twins/--hub-report порог не нужен вовсе")

    corpus_dir = default_corpus_dir()
    try:
        env = load_pgenv(args.pgenv or (corpus_dir / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres недоступен: {exc}", file=sys.stderr)
        return 1
    # --dry-run не трогает базу ВООБЩЕ, схему включительно: применение
    # pg_schema_citation.sql — это ALTER/CREATE на живой базе, то есть ровно
    # то, чего режим обещает не делать. Схема должна уже быть, и применяет
    # её отдельная санкционированная точка входа.
    if args.dry_run:
        if not citation_schema_exists(env):
            print("схема citation не применена — python3 pg_graph.py init", file=sys.stderr)
            return 1
    else:
        init_schema(env)
    # Оба писателя — здесь и один раз (writers_for), как и все четыре кэша
    # ниже: у канала записи нет умолчания, и правило «режим -> объект»
    # живёт в одном выражении на писателя, а не в трёх по месту вызова.
    writer, measurements = writers_for(args, env)

    # Оба режима ниже считают по уже записанному и кэшу: ни семян, ни сети.
    if args.hub_report:
        return do_hub_report(args, data_root(), measurements)
    if args.merge_twins:
        return do_merge_twins(env, args.crawl_id or "merge-twins", writer)

    resolved = resolve_model(env)
    if resolved is None:
        print("corpus.embedding_model пуста: сначала python3 pg_embed.py — "
              "модель кандидатов обязана быть моделью корпуса", file=sys.stderr)
        return 1
    model, dims = resolved
    documents = corpus_document_ids(env)
    matches = seed_matches(env, COVERAGE_RUN, "openalex")
    print(f"документов ИИШ: {len(documents)}; матчей OpenAlex (run {COVERAGE_RUN}): {len(matches)}; "
          f"модель эмбеддингов: {model}/{dims}")

    crawl_id = args.crawl_id or time.strftime("%Y%m%dT%H%M%S")
    # Four caches in the data tree, all four chosen HERE and handed to their
    # readers as objects -- the same construction the two writers above get,
    # and for the same reason: --dry-run's promise about the tree must not
    # depend on a keyword nobody forgot (DRY_RUN_WRITES_NOTHING). The fourth
    # memoises the VECTORS --calibrate buys and, writing no work row, loses.
    client = build_client(args, cache_for(Path(args.cache_dir), read_only=args.dry_run))
    zbmath_cache = cache_for(default_zbmath_cache_dir(), read_only=args.dry_run)
    mathnet_cache = cache_for(default_mathnet_cache_dir(), read_only=args.dry_run)
    memo = VectorMemo(cache_for(default_embedding_cache_dir(), read_only=args.dry_run), model)
    skip = fresh_keys(env, args.fresh_days) if args.resume else frozenset()
    if skip:
        print(f"--resume: {len(skip)} узлов свежее {args.fresh_days} дней не раскрываются")

    snowball = Snowball(client, frontier.bound_embedder(model, dims, memo), writer,
                        tau=args.tau if args.tau is not None else float("inf"),
                        crawl_id=crawl_id, skip_keys=skip, hub_cap=args.hub_cap,
                        known_vectors=lambda keys: known_embeddings(env, keys))
    try:
        abstracts = zbmath_abstracts(env, documents, matches, cache=zbmath_cache,
                                     writer=writer, crawl_id=crawl_id)
        snowball.seed(documents, matches, abstracts,
                      mathnet_names(env, cache=mathnet_cache))
        print(f"семян: {len(snowball.seed_keys)}; "
              f"без матча: {len(documents) - len(matches)} (журнал seed-missing)")
        if args.calibrate:
            return do_calibrate(snowball, client, data_root(), measurements)
        return do_crawl(snowball, client, writer, args.depth)
    except QuotaExhausted as exc:
        _journal_error(writer, crawl_id, args.depth, exc)
        print(f"квота OpenAlex исчерпана: {exc}", file=sys.stderr)
        return 2
    except OpenAlexError as exc:
        # Тот же журнальный след, что у исчерпанной квоты: обход остановился
        # на полпути, и почему — знает только он сам. Код другой, потому что
        # и ответ другой: квоту пережидают, а отказ источника разбирают.
        _journal_error(writer, crawl_id, args.depth, exc)
        print(f"OpenAlex не ответил: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
