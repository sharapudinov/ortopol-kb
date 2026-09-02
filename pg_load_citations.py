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

from citations import calibration, frontier, hub_report, twin_pass
from citations.crawl import HUB_CAP, Snowball
from citations.http_cache import cache_for
from citations.openalex_client import OpenAlexClient, OpenAlexError, QuotaExhausted
from citations.spike_runs import (
    DryRunMeasurementsWriter,
    MeasurementsWriter,
    NothingToMeasure,
    record_calibration,
    record_hub_report,
)
from citations.inputs import (
    COVERAGE_RUN,
    corpus_document_ids,
    fresh_keys,
    known_embeddings,
    seed_matches,
)
from citations.seed_metadata import mathnet_names, zbmath_abstracts
from citation_vocab import CrawlAction
from citations.store import DryRunWriter, PostgresWriter
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
from pg_graph_common import check as graph_check
from pg_graph_common import citation_schema_exists, init_schema, project


def build_client(args, cache) -> OpenAlexClient:
    """The crawl's HTTP client, with the cache main() chose for the run."""
    return OpenAlexClient(cache=cache,
                          quota_floor=args.quota_floor,
                          max_quota_wait=args.max_quota_wait)


def do_merge_twins(env, args) -> int:
    # Same construction main() makes for the crawl: the mode's promise is
    # kept by WHICH writer exists, not by a flag consulted per statement.
    writer = DryRunWriter() if args.dry_run else PostgresWriter(env)
    merged = twin_pass.merge_twins(env, args.crawl_id or "merge-twins", writer)
    for item in merged:
        print(f"  {item['key']} -> {item['document_id']} [{item['rule']}] "
              f"(семя {item['seed_key']}): {item['title'][:64]}")
    print(f"склеено двойников наших работ: {len(merged)}"
          + (" (--dry-run, ничего не записано)" if args.dry_run else ""))
    if not args.dry_run and merged:
        print("kind после склейки: " + twin_pass.kind_census(env))
        vertices, edges = project(env)
        print(f"проекция графа: V={vertices} E={edges}")
        return graph_check(env)
    return 0


def do_hub_report(env, args, tree_root: Path, writer) -> int:
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
        record = record_hub_report(env, cache, tree_root, writer, args.hub_cap)
    except NothingToMeasure as exc:
        print(f"{exc} (кэш {cache_path})", file=sys.stderr)
        return 1
    if writer.dry:
        print("--dry-run: ничего не записано. meta.count батчей cites: "
              + ", ".join(str(c) for c in record.counts)
              + f" (сумма {sum(record.counts)}). Таблица узлов не заполнялась, "
              "поэтому ни её статистик, ни отчёта в этом режиме нет: они "
              "читаются из записанного.")
        return 0
    total = sum(int(row[1]) for row in record.rows)
    print(f"run {record.run_id} ({hub_report.SPIKE}); узлов depth-1: {total}; "
          f"отчёт: {record.report}")
    for row in record.rows:
        print("  " + " | ".join(row))
    return 0


def do_calibrate(snowball: Snowball, client, tree_root: Path, writer) -> int:
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


def do_crawl(env, snowball: Snowball, client, args) -> int:
    summary = snowball.run(args.depth)
    for depth in sorted(summary):
        print(f"  depth {depth}: " + ", ".join(f"{k}={v}" for k, v in summary[depth].items()))
    print(f"запросов OpenAlex: {client.n_requests} (из кэша: {client.n_cache_hits})")
    if args.dry_run:
        print("--dry-run: в базу ничего не записано")
        return 0
    vertices, edges = project(env)
    print(f"проекция графа: V={vertices} E={edges}")
    return graph_check(env)


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
    measurements = DryRunMeasurementsWriter() if args.dry_run else MeasurementsWriter(env)

    # Оба режима ниже считают по уже записанному и кэшу: ни семян, ни сети.
    if args.hub_report:
        return do_hub_report(env, args, data_root(), measurements)
    if args.merge_twins:
        return do_merge_twins(env, args)

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
    writer = DryRunWriter() if (args.dry_run or args.calibrate) else PostgresWriter(env)
    # Four caches in the data tree, all four chosen HERE and handed to
    # their readers as objects -- the same construction the two writers
    # above get, and for the same reason: --dry-run's promise about the tree
    # must not depend on a keyword nobody forgot (DRY_RUN_WRITES_NOTHING).
    # The fourth memoises the VECTORS, which --calibrate buys and, writing
    # no work row, leaves nowhere.
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
        return do_crawl(env, snowball, client, args)
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
