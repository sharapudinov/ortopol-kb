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

from citations import frontier, hub_report, threshold_store, twin_pass
from citations.calibration import (
    REPORT_PATH,
    SPIKE,
    calibration_report,
    carry_over_verdict,
    run_fields,
    suggest_tau,
)
from citations.crawl import HUB_CAP, Snowball
from citations.openalex_client import OpenAlexClient, QuotaExhausted
from citations.store import (
    DryRunWriter,
    PostgresWriter,
    corpus_document_ids,
    embedding_model,
    fresh_keys,
    seed_matches,
)
from citations.mathnet import MathnetClient, mathnet_id
from citations.zbmath_client import ZbmathClient, abstract_of
from paths import default_cache_dir, default_corpus_dir, default_mathnet_cache_dir
from pg_common import PostgresUnavailable, load_pgenv, run_sql
from pg_graph import check as graph_check
from pg_graph import init_schema, project

COVERAGE_RUN = 85


def build_client(args) -> OpenAlexClient:
    return OpenAlexClient(cache_dir=Path(args.cache_dir),
                          quota_floor=args.quota_floor,
                          max_quota_wait=args.max_quota_wait)


def zbmath_abstracts(env, documents, matches) -> dict[str, tuple[str, str]]:
    """document_id -> (abstract, zbmath id) for seeds OpenAlex left blank.

    Called for every seed matched in zbMATH; the crawl decides per node
    whether it needs the fallback (OpenAlex abstract wins when it exists).
    """
    zb_matches = seed_matches(env, COVERAGE_RUN, "zbmath")
    client = ZbmathClient()
    out = {}
    for document in documents:
        if document not in matches or document not in zb_matches:
            continue
        text, types = abstract_of(client.document(zb_matches[document]))
        if text:
            out[document] = (text, zb_matches[document])
    print(f"zbMATH: рефератов добыто {len(out)} за {client.n_requests} запросов")
    return out


def mathnet_names(env) -> dict[str, tuple[list[str], list[int]]]:
    """document_id -> (titles, years) off Math-Net, both languages at once.

    The identity anchor run 85 measured as worth 13 extra matches out of 69,
    cached on the seed rows so the twin rule can use it offline afterwards.
    The five documents Math-Net does not carry contribute nothing, which is a
    fact about them, not a failure here.
    """
    rows = run_sql(
        env,
        "SELECT id, coalesce(source_url, '') FROM corpus.documents "
        "WHERE source_dir = 'theory/iis' AND extraction_state <> 'metadata' ORDER BY id;",
        extra_args=["-t", "-A", "-F", "\x1f"],
    ).stdout.strip()
    client = MathnetClient(cache_dir=default_mathnet_cache_dir())
    names = {}
    for line in rows.split("\n"):
        if not line:
            continue
        document_id, _, url = line.partition("\x1f")
        identifier = mathnet_id(url)
        if not identifier:
            continue
        titles, years = client.titles(identifier)
        if titles:
            names[document_id] = (titles, years)
    print(f"Math-Net: названий добыто для {len(names)} документов "
          f"за {client.n_requests} запросов (из кэша: {client.n_cache_hits})")
    if client.failures:
        # Not a crash: the crawl runs without the anchor. But a silent gap
        # here weakens the twin index invisibly, which already happened once.
        print(f"Math-Net НЕ ОТДАЛ {len(client.failures)} страниц — "
              f"индекс двойников неполон: {', '.join(client.failures[:10])}")
    return names


def make_embedder(model: str, dims: int):
    return lambda texts: frontier.embed_texts(texts, model, dims)


def do_calibrate(env, snowball: Snowball, client, data_root: Path) -> int:
    rows = snowball.calibrate()
    if not rows:
        print("кандидатов depth-1 нет — калибровать нечего", file=sys.stderr)
        return 1
    tau_hint = suggest_tau(rows)
    run_sql(env, threshold_store.THRESHOLD_DDL)
    run_id = threshold_store.upsert_run(env, SPIKE, run_fields(rows, tau_hint))
    written = threshold_store.insert_threshold_rows(env, run_id, rows)
    report = data_root / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        carry_over_verdict(calibration_report(rows, tau_hint, snowball.candidate_refs), report),
        encoding="utf-8")
    print(f"run {run_id} ({SPIKE}); строк порога: {written}; отчёт: {report}")
    print(f"запросов OpenAlex: {client.n_requests} (из кэша: {client.n_cache_hits})")
    print(f"рекомендация τ (не вердикт): {tau_hint:.4f}")
    return 0


def do_hub_report(env, args, data_root: Path) -> int:
    """Отрицательный результат про цену расширения вверх. Сети не требует."""
    counts = hub_report.batch_counts(Path(args.cache_dir))
    run_sql(env, hub_report.DDL)
    run_id = threshold_store.upsert_run(env, hub_report.SPIKE,
                                        hub_report.run_fields(counts, []))
    run_sql(env, hub_report.POPULATE, variables={"run": str(run_id)})
    rows = hub_report.stats(env, run_id)
    # Второй проход по строке прогона: verify_query называет ожидаемые числа,
    # а они известны только после заполнения таблицы.
    run_id = threshold_store.upsert_run(env, hub_report.SPIKE,
                                        hub_report.run_fields(counts, rows))
    run_sql(env, hub_report.POPULATE, variables={"run": str(run_id)})
    rows = hub_report.stats(env, run_id)
    report = data_root / hub_report.REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(hub_report.report(counts, rows, hub_report.worst_nodes(env, run_id),
                                        run_id, args.hub_cap), encoding="utf-8")
    total = sum(int(r[1]) for r in rows)
    print(f"run {run_id} ({hub_report.SPIKE}); узлов depth-1: {total}; отчёт: {report}")
    for row in rows:
        print("  " + " | ".join(row))
    return 0


def do_merge_twins(env, args) -> int:
    merged = twin_pass.merge_twins(env, args.crawl_id or "merge-twins",
                                   dry_run=args.dry_run)
    for item in merged:
        print(f"  {item['key']} -> {item['document_id']} [{item['rule']}] "
              f"(семя {item['seed_key']}): {item['title'][:64]}")
    print(f"склеено двойников наших работ: {len(merged)}"
          + (" (--dry-run, ничего не записано)" if args.dry_run else ""))
    if not args.dry_run and merged:
        print("kind после склейки: " + twin_pass.kind_counts(env))
        vertices, edges = project(env)
        print(f"проекция графа: V={vertices} E={edges}")
        return graph_check(env)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tau", type=float, default=None,
                        help="порог косинуса; обязателен для обхода, умолчания нет")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true",
                        help="посчитать score всех кандидатов depth-1 и записать распределение")
    parser.add_argument("--merge-twins", action="store_true",
                        help="склеить с корпусом переводы наших же работ (сети не требует)")
    parser.add_argument("--hub-report", action="store_true",
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

    offline = args.merge_twins or args.hub_report
    if not offline and not args.calibrate and args.tau is None:
        parser.error("--tau обязателен для обхода (умолчания нет); сначала --calibrate")

    corpus_dir = default_corpus_dir()
    try:
        env = load_pgenv(args.pgenv or (corpus_dir / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres недоступен: {exc}", file=sys.stderr)
        return 1
    init_schema(env)

    # Оба режима ниже считают по уже записанному и кэшу: ни семян, ни сети.
    if args.hub_report:
        return do_hub_report(env, args, corpus_dir.parent)
    if args.merge_twins:
        return do_merge_twins(env, args)

    model, dims = embedding_model(env)
    documents = corpus_document_ids(env)
    matches = seed_matches(env, COVERAGE_RUN, "openalex")
    print(f"документов ИИШ: {len(documents)}; матчей OpenAlex (run {COVERAGE_RUN}): {len(matches)}; "
          f"модель эмбеддингов: {model}/{dims}")

    crawl_id = args.crawl_id or time.strftime("%Y%m%dT%H%M%S")
    writer = DryRunWriter() if (args.dry_run or args.calibrate) else PostgresWriter(env)
    client = build_client(args)
    skip = fresh_keys(env, args.fresh_days) if args.resume else frozenset()
    if skip:
        print(f"--resume: {len(skip)} узлов свежее {args.fresh_days} дней не раскрываются")

    snowball = Snowball(client, make_embedder(model, dims), writer,
                        tau=args.tau if args.tau is not None else float("inf"),
                        crawl_id=crawl_id, skip_keys=skip, hub_cap=args.hub_cap)
    try:
        abstracts = zbmath_abstracts(env, documents, matches)
        snowball.seed(documents, matches, abstracts, mathnet_names(env))
        print(f"семян: {len(snowball.seed_keys)}; "
              f"без матча: {len(documents) - len(matches)} (журнал seed-missing)")
        if args.calibrate:
            return do_calibrate(env, snowball, client, corpus_dir.parent)
        return do_crawl(env, snowball, client, args)
    except QuotaExhausted as exc:
        writer.journal([{"crawl_id": crawl_id, "depth": args.depth, "action": "error",
                         "reason": str(exc)}])
        print(f"квота OpenAlex исчерпана: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
