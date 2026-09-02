#!/usr/bin/env python3
"""Проверка полноты корпуса: каждый файл под theory/ классифицирован и учтён.

Это команда, делающая утверждение «корпус проиндексирован» предикатом, а не
впечатлением. Правила жёсткие:

- перечисляется КАЖДЫЙ файл под theory/, без исключений;
- файл, не подошедший ни под одно правило, РОНЯЕТ проверку — новый тип файла не
  может проскользнуть молча, он обязан быть либо включён, либо исключён с причиной;
- для каждого включённого документа число страниц в индексе сверяется с источником
  НЕЗАВИСИМЫМ инструментом (pdfinfo для PDF; разбор FORM:DJVU-чанков для djvu);
- фантомные строки (документ в базе, файла на диске нет) — тоже падение.

История, ради которой это написано: покрытие здесь трижды оказывалось меньше
заявленного (95.9% выдавали себя за полноту), и каждый раз дыру находил заказчик.

Выход 0 = полнота доказана. Любой другой код = утверждать её нельзя.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from external_registry import REGISTRY_FILENAME
from external_checks import external_problems
from citation_checks import citation_problems, citation_summary
from paths import data_root, default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"

# Правила классификации, в порядке применения. Первое совпавшее решает.
# (предикат по относительному пути, вердикт, причина/вид)
def classify(rel: str) -> tuple[str, str]:
    """-> ('include-pdf'|'include-djvu'|'include-metadata'|'include-external'
           |'exclude'|'unclassified', reason)"""
    if rel.startswith("iis/wiki/"):
        return "exclude", "wiki side-branch (non-mathematical); theory/iis/wiki/GUIDE.md"
    if re.fullmatch(r"iis/[^/]+\.pdf", rel):
        return "include-pdf", "source paper/monograph"
    if re.fullmatch(r"iis/[^/]+\.djvu", rel):
        return "include-djvu", "source monograph (OCR layer)"
    if rel in ("iis/INDEX.md", "iis/THEMES.md"):
        return "include-metadata", "hand-curated corpus metadata"
    # Внешняя литература: чужой автор, свой каталог, свой загрузчик и свой
    # правовой режим (external_registry.py). Реестр обязателен — источник без
    # ответа «зачем взят» база не хранит.
    if rel == f"external/{REGISTRY_FILENAME}":
        return "include-metadata", "hand-curated external sources registry"
    if re.fullmatch(r"external/[^/]+\.pdf", rel):
        return "include-external", f"external literature; must have a {REGISTRY_FILENAME} row"
    if re.fullmatch(r"external/[^/]+\.md", rel):
        return "include-external", f"external bibliography record; {REGISTRY_FILENAME} row"
    return "unclassified", "no rule matched — decide: include or exclude WITH a reason"


def pdf_pages(path: Path) -> int | None:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else None


def djvu_pages(path: Path) -> int:
    d = path.read_bytes()
    return sum(1 for i in range(len(d) - 12)
               if d[i:i + 4] == b"FORM" and d[i + 8:i + 12] == b"DJVU")


def expected_id(rel: str, kind: str) -> str:
    stem = Path(rel).stem
    if kind == "include-djvu":
        return re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return stem  # pdf и metadata: id = stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theory-dir", type=Path, default=data_root() / "theory")
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"FAIL: Postgres unavailable: {exc}", file=sys.stderr)
        return 2
    if not check_postgres_available(env):
        print("FAIL: Postgres not reachable", file=sys.stderr)
        return 2

    rows = run_sql(env,
                   "SELECT id, pages_count, source_path, coalesce(source_url,''), "
                   "coalesce(source_sha256,''), (source_blob IS NOT NULL) "
                   "FROM corpus.documents;",
                   extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    in_db, src_path, src_url, src_sha, has_blob = {}, {}, {}, {}, {}
    for r in rows.splitlines():
        if not r.strip():
            continue
        did, pc, sp, su, sha, blob = r.split(FIELD_SEP)
        in_db[did], src_path[did], src_url[did] = int(pc), sp, su
        src_sha[did], has_blob[did] = sha, blob == "t"

    problems: list[str] = []
    included_ids: set[str] = set()
    counts = {"include-pdf": 0, "include-djvu": 0, "include-metadata": 0,
              "include-external": 0, "exclude": 0}

    for f in sorted(args.theory_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(args.theory_dir))
        kind, reason = classify(rel)
        if kind == "unclassified":
            problems.append(f"UNCLASSIFIED: {rel} — {reason}")
            continue
        counts[kind] += 1
        if kind == "exclude":
            continue

        did = expected_id(rel, kind)
        included_ids.add(did)
        if did not in in_db:
            problems.append(f"NOT INDEXED: {rel} (ожидался документ id={did})")
            continue
        if kind == "include-pdf" or (kind == "include-external" and f.suffix == ".pdf"):
            src = pdf_pages(f)
            if src is not None and src != in_db[did]:
                problems.append(f"PAGE MISMATCH: {rel}: источник {src}, индекс {in_db[did]}")
        elif kind == "include-djvu":
            src = djvu_pages(f)
            if src != in_db[did]:
                problems.append(f"PAGE MISMATCH: {rel}: источник {src}, индекс {in_db[did]}")
        elif in_db[did] < 1:
            # metadata и библиографические записи: одна страница минимум,
            # иначе документ есть, а содержания у него нет.
            problems.append(f"EMPTY METADATA: {rel}")

    for did in sorted(set(in_db) - included_ids):
        problems.append(f"PHANTOM ROW: документ {did} в базе, файла на диске нет")

    # Прослеживаемость: путь из базы обязан указывать на существующий файл,
    # а статьи, перечисленные в INDEX.md, — нести канонический URL.
    repo = args.theory_dir.parent
    # Только строки INDEX.md, у которых mathnet-ссылка ЕСТЬ: у монографий её нет,
    # и требовать URL там — требовать несуществующее. Тот же regex, что в
    # pg_source_urls.py, — расхождение шаблонов дало бы ложные NO SOURCE URL.
    index_stems = set()
    index_md = args.theory_dir / "iis" / "INDEX.md"
    if index_md.exists():
        row = re.compile(r"\|\s*\d{4}\s*\|\s*`([^`]+\.pdf)`\s*\|.*?\|\s*\[[^\]]+\]\((https?://[^)]+)\)")
        index_stems = {Path(m.group(1)).stem for line in
                       index_md.read_text(encoding="utf-8").splitlines()
                       if (m := row.search(line))}
    for did in sorted(included_ids & set(in_db)):
        f = repo / src_path[did]
        if not f.exists():
            problems.append(f"BROKEN SOURCE PATH: {did} -> {src_path[did]}")
        elif not has_blob[did]:
            problems.append(f"NO BLOB: {did} — база не самодостаточна (pg_load_blobs.py)")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != src_sha[did]:
            # Файл на диске разошёлся с каноническим блобом: либо источник
            # подменён/испорчен, либо блоб устарел. Обе версии молча жить не могут.
            problems.append(f"HASH MISMATCH: {did} — диск и база расходятся; "
                            f"решить, что канонично, и перегрузить блоб")
        if did in index_stems and not src_url[did]:
            problems.append(f"NO SOURCE URL: {did} есть в INDEX.md, ссылки в базе нет "
                            f"(прогнать pg_source_urls.py)")

    # Внешняя литература: реестр против диска, реестр против базы, и правовой
    # режим класса как предикат (external_checks.py объясняет, почему это не
    # противоречит LEGAL_IS_DATA).
    problems += external_problems(args.theory_dir, env)

    # Citation graph: its own structure (a graph, not a file tree), its own
    # predicates (citation_checks.py explains each one).
    problems += citation_problems(env)

    dangling = run_sql(env,
        "SELECT r.spike || ' -> ' || e.doc_id "
        "FROM measurements.run r, unnest(r.evidence_docs) AS e(doc_id) "
        "WHERE NOT EXISTS (SELECT 1 FROM corpus.documents d WHERE d.id = e.doc_id);",
        extra_args=["-t", "-A"]).stdout.strip()
    for line in dangling.splitlines():
        if line.strip():
            problems.append(f"DANGLING EVIDENCE: {line.strip()} — находка ссылается "
                            f"на документ, которого нет в корпусе")

    missing_keys = run_sql(env,
        "SELECT count(*) FROM corpus.pages "
        "WHERE embedding IS NULL AND btrim(body) <> '';",
        extra_args=["-t", "-A"]).stdout.strip()
    if missing_keys != "0":
        problems.append(f"NO SEMANTIC KEY: {missing_keys} непустых страниц без вектора")

    total_included = sum(counts[k] for k in counts if k.startswith("include"))
    print(f"опись: {total_included} включено "
          f"({counts['include-pdf']} pdf, {counts['include-djvu']} djvu, "
          f"{counts['include-metadata']} metadata, "
          f"{counts['include-external']} external), "
          f"{counts['exclude']} исключено с причиной")
    print(citation_summary(env))
    if problems:
        print(f"\nПОЛНОТА НЕ ДОКАЗАНА — {len(problems)} проблем(ы):")
        for p in problems:
            print(" ", p)
        return 1
    print("полнота доказана: каждый файл классифицирован, каждый документ в индексе, "
          "страницы сходятся с источником, семантические ключи и ссылки на исходники "
          "на месте")
    return 0


if __name__ == "__main__":
    sys.exit(main())
