#!/usr/bin/env python3
"""Загружает рукописные метаданные корпуса (INDEX.md, THEMES.md) со статусом `metadata`.

Зачем они в индексе: на вопрос «какие работы про смешанные ряды» таблица
год|файл|название|mathnet-id и восемь размеченных направлений отвечают точнее,
чем полнотекст по 2226 страницам. Это курируемое знание О корпусе, и оно должно
находиться теми же двумя ключами, что и сам корпус.

Один md-файл = один документ с одной страницей: у этих файлов нет постраничной
структуры, а дробить таблицу — терять её связность.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

from paths import default_corpus_dir, default_pdf_dir
from pg_common import PostgresUnavailable, copy_csv_into, load_pgenv, run_sql

STATE = "metadata"
NOTE = ("hand-curated corpus metadata (not extracted from a PDF); "
        "maintained by the project, edit the source .md and re-run this loader")
DEFAULT_FILES = ("INDEX.md", "THEMES.md")


def load(files: list[Path], env: dict) -> list[tuple[str, int]]:
    docs, pages = io.StringIO(), io.StringIO()
    dw, pw = csv.writer(docs), csv.writer(pages)
    out = []
    for f in files:
        body = f.read_text(encoding="utf-8")
        did = f.stem  # 'INDEX', 'THEMES' — с mathnet-стемами не пересекаются
        run_sql(env, f"DELETE FROM corpus.documents WHERE id = $q${did}$q$;")
        dw.writerow([did, f.name, STATE, "local_corpus", 1, len(body), NOTE])
        pw.writerow([did, 1, body])
        out.append((did, len(body)))
    copy_csv_into(env, "corpus.documents (id, filename, extraction_state, "
                       "source_tier, pages_count, chars_extracted, note)", docs.getvalue())
    copy_csv_into(env, "corpus.pages (document_id, page_number, body)", pages.getvalue())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path,
                    default=[default_pdf_dir() / n for n in DEFAULT_FILES])
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1
    for did, chars in load(list(args.files), env):
        print(f"{did}: {chars} символов, state={STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
