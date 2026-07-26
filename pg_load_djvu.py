#!/usr/bin/env python3
"""Загружает djvu-документ с OCR-слоем в индекс со статусом `ocr`.

ПОЧЕМУ ОТДЕЛЬНЫЙ СТАТУС. Текстовый слой djvu — продукт распознавания (эпоха
FineReader), а не механического извлечения: ошибки вида «Кристоффсля» вместо
«Кристоффеля» в нём есть и их доля неизвестна. Для поиска годится; перед
цитированием в код сверять с изображением страницы (`ddjvu -page=N`).

Страницы разделяются `\\f`, как у pdftotext, — единый конвейер с PDF.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

from paths import default_corpus_dir, default_pdf_dir
from pg_common import PostgresUnavailable, copy_csv_into, load_pgenv, run_sql

STATE = "ocr"
NOTE = (
    "OCR text layer embedded in the djvu, extracted with djvutxt; recognition "
    "errors present and unquantified (e.g. 'Кристоффсля'). Usable for search; "
    "verify against the page image (ddjvu -page=N) before citing in code."
)


def doc_id(stem: str) -> str:
    """Детерминированная санация: апострофы и запятые имени файла не должны
    жить в первичном ключе, которым потом набиваются SQL-запросы руками."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def extract_pages(djvu: Path, djvutxt: str) -> list[str]:
    out = subprocess.run([djvutxt, str(djvu)], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"djvutxt failed: {out.stderr.strip()}")
    pages = out.stdout.split("\f")
    if pages and pages[-1] == "":
        pages.pop()
    return pages


def source_page_count(djvu: Path) -> int:
    """Число страниц по структуре файла — независимо от инструмента извлечения."""
    d = djvu.read_bytes()
    return sum(1 for i in range(len(d) - 12)
               if d[i:i + 4] == b"FORM" and d[i + 8:i + 12] == b"DJVU")


def load(djvu: Path, env: dict, djvutxt: str) -> dict:
    pages = extract_pages(djvu, djvutxt)
    n_source = source_page_count(djvu)
    if len(pages) != n_source:
        # Расхождение экстрактора со структурой файла — не замалчивается.
        raise SystemExit(
            f"djvutxt дал {len(pages)} страниц, в структуре файла {n_source}")

    did = doc_id(djvu.stem)
    chars = sum(len(p) for p in pages)

    run_sql(env, f"DELETE FROM corpus.documents WHERE id = $q${did}$q$;")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([did, djvu.name, STATE, "local_corpus", len(pages), chars, NOTE])
    copy_csv_into(env, "corpus.documents (id, filename, extraction_state, "
                       "source_tier, pages_count, chars_extracted, note)", buf.getvalue())

    buf = io.StringIO()
    w = csv.writer(buf)
    for n, body in enumerate(pages, start=1):
        w.writerow([did, n, body])
    copy_csv_into(env, "corpus.pages (document_id, page_number, body)", buf.getvalue())
    return {"id": did, "pages": len(pages), "chars": chars}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--djvu", type=Path, default=None,
                    help="путь к djvu; по умолчанию — единственный djvu в theory/iis/")
    ap.add_argument("--djvutxt", default=shutil.which("djvutxt") or "djvutxt")
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)

    djvu = args.djvu
    if djvu is None:
        found = sorted(default_pdf_dir().glob("*.djvu"))
        if len(found) != 1:
            raise SystemExit(f"ожидался ровно один djvu в {default_pdf_dir()}, найдено {len(found)}")
        djvu = found[0]

    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1
    r = load(djvu, env, args.djvutxt)
    print(f"{r['id']}: {r['pages']} страниц, {r['chars']} символов, state={STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
