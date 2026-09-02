#!/usr/bin/env python3
"""Загружает транскрибированные страницы в индекс со статусом `transcribed`.

ПОЧЕМУ ОТДЕЛЬНЫЙ СТАТУС. Извлечение механическое: что в файле, то и вышло.
Транскрипция — чтение изображения моделью, то есть источник ошибок, и самых
неприятных именно в формулах: перепутанный индекс выглядит ровно как верный.
Для поиска этого достаточно; для цитирования в код требуется сверка с
изображением. Смешать оба состояния значило бы потерять это различие
безвозвратно.

Источник — каталог с файлами <stem>-NN.txt, по файлу на страницу.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

from paths import default_corpus_dir
from pg_common import PostgresUnavailable, load_pgenv, run_sql
from pg_copy import copy_csv_into

STATE = "transcribed"
NOTE = (
    "transcribed from rendered page images at 150 dpi; the PDF's text layer is "
    "unrecoverable (several Type 3 fonts, no consistent ToUnicode, font attribution "
    "discarded by pdftotext). Model-read, not mechanically extracted: usable for "
    "search, requires checking against the image before being cited in code."
)


def pages_for(directory: Path, stem: str) -> list[tuple[int, str]]:
    out = []
    for f in sorted(directory.glob(f"{stem}-*.txt")):
        m = re.search(rf"{re.escape(stem)}-(\d+)\.txt$", f.name)
        if not m:
            continue
        body = f.read_text(encoding="utf-8", errors="replace").strip()
        if body:
            out.append((int(m.group(1)), body))
    return out


def load(text_dir: Path, env: dict) -> dict:
    stems = sorted({re.sub(r"-\d+\.txt$", "", f.name) for f in text_dir.glob("*-*.txt")})
    if not stems:
        raise SystemExit(f"нет файлов вида <stem>-NN.txt в {text_dir}")

    run_sql(env, """
        ALTER TABLE corpus.documents DROP CONSTRAINT IF EXISTS documents_extraction_state_check;
        ALTER TABLE corpus.documents ADD CONSTRAINT documents_extraction_state_check
            CHECK (extraction_state IN ('clean','recoded','degraded','transcribed','ocr','metadata','unreadable'));
    """)

    stats = {}
    for stem in stems:
        pages = pages_for(text_dir, stem)
        chars = sum(len(b) for _, b in pages)
        # Документ уже существует со статусом unreadable — обновляем на месте,
        # чтобы не потерять его историю и не создать дубль.
        run_sql(env, f"""
            UPDATE corpus.documents
               SET extraction_state = '{STATE}', pages_count = {len(pages)},
                   chars_extracted = {chars}, note = $note${NOTE}$note$
             WHERE id = '{stem}';
            DELETE FROM corpus.pages WHERE document_id = '{stem}';
        """)
        buf = io.StringIO()
        w = csv.writer(buf)
        for n, body in pages:
            w.writerow([stem, n, body])
        copy_csv_into(env, "corpus.pages (document_id, page_number, body)", buf.getvalue())
        stats[stem] = len(pages)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-dir", type=Path, required=True)
    ap.add_argument("--note", default=None,
                    help="почему документ транскрибирован (дефолт — про нечитаемый "
                         "текстовый слой; для документов с другой причиной, например "
                         "замена ненадёжного OCR-слоя, писать её явно)")
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)
    corpus_dir = default_corpus_dir()
    try:
        env = load_pgenv(args.pgenv or (corpus_dir / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1
    global NOTE
    if args.note:
        NOTE = args.note
    stats = load(args.text_dir, env)
    for stem, n in sorted(stats.items()):
        print(f"  {stem}: {n} страниц")
    print(f"загружено документов: {len(stats)}, страниц: {sum(stats.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
