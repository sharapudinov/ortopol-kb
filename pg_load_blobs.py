#!/usr/bin/env python3
"""Кладёт сами исходники (pdf/djvu/md) в corpus.documents.source_blob + sha256.

Зачем блоб, а не путь: docker-пакет базы разворачивается на машине,
где theory/ нет — путь там битая ссылка по построению, а сверка транскрипции с
изображением страницы невозможна. С блобом база самодостаточна: исходник
рендерится из неё самой. Хеш делает подмену/порчу файла на диске обнаружимой —
это проверяет corpus_completeness.py.

Идемпотентен: документ пропускается, если sha256 диска совпадает с базой.
Байты передаются клиентски (hex + decode) — сервер в контейнере и файлов
хоста не видит, pg_read_binary_file здесь принципиально не работает.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from paths import data_root, default_corpus_dir
from pg_common import FIELD_SEP, PostgresUnavailable, load_pgenv, run_sql



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    rows = run_sql(env, "SELECT id, source_path, coalesce(source_sha256,'') "
                        "FROM corpus.documents ORDER BY id;",
                   extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    repo = data_root()
    loaded = skipped = missing = 0
    for r in rows.splitlines():
        if not r.strip():
            continue
        did, spath, db_sha = r.split(FIELD_SEP)
        f = repo / spath
        if not f.exists():
            print(f"  НЕТ ФАЙЛА: {did} -> {spath}", file=sys.stderr)
            missing += 1
            continue
        data = f.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha == db_sha:
            skipped += 1
            continue
        run_sql(env,
                f"UPDATE corpus.documents SET "
                f"source_blob = decode('{data.hex()}', 'hex'), "
                f"source_sha256 = '{sha}' WHERE id = $q${did}$q$;")
        loaded += 1
    total = run_sql(env, "SELECT pg_size_pretty(sum(length(source_blob))) "
                         "FROM corpus.documents;", extra_args=["-t", "-A"]).stdout.strip()
    print(f"загружено: {loaded}, без изменений: {skipped}, без файла: {missing}; "
          f"объём блобов: {total}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
