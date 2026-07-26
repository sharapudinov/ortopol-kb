#!/usr/bin/env python3
"""Проставляет source_url документам корпуса из таблицы INDEX.md.

INDEX.md — курируемый источник соответствия «файл -> страница Math-Net.Ru».
Прослеживаемость: находка в индексе обязана вести к исходнику в один переход,
а не через конвенцию в голове. Идемпотентен; перезапуск после правки INDEX.md
обновляет ссылки.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from paths import default_corpus_dir, default_pdf_dir
from pg_common import PostgresUnavailable, load_pgenv, run_sql

ROW = re.compile(r"\|\s*\d{4}\s*\|\s*`([^`]+\.pdf)`\s*\|.*?\|\s*\[[^\]]+\]\((https?://[^)]+)\)")


def parse_index(index_md: Path) -> dict[str, str]:
    mapping = {}
    for line in index_md.read_text(encoding="utf-8").splitlines():
        m = ROW.search(line)
        if m:
            mapping[Path(m.group(1)).stem] = m.group(2)
    return mapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=default_pdf_dir() / "INDEX.md")
    ap.add_argument("--pgenv", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        env = load_pgenv(args.pgenv or (default_corpus_dir() / ".pgenv"))
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    mapping = parse_index(args.index)
    if not mapping:
        print("FAIL: в INDEX.md не распознано ни одной строки с mathnet-ссылкой",
              file=sys.stderr)
        return 1
    updates = "\n".join(
        f"update corpus.documents set source_url = $u${url}$u$ where id = $i${stem}$i$;"
        for stem, url in mapping.items()
    )
    run_sql(env, "begin;\n" + updates + "\ncommit;")
    n = run_sql(env, "SELECT count(*) FROM corpus.documents WHERE source_url IS NOT NULL;",
                extra_args=["-t", "-A"]).stdout.strip()
    print(f"распознано в INDEX.md: {len(mapping)}, документов со ссылкой в базе: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
