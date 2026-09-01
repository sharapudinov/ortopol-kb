#!/usr/bin/env python3
"""Заполняет pages.embedding векторами bge-m3 через локальную ollama.

Идемпотентен: берёт только строки, где embedding IS NULL, поэтому прерванный
прогон продолжается с того же места, а повторный запуск ничего не пересчитывает.

Никаких зависимостей: HTTP через urllib, Postgres через psql — драйвера Postgres
в системе нет, и ради одного скрипта он не заводится.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:5471/api/embed"
MODEL = "bge-m3"
DIMS = 1024
# bge-m3 держит 8192 токена; режем по символам с большим запасом, чтобы длинная
# страница не отбрасывалась молча.
MAX_CHARS = 6000
BATCH = 16


def psql(sql: str, tuples_only: bool = True) -> str:
    """Выполняет SQL через psql. Возвращает stdout."""
    args = ["psql", "-v", "ON_ERROR_STOP=1", "-q"]
    if tuples_only:
        args += ["-tA"]
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as fh:
        fh.write(sql)
        path = fh.name
    try:
        r = subprocess.run(args + ["-f", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"psql: {r.stderr.strip()}")
        return r.stdout
    finally:
        Path(path).unlink(missing_ok=True)


def embed(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        OLLAMA, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.load(resp)
    vecs = out["embeddings"]
    for v in vecs:
        if len(v) != DIMS:
            raise RuntimeError(f"ожидалось {DIMS} измерений, пришло {len(v)}")
    return vecs


# Каждый вид записи, который может стать результатом, обязан нести семантический
# ключ — иначе он находится только тем, кто уже знает нужное слово. Добавление
# нового вида записи означает добавление строки СЮДА, а не отдельного скрипта.
TARGETS = {
    # имя: (таблица, выражение текста для вектора, предикат «есть содержание»)
    # Предикат нужен, потому что пустая страница семантического содержания не несёт
    # и ключ ей не положен, а у measurements.run колонки body нет вовсе.
    "pages": ("corpus.pages",
              "replace(replace(body, E'\\n', ' '), E'\\r', ' ')",
              "btrim(body) <> ''"),
    "runs": (
        "measurements.run",
        # Смысл прогона — в вопросе, вердикте и в том, что он ИСКЛЮЧАЕТ.
        # Переводы строк внутри полей вычищаются, как у pages: парсер ниже
        # построчный, и многострочный вердикт иначе рвёт разбор id|text.
        "replace(replace(coalesce(question,'')||' '||coalesce(verdict,'')||' '"
        "||coalesce(rules_out,'')||' '||coalesce(arbiter,''), E'\n', ' '), E'\r', ' ')",
        "true",
    ),
}


def pending(table: str, content_pred: str = "true") -> int:
    return int(psql(
        f"select count(*) from {table} "
        f"where embedding is null and ({content_pred});").strip())


def missing_semantic_key() -> list[tuple[str, int]]:
    """Записи-результаты без семантического ключа. Пустой список — инвариант держится."""
    return [
        (name, n)
        for name, (table, _, pred) in TARGETS.items()
        if (n := pending(table, pred)) > 0
    ]


def embed_target(name: str) -> int:
    table, text_expr, content_pred = TARGETS[name]
    total = pending(table, content_pred)
    if total == 0:
        print(f"{name}: все записи уже несут семантический ключ")
        return 0
    print(f"{name}: к обсчёту {total}, модель {MODEL}, партиями по {BATCH}")

    done, started = 0, time.monotonic()
    while True:
        rows = psql(
            f"select id, left({text_expr}, {MAX_CHARS}) from {table} "
            f"where embedding is null and ({content_pred}) "
            f"order by id limit {BATCH};"
        ).strip()
        if not rows:
            break
        pairs = []
        for line in rows.split("\n"):
            pid, _, text = line.partition("|")
            if text.strip():
                pairs.append((int(pid), text))
        if not pairs:
            # Все оставшиеся записи с пустым текстом — вектор им не из чего строить.
            break

        vecs = embed([t for _, t in pairs])
        updates = "\n".join(
            f"update {table} set embedding = '{json.dumps(v)}' where id = {pid};"
            for (pid, _), v in zip(pairs, vecs)
        )
        psql("begin;\n" + updates + "\ncommit;")

        done += len(pairs)
        rate = done / max(time.monotonic() - started, 1e-9)
        print(f"  {name}: {done}/{total}  ({rate:.1f} зап/с)", flush=True)
    return done


def main() -> int:
    which = sys.argv[1:] or list(TARGETS)
    for name in which:
        if name not in TARGETS:
            print(f"неизвестная цель: {name}; известны {list(TARGETS)}")
            return 2
        embed_target(name)

    gaps = missing_semantic_key()
    if gaps:
        # Не падаем: пустой текст встречается легитимно. Но молчать нельзя —
        # запись без ключа находится только тем, кто знает точное слово.
        print("БЕЗ СЕМАНТИЧЕСКОГО КЛЮЧА:", ", ".join(f"{n}: {c}" for n, c in gaps))
    else:
        print("все записи-результаты несут семантический ключ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
