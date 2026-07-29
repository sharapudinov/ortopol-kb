"""Completeness checks specific to theory/external/ (literature by others).

Split out of corpus_completeness.py rather than folded into it: the corpus
proper is checked against pdfinfo and INDEX.md, while an external source is
checked against its registry and against the regime its class must carry.
Two different questions, and the second one grows (a tier vocabulary, a
per-source verification date) while the first does not.

What it refuses, and why each refusal exists:

- a file with no registry row, or a row with no file -- the base is not a
  bookmark folder, and a source nobody can say WHY we hold is not a holding;
- an external document whose URL in the database disagrees with the registry
  -- the row is what a reader follows back to the publisher;
- an external document whose legal columns are anything other than
  external-literature / excluded. This is the one place the class is named
  as a predicate rather than merely written by a loader. It does not put
  legal knowledge into the PACKAGER (deploy/legal_profile.py still reads
  public_distribution and nothing else, invariant LEGAL_IS_DATA): it makes
  "somebody else's copyright never leaves in a public artifact" checkable by
  a command instead of resting on the loader having been run.
"""
from __future__ import annotations

from pathlib import Path

from external_registry import (
    LEGAL_CLASS,
    PUBLIC_DISTRIBUTION,
    REGISTRY_DISTRIBUTION,
    REGISTRY_DOCUMENT_ID,
    REGISTRY_FILENAME,
    REGISTRY_LEGAL_CLASS,
    RegistryError,
    load_registry,
    registry_problems,
)
from paths import EXTERNAL_SOURCE_DIR
from pg_common import run_sql

FIELD_SEP = "\x1f"

_DB_SQL = (
    "SELECT id, coalesce(source_url, ''), coalesce(legal_class, ''), "
    "coalesce(public_distribution, '') FROM corpus.documents "
    f"WHERE source_dir = '{EXTERNAL_SOURCE_DIR}' ORDER BY id;"
)


def external_rows(env: dict) -> dict[str, tuple[str, str, str]]:
    """{document_id: (source_url, legal_class, public_distribution)} for every
    row the database holds under the external tree.
    """
    out = run_sql(env, _DB_SQL, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    rows = {}
    for line in out.splitlines():
        if line.strip():
            doc_id, url, legal_class, distribution = line.split(FIELD_SEP)
            rows[doc_id] = (url, legal_class, distribution)
    return rows


def external_problems(theory_dir: Path, env: dict) -> list[str]:
    directory = theory_dir / Path(EXTERNAL_SOURCE_DIR).name
    try:
        sources = load_registry(directory)
    except RegistryError as exc:
        return [f"РЕЕСТР НЕЧИТАЕМ: {exc}"]

    problems = registry_problems(directory, sources)
    in_db = external_rows(env)
    expected = {source.document_id: (source.source_url, LEGAL_CLASS, PUBLIC_DISTRIBUTION)
                for source in sources}
    # Реестр — наш собственный документ, а не чужая работа: свой класс,
    # проверяется тем же предикатом, чтобы под external не осталось ни одной
    # строки, чей режим никто не сверил.
    if sources or REGISTRY_DOCUMENT_ID in in_db:
        expected[REGISTRY_DOCUMENT_ID] = ("", REGISTRY_LEGAL_CLASS, REGISTRY_DISTRIBUTION)

    for doc_id, (want_url, want_class, want_distribution) in sorted(expected.items()):
        row = in_db.get(doc_id)
        if row is None:
            continue  # уже сообщено как NOT INDEXED перечислителем файлов
        url, legal_class, distribution = row
        if url != want_url:
            problems.append(
                f"URL РАСХОДИТСЯ: {doc_id} — в {REGISTRY_FILENAME} {want_url!r}, "
                f"в базе {url!r} (перезагрузить pg_load_external.py)")
        if (legal_class, distribution) != (want_class, want_distribution):
            problems.append(
                f"РЕЖИМ НЕ ТОТ: {doc_id} несёт ({legal_class!r}, "
                f"{distribution!r}), обязан нести ({want_class!r}, "
                f"{want_distribution!r}) — иначе чужая работа может уехать "
                "в public-артефакт")

    problems += [f"ЛИШНЯЯ СТРОКА: документ {doc_id} числится в "
                 f"{EXTERNAL_SOURCE_DIR}, в реестре его нет"
                 for doc_id in sorted(set(in_db) - set(expected))]
    return problems
