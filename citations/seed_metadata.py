#!/usr/bin/env python3
"""Что семена знают о себе помимо OpenAlex: рефераты zbMATH и названия
Math-Net.

Обе величины устанавливают ПРОГОН, а не ищутся по ходу обхода, поэтому
живут рядом с inputs.py, а не в CLI: чтения — через inputs.py, сеть — через
клиенты (zbmath_client / mathnet), запись — через writer-шов. Модуль ничего
не знает про argparse, и его можно позвать из чего угодно — например из
повторного пересева, у которого командной строки нет.

Разделение внутри пакета то же, что объявлено в __init__: клиент отвечает
за HTTP, inputs — за реляционные чтения, здесь — порядок, в котором они
складываются в ответ, и то, что об этом сообщается человеку.
"""
from __future__ import annotations

from paths import default_mathnet_cache_dir, default_zbmath_cache_dir

from . import journal
from .inputs import (
    COVERAGE_RUN,
    corpus_seed_documents,
    seed_matches,
    stored_zbmath_abstracts,
)
from .mathnet import MathnetClient, mathnet_id
from .zbmath_client import ZbmathClient, ZbmathUnavailable, abstract_of


def zbmath_abstracts(env, documents, matches, writer=None, crawl_id=None,
                     log=print) -> dict[str, tuple[str, str]]:
    """document_id -> (abstract, zbmath id) for seeds OpenAlex left blank.

    Called for every seed matched in zbMATH; the crawl decides per node
    whether it needs the fallback (OpenAlex abstract wins when it exists).

    A seed simply missing from zbMATH contributes nothing and needs no row.
    A seed whose FETCH failed is a different thing entirely -- we never
    learned whether it has a review -- and it goes into the journal as
    action='error' (citations/journal.zbmath_error), so a later reader can
    tell the two apart instead of seeing one indistinguishable blank.

    Two layers keep the network out of a repeat run: an abstract already
    stored on the seed's own citation.work row (with zbMATH recorded as its
    provenance) is used as it stands, and everything else goes through the
    client's disk cache. What is left is genuinely new.
    """
    zb_matches = seed_matches(env, COVERAGE_RUN, "zbmath")
    stored = stored_zbmath_abstracts(env)
    client = ZbmathClient(cache_dir=default_zbmath_cache_dir())
    out, errors = {}, []
    for document in documents:
        if document not in matches or document not in zb_matches:
            continue
        if document in stored:
            out[document] = (stored[document], zb_matches[document])
            continue
        try:
            record = client.document(zb_matches[document])
        except ZbmathUnavailable as exc:
            errors.append(journal.zbmath_error(crawl_id, document, zb_matches[document], str(exc)))
            continue
        text, _types = abstract_of(record)
        if text:
            out[document] = (text, zb_matches[document])
    if errors and writer is not None:
        writer.journal(errors)
    log(f"zbMATH: рефератов добыто {len(out)} за {client.n_requests} запросов "
        f"(из кэша: {client.n_cache_hits}, уже в базе: {len(stored)})")
    if client.failures:
        log(f"zbMATH НЕ ОТВЕТИЛ по {len(client.failures)} запросам — "
            f"это не «реферата нет», а «мы не узнали»: {', '.join(client.failures[:10])}")
    return out


def mathnet_names(env, log=print) -> dict[str, tuple[list[str], list[int]]]:
    """document_id -> (titles, years) off Math-Net, both languages at once.

    The identity anchor run 85 measured as worth 13 extra matches out of 69,
    cached on the seed rows so the twin rule can use it offline afterwards.
    The five documents Math-Net does not carry contribute nothing, which is a
    fact about them, not a failure here.

    The seed document set comes from inputs.corpus_seed_documents(), the one
    place that predicate is written: a second copy of it here would let "what
    counts as a seed document" change for the crawl and not for its Math-Net
    anchor, silently producing seeds with no title anchor -- which is exactly
    what the twin rule depends on.
    """
    client = MathnetClient(cache_dir=default_mathnet_cache_dir())
    names = {}
    for document_id, url in corpus_seed_documents(env):
        identifier = mathnet_id(url)
        if not identifier:
            continue
        titles, years = client.titles(identifier)
        if titles:
            names[document_id] = (titles, years)
    log(f"Math-Net: названий добыто для {len(names)} документов "
        f"за {client.n_requests} запросов (из кэша: {client.n_cache_hits})")
    if client.failures:
        # Not a crash: the crawl runs without the anchor. But a silent gap
        # here weakens the twin index invisibly, which already happened once.
        log(f"Math-Net НЕ ОТДАЛ {len(client.failures)} страниц — "
            f"индекс двойников неполон: {', '.join(client.failures[:10])}")
    return names
