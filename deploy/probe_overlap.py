"""The guard that keeps the vector probe a VECTOR probe.

Split out of manifest_probe.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that module gathers what the live instance says about the
package; this one answers a different question about one of its probes --
does the probe query share a stemmed lexeme with the page it is supposed to
find? If it does, the smoke test on the recipient's side cannot tell vector
search apart from fulltext, and the build refuses to record such a pair.

Lexemes, not surface forms: the fulltext side this probe must stay
independent of stems both query and page body through the SAME 'russian'
snowball configuration (phraseto_tsquery('russian', ...), see pg_search.py's
TS_CONFIG), under which e.g. "полином"/"полинома"/"полиномов" are one
lexeme. A Python regex over lowercased surface forms would show zero
overlap for exactly that case while tsquery would still match -- comparing
stemmed lexemes computed by Postgres itself, in one round trip, is the
only way this guard sees what phraseto_tsquery sees. The ASCII unit
separator pg_common.FIELD_SEP names joins the result: real Russian lexemes
never contain it, unlike a comma."""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from pg_common import FIELD_SEP, scalar  # noqa: E402

TOKEN_OVERLAP_SQL = f"""
SELECT coalesce(string_agg(DISTINCT lexeme, chr({ord(FIELD_SEP)})), '')
FROM (
    SELECT lexeme FROM unnest(to_tsvector('russian', :'q'))
    INTERSECT
    SELECT lexeme FROM unnest(to_tsvector('russian', coalesce(
        (SELECT body FROM corpus.pages WHERE document_id = :'doc' AND page_number = :page),
        ''
    )))
) overlap;
"""


def stemmed_token_overlap(env: dict, query: str, document_id: str, page_number: int) -> list[str]:
    raw = scalar(
        env, TOKEN_OVERLAP_SQL,
        variables={"q": query, "doc": document_id, "page": str(int(page_number))},
    )
    return sorted(raw.split(FIELD_SEP)) if raw else []
