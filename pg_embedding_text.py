#!/usr/bin/env python3
"""What text a citation.work row means, written once for both producers.

citation.work.embedding has two writers, and they must agree on the text
as exactly as they agree on the model. The crawl embeds a candidate as it
scores it (citations/frontier.candidate_text, through the embedder bound
from corpus.embedding_model); pg_embed.py fills in rows that have no
vector yet -- a title edited by hand, a row written before the column
existed -- with a SQL expression. Two spellings of "title plus abstract"
produce two vectors that differ by a plausible amount rather than by an
error, and there is no per-row model or text column to catch it.

So the rule lives here, in both dialects, and both consumers import it:

  * newlines and carriage returns become spaces. Not cosmetic on either
    side -- pg_embed.py parses its rows line by line, and a multi-line
    abstract would split one row into several;
  * each part is trimmed of spaces and tabs, and an empty one contributes
    nothing (no double space, no trailing space);
  * the result is cut at MAX_CHARS. bge-m3 holds 8192 tokens; the cut is
    by characters with room to spare, so a long abstract is truncated
    rather than dropped.

The pair is held to each other by a live test over citation.work
(tests/test_embedding_text.py), not by these two paragraphs.
"""
from __future__ import annotations

MAX_CHARS = 6000

# The trim set, spelled with chr() rather than an E'' literal: this string
# travels into a psql script through a Python string, and one backslash
# convention fewer is one hazard fewer.
_TRIM = "' ' || chr(9)"
_FLAT = "replace(replace(coalesce({column},''), chr(10), ' '), chr(13), ' ')"

# NOT wrapped in left(...): pg_embed.py applies MAX_CHARS itself, the same
# way it does for every other target.
WORKS_TEXT_SQL = (
    f"btrim(btrim({_FLAT.format(column='title')}, {_TRIM})"
    f" || ' ' || "
    f"btrim({_FLAT.format(column='abstract')}, {_TRIM}), {_TRIM})"
)


def _flatten(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\n", " ").replace("\r", " ").strip(" \t")


def works_text(title: str | None, abstract: str | None) -> str:
    """The same string WORKS_TEXT_SQL produces, cut at MAX_CHARS."""
    return " ".join(part for part in (_flatten(title), _flatten(abstract))
                    if part)[:MAX_CHARS]
