"""Which slice of the citation graph may leave the machine, read from the
database -- the same DATA-not-code discipline LEGAL_IS_DATA applies to
corpus.documents (legal_profile.py), applied here to the crawl's own record
as a WHOLE rather than to an individual document (see
pg_schema_citation.sql's citation.public_policy and its own comment).

Absence of a decision (no row in citation.public_policy) FAILS a public
build exactly like an unclassified corpus.documents row does
(legal_profile.require_classified / UNCLASSIFIED_FAILS_BUILD): the crawl
kept titles, abstracts and citation edges of works by OTHER authors, and
shipping that by default -- or stripping it by default -- would both be the
packager deciding a question only the corpus owner can answer.

Three modes (manifest_contract.CitationMode, mirrored by
citation.public_policy's own CHECK):

  full-skeleton   citation.work (every column, abstract/evidence included)
                  + cites (with evidence) + crawl_step + public_policy.
  topology-only   citation.work with abstract/evidence forced NULL; cites
                  with evidence forced NULL; crawl_step and public_policy
                  ship whole -- both are OUR OWN journal/decision record, no
                  third-party content in either.
  none            citation is not dumped at all: no CREATE SCHEMA, no table,
                  no row. The FULL profile still carries everything
                  regardless of this table (see build_package.py's own
                  docstring: full never applies any legal/policy cut).

--policy-override on build_package.py (TEST ONLY, see its own --help) skips
the database read entirely and substitutes a caller-given mode at the call
site; this module only ever answers "what does the DATABASE say", so a test
of the override still exercises the real refusal path for the
non-overridden case.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from manifest_contract import CitationMode  # noqa: E402
from pg_common import run_sql, scalar  # noqa: E402

FIELD_SEP = "\x1f"

_SCHEMA_EXISTS_SQL = "SELECT to_regclass('citation.work') IS NOT NULL;"
_POLICY_SQL = "SELECT mode FROM citation.public_policy WHERE id = 1;"
_COUNTS_SQL = "SELECT kind, count(*) FROM citation.work GROUP BY kind ORDER BY kind;"


class CitationUnclassified(RuntimeError):
    """The citation schema's public-artifact policy is not decided (or not
    recognised) -- the packager must not guess (see module docstring).
    """


def citation_schema_exists(env: dict) -> bool:
    return scalar(env, _SCHEMA_EXISTS_SQL) == "t"


def citation_public_policy(env: dict) -> str | None:
    """The mode currently recorded, or None when citation.public_policy has
    no row. Callers that must not proceed on None use require_citation_mode.
    """
    return scalar(env, _POLICY_SQL) or None


def require_citation_mode(env: dict) -> str:
    if not citation_schema_exists(env):
        raise CitationUnclassified(
            "citation schema not found in the database -- the citation graph is part of "
            "the knowledge base, not an optional extra (see kb/CLAUDE.md); apply "
            "pg_schema_citation.sql (python3 pg_graph.py init) before building a public "
            "artifact"
        )
    mode = citation_public_policy(env)
    if mode not in CitationMode.ALL:
        raise CitationUnclassified(
            f"citation.public_policy has no usable mode (got {mode!r}) -- the public "
            "artifact's citation-graph policy has not been decided by the corpus owner "
            f"(one of {CitationMode.ALL}); write it to citation.public_policy (see "
            "pg_schema_citation.sql's comment) before building a public artifact; "
            "the packager must not guess"
        )
    return mode


def citation_counts(env: dict) -> tuple[int, int, dict[str, int]]:
    """(work_count, cites_count, {kind: count}) over the WHOLE citation
    schema -- callers decide what a given mode ships (citation_dump.py) or
    describes (manifest_probe.py); this module only reads the database.
    """
    work_n = int(scalar(env, "SELECT count(*) FROM citation.work;"))
    cites_n = int(scalar(env, "SELECT count(*) FROM citation.cites;"))
    rows = run_sql(env, _COUNTS_SQL, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
    by_kind: dict[str, int] = {}
    for line in rows.splitlines():
        if line.strip():
            kind, n = line.split(FIELD_SEP)
            by_kind[kind] = int(n)
    return work_n, cites_n, by_kind
