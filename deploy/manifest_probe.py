"""Gathers build_package.py's manifest.json against the live instance: counts,
the embedding model's identity (name/dims/ollama digest), and two reference
probes (fulltext, vector) smoke_test.py later reproduces against a fresh
deploy. Split out of build_package.py to keep that file to its own job
(CLI + artifact assembly) -- this one owns "what does the live DB currently
say".

Every count and every probe is recorded for the profile being built, not
for the live database as such: the public profile omits excluded documents
entirely and ships without the text of metadata-only ones, so its
documents/pages counts count only rows that artifact will contain, its
fulltext probe counts only pages whose body it carries, and its blob and
vector probes name documents it actually ships. A manifest that recorded
the live numbers would make its own artifact fail its own smoke test -- and
would describe content the package does not have.
"""
from __future__ import annotations

from datetime import datetime, timezone

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import citation_profile  # noqa: E402
import legal_profile  # noqa: E402
import pg_rank_probe  # noqa: E402
import pg_search  # noqa: E402
from manifest_contract import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    VECTOR_PROBE_QUERY,
    CitationMode,
    Key,
    Profile,
    schemas_for,
)
from ollama_registry import served_model_digest  # noqa: E402
from pg_common import scalar, scalar_row  # noqa: E402

FULLTEXT_PROBE_QUERY = "повторные средние"

# Blob probe document, per profile. The full artifact keeps 1997_sm280 (the
# historical choice, a transcribed Мат. сб. paper). The public artifact
# cannot use it: that paper is publisher-exclusive-license, so its blob is
# deliberately absent there -- the probe names a CC-BY document instead, one
# whose blob the public artifact really ships. Both are verified to exist
# with a blob at build time (gather_manifest raises otherwise), so a
# reclassification that removes either from its profile fails the build
# rather than producing a manifest nothing can satisfy.
BLOB_PROBE_DOC = "1997_sm280"
PUBLIC_BLOB_PROBE_DOC = "2009_isu34"

def blob_probe_doc(profile: str) -> str:
    return PUBLIC_BLOB_PROBE_DOC if profile == Profile.PUBLIC else BLOB_PROBE_DOC


def _citation_block(env: dict, mode: str, public: bool) -> dict:
    """MANIFEST_DESCRIBES_ARTIFACT: the counts are of the rows THIS package
    carries, not of the live schema. The public profile drops every work row
    (and every edge and journal row that names it) whose document its own
    legal cut removed, so its counts are taken with that cut applied --
    citation_content_checks.py compares exactly these numbers against the
    rows the dump turns out to contain.

    `mode` is resolved once per build by citation_profile.
    resolve_citation_mode() and handed in; this module never re-derives it.
    """
    if mode == CitationMode.NONE:
        return {Key.CITATION_MODE: mode, Key.WORK_COUNT: 0, Key.CITES_COUNT: 0, Key.WORK_BY_KIND: {}}
    work_n, cites_n, by_kind = citation_profile.citation_counts(env, shipped_only=public)
    return {Key.CITATION_MODE: mode, Key.WORK_COUNT: work_n, Key.CITES_COUNT: cites_n,
            Key.WORK_BY_KIND: by_kind}


# One round trip for the six independent scalar reads gather_manifest()
# needs before the (necessarily sequential) vector probe: each is a scalar
# subselect with no shared FROM clause, so the outer query always returns
# exactly one row even when a probed document/table is empty -- NULLs are
# checked explicitly below rather than relying on scalar_row's row-count
# guard, which can only catch a structurally wrong number of columns.
#
# WHERE id = 1 on both embedding_model reads, matching pg_search.embed_query
# and smoke_checks.check_embedding_model_dims: corpus.embedding_model is
# meant to carry exactly one row (pg_schema.sql now enforces that with
# CHECK (id = 1)), but nothing stopped a second row from existing before
# that constraint shipped, and an unfiltered read against two rows fails
# with a bare Postgres "more than one row" error instead of naming which
# model actually produced the vectors.
#
# Three reads must respect the profile. The row counts: the public artifact
# has no row at all for an excluded document, nor for its pages, so live
# counts would describe a package that does not exist. The fulltext count:
# that artifact carries no body (hence no tsv) for metadata-only documents
# either, so counting live matches would record a number its own smoke test
# could never reproduce. {shipped} and {content} are module-owned SQL
# predicates ("TRUE", or legal_profile.SHIPPED_SQL/FULL_CONTENT_SQL), never
# caller input; the pages counts join corpus.documents because the
# classification lives there, and the FK makes the join row-preserving.
_MANIFEST_SCALARS_SQL = """
SELECT
    (SELECT count(*) FROM corpus.documents WHERE {shipped}),
    (SELECT count(*) FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id
      WHERE {shipped}),
    (SELECT model FROM corpus.embedding_model WHERE id = 1),
    (SELECT dims FROM corpus.embedding_model WHERE id = 1),
    (SELECT count(*) FROM measurements.run),
    (SELECT length(source_blob) FROM corpus.documents WHERE id = :'doc'),
    (SELECT source_sha256 FROM corpus.documents WHERE id = :'doc'),
    (SELECT count(*) FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id
      WHERE p.tsv @@ phraseto_tsquery('russian', :'q') AND {content});
"""


# Lexemes, not surface forms: the fulltext side this probe must stay
# independent of stems both query and page body through the SAME 'russian'
# snowball configuration (phraseto_tsquery('russian', ...), see pg_search.py's
# TS_CONFIG), under which e.g. "полином"/"полинома"/"полиномов" are one
# lexeme. A Python regex over lowercased surface forms would show zero
# overlap for exactly that case while tsquery would still match -- comparing
# stemmed lexemes computed by Postgres itself, in one round trip, is the
# only way this guard sees what phraseto_tsquery sees. chr(31) (ASCII unit
# separator) joins the result: real Russian lexemes never contain it, unlike
# a comma.
_TOKEN_OVERLAP_SQL = """
SELECT coalesce(string_agg(DISTINCT lexeme, chr(31)), '')
FROM (
    SELECT lexeme FROM unnest(to_tsvector('russian', :'q'))
    INTERSECT
    SELECT lexeme FROM unnest(to_tsvector('russian', coalesce(
        (SELECT body FROM corpus.pages WHERE document_id = :'doc' AND page_number = :page),
        ''
    )))
) overlap;
"""


def _stemmed_token_overlap(env: dict, query: str, document_id: str, page_number: int) -> list[str]:
    raw = scalar(
        env, _TOKEN_OVERLAP_SQL,
        variables={"q": query, "doc": document_id, "page": str(int(page_number))},
    )
    return sorted(raw.split("\x1f")) if raw else []


def gather_manifest(
    env: dict, ollama_url: str, profile: str = Profile.FULL,
    citation_mode: str = CitationMode.NONE,
) -> dict:
    if profile not in Profile.ALL:
        raise ValueError(f"unknown profile {profile!r} -- expected one of {Profile.ALL}")
    public = profile == Profile.PUBLIC
    # The public build refuses to describe (let alone package) a corpus with
    # an unclassified document -- the same gate public_dump.py applies before
    # writing any data, checked here too so the build fails on the FIRST
    # database read rather than after the manifest work. The citation-schema
    # policy gate is the caller's: build_package.main() resolves the mode
    # (citation_profile.resolve_citation_mode) before it gets here and hands
    # the resolved literal to this function and to the dump alike.
    if public:
        legal_profile.require_classified(env)
    content_predicate = legal_profile.FULL_CONTENT_SQL if public else "TRUE"
    shipped_predicate = legal_profile.SHIPPED_SQL if public else "TRUE"
    probe_doc = blob_probe_doc(profile)
    (
        documents_count_str, pages_count_str, model, dims_str, runs_count_str,
        blob_len_str, blob_sha, fulltext_hits_str,
    ) = scalar_row(
        env,
        _MANIFEST_SCALARS_SQL.format(content=content_predicate, shipped=shipped_predicate),
        variables={"doc": probe_doc, "q": FULLTEXT_PROBE_QUERY},
        expected_columns=8,
    )
    if not model or not dims_str:
        raise RuntimeError(
            "corpus.embedding_model is empty -- run pg_embed.py before building the package"
        )
    if not blob_len_str or not blob_sha:
        raise RuntimeError(
            f"corpus.documents has no row for id={probe_doc!r} (blob probe document) "
            f"-- the {profile} profile's blob probe no longer matches the corpus"
        )
    documents_count = int(documents_count_str)
    pages_count = int(pages_count_str)
    dims = int(dims_str)
    runs_count = int(runs_count_str)
    blob_len = int(blob_len_str)
    fulltext_hits = int(fulltext_hits_str)

    digest, size_bytes = served_model_digest(ollama_url, model)
    legal = legal_profile.legal_summary(env)

    vec_json = pg_search.embed_query(VECTOR_PROBE_QUERY, env, ollama_url=ollama_url)
    if vec_json is None:
        raise RuntimeError(
            f"embedding service at {ollama_url} unreachable -- cannot build the vector probe"
        )
    nearest = pg_rank_probe.nearest_page(env, vec_json)
    if nearest is None:
        raise RuntimeError(
            "corpus.pages has no embedded rows -- cannot build the vector probe "
            "(run pg_embed.py against the corpus first)"
        )
    # The nearest page is found over the LIVE corpus, which includes the
    # documents the public profile omits. Recording one of those would name
    # a page the artifact does not contain -- its own smoke test would then
    # fail on a package that is otherwise correct. Refused at build time,
    # like the blob probe above, rather than left to be discovered on
    # deploy: the fix is a different probe query, and the builder is the one
    # who can choose it.
    if public and nearest["document_id"] not in legal_profile.shipped_ids(legal):
        raise RuntimeError(
            f"vector probe's nearest page belongs to {nearest['document_id']}, which the "
            "public profile does not ship (public_distribution excludes it) -- pick a probe "
            "query whose nearest page is in the artifact"
        )
    # Otherwise profile-independent by construction: every shipped page
    # carries its embedding in both profiles (only body/blob are cut), so
    # the nearest page, its rank and its distance are the same numbers
    # either way for a document both profiles carry. The overlap
    # guard below reads the live body -- the strictest of the two cases,
    # since a metadata-only page ships with an empty body and could not
    # overlap anything.
    overlap = _stemmed_token_overlap(
        env, VECTOR_PROBE_QUERY, nearest["document_id"], nearest["page_number"]
    )
    if overlap:
        raise RuntimeError(
            f"vector probe query shares content token(s) {overlap} with "
            f"{nearest['document_id']} p.{nearest['page_number']} -- pick a query/page "
            "pair with no lexical overlap, otherwise the smoke check cannot tell vector "
            "search apart from fulltext"
        )
    # The second-nearest page's distance, from the SAME ranked CTE nearest
    # came from (pg_rank_probe.runner_up_distance) -- records the margin
    # between rank 1 and rank 2 so smoke_checks.check_vector can report how
    # much of VECTOR_PROBE_DISTANCE_TOLERANCE's budget a rank swap would
    # actually need, instead of asserting a rank tolerance with no measured
    # backing. None only if corpus.pages has fewer than two embedded rows,
    # which nearest_page succeeding above already rules out in practice.
    runner_up_distance = pg_rank_probe.runner_up_distance(env, vec_json)

    return {
        Key.SCHEMA_VERSION: MANIFEST_SCHEMA_VERSION,
        Key.PROFILE: profile,
        # The dumpers read the same list (manifest_contract.schemas_for),
        # so this is what the package carries, not a parallel claim about
        # it; consumed by profile_checks.py against the dump itself and by
        # smoke_test.py, which skips the measurements check when the
        # artifact declares no measurements schema.
        Key.SCHEMAS: schemas_for(profile, citation_mode),
        Key.CITATION: _citation_block(env, citation_mode, public),
        Key.CREATED_AT: datetime.now(timezone.utc).isoformat(),
        Key.DOCUMENTS_COUNT: documents_count,
        Key.PAGES_COUNT: pages_count,
        Key.EMBEDDING_MODEL: {
            Key.MODEL: model, Key.DIMS: dims, Key.DIGEST: digest, Key.SIZE_BYTES: size_bytes,
        },
        # 0 for the public profile, and true of the artifact rather than of
        # the live database: public_dump.py dumps schema corpus only, so the
        # package contains no measurements rows at all. Our own research
        # records ship separately.
        Key.MEASUREMENTS_RUN_COUNT: 0 if public else runs_count,
        Key.LEGAL: legal,
        Key.BLOB_PROBE: {
            Key.DOCUMENT_ID: probe_doc,
            Key.BYTE_LENGTH: blob_len,
            Key.SHA256: blob_sha,
        },
        Key.FULLTEXT_PROBE: {Key.QUERY: FULLTEXT_PROBE_QUERY, Key.HITS: fulltext_hits},
        Key.VECTOR_PROBE: {
            Key.QUERY: VECTOR_PROBE_QUERY,
            Key.DOCUMENT_ID: nearest["document_id"],
            Key.PAGE_NUMBER: nearest["page_number"],
            Key.RANK: nearest["rank"],
            Key.DISTANCE: nearest["distance"],
            Key.RUNNER_UP_DISTANCE: runner_up_distance,
            # Always [] here -- the RuntimeError above refuses to record any
            # other value. Kept in the manifest so a future reader can see
            # the invariant was checked, not just assumed.
            Key.TOKEN_OVERLAP: overlap,
        },
    }
