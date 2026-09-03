"""Gathers build_package.py's manifest.json against the live instance: the
embedding model's identity (name/dims/ollama digest) and two reference
probes (fulltext, vector) smoke_test.py later reproduces against a fresh
deploy. Split out of build_package.py to keep that file to its own job
(CLI + artifact assembly) -- this one owns "what does the live DB currently
say". Its two neighbours own the rest: manifest_citation.py the graph
block's census (the one number no COPY stream can produce), and
manifest_rows.py every number that is a row count of the dump, stamped
after the dump instead of gathered before it.

Every probe is recorded for the profile being built, not for the live
database as such: the public profile omits excluded documents entirely and
ships without the text of metadata-only ones, so its fulltext probe counts
only pages whose body it carries, and its blob and vector probes name
documents it actually ships. A manifest that recorded the live numbers
would make its own artifact fail its own smoke test -- and would describe
content the package does not have.

The ROW COUNTS are not gathered here at all any more, for the same reason
one step further: how many rows an artifact carries is answered by the
artifact. They are stamped afterwards from what the dump wrote
(manifest_rows.py); the keys are declared here as 0 so the SHAPE of the
manifest is still written in one place.
"""
from __future__ import annotations

from datetime import datetime, timezone

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import legal_profile  # noqa: E402
import pg_rank_probe  # noqa: E402
import pg_search  # noqa: E402
from manifest_keys import MANIFEST_SCHEMA_VERSION, Key  # noqa: E402
from manifest_citation import citation_block  # noqa: E402
from manifest_contract import Profile, schemas_for  # noqa: E402
from ollama_registry import served_model_digest  # noqa: E402
from pg_common import scalar_row  # noqa: E402
from probe_overlap import stemmed_token_overlap  # noqa: E402
from probe_query import VECTOR_PROBE_QUERY  # noqa: E402

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


# One round trip for the independent scalar reads gather_manifest()
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
# The documents/pages counts are NOT here any more: a row count of the
# artifact is answered by the artifact (manifest_rows.py), and this query
# runs before the dump exists. The fulltext count stays and must respect the
# profile -- the public artifact carries no body (hence no tsv) for
# metadata-only documents, so counting live matches would record a number
# its own smoke test could never reproduce. {content} is a module-owned SQL
# predicate ("TRUE", or legal_profile.FULL_CONTENT_SQL), never caller input;
# it joins corpus.documents because the classification lives there, and the
# FK makes the join row-preserving.
_MANIFEST_SCALARS_SQL = """
SELECT
    (SELECT model FROM corpus.embedding_model WHERE id = 1),
    (SELECT dims FROM corpus.embedding_model WHERE id = 1),
    (SELECT count(*) FROM measurements.run),
    (SELECT length(source_blob) FROM corpus.documents WHERE id = :'doc'),
    (SELECT source_sha256 FROM corpus.documents WHERE id = :'doc'),
    (SELECT count(*) FROM corpus.pages p JOIN corpus.documents d ON d.id = p.document_id
      WHERE p.tsv @@ phraseto_tsquery('russian', :'q') AND {content});
"""


def gather_manifest(
    env: dict, ollama_url: str, profile: str = Profile.FULL,
    *, citation_mode: str, policy_source: str,
) -> dict:
    """The manifest for `profile`, read against the live instance.

    citation_mode and policy_source are required and keyword-only: they are
    the two halves of ONE resolution (citation_profile.resolve_citation_mode),
    and this function can arrive at neither. A default would make omission
    indistinguishable from a decision -- and the decision it would fabricate
    is the owner's own: mode `none` with provenance `owner` is a
    self-consistent package asserting the owner said the citation graph does
    not travel, which every recipient-side gate then certifies. Refusing is
    the only answer a producer can give about a fact it does not hold -- the
    polarity require_classified and require_citation_mode apply to theirs.
    """
    if profile not in Profile.ALL:
        raise ValueError(f"unknown profile {profile!r} -- expected one of {Profile.ALL}")
    public = profile == Profile.PUBLIC
    # No build describes (let alone packages) a corpus with an unclassified
    # document -- the same gate public_dump.py applies before writing any
    # data, applied here too so a build fails on the FIRST database read
    # rather than after the manifest work.
    #
    # BOTH profiles, for the reason classification_gate.py gives one level
    # down: the full profile decides nothing by the classification, but the
    # checker that travels INSIDE the artifact is profile-blind --
    # corpus_content_checks.check_classification_complete demands
    # legal.unclassified_documents == 0 whichever profile produced the
    # package, and legal_summary() stamps that number into every manifest.
    # Gated on the public path alone, a full build over an unclassified
    # document reported success and then failed its own bundled
    # certification, which is the failure MANIFEST_DESCRIBES_ARTIFACT names.
    # The classification is the precondition the recipient enforces, not a
    # decision about what travels.
    #
    # The citation-schema policy gate is the caller's: build_package.main()
    # resolves the mode (citation_profile.resolve_citation_mode) before it
    # gets here and hands the resolved literal to this function and to the
    # dump alike.
    legal_profile.require_classified(env)
    content_predicate = legal_profile.FULL_CONTENT_SQL if public else "TRUE"
    probe_doc = blob_probe_doc(profile)
    (
        model, dims_str, runs_count_str, blob_len_str, blob_sha, fulltext_hits_str,
    ) = scalar_row(
        env,
        _MANIFEST_SCALARS_SQL.format(content=content_predicate),
        variables={"doc": probe_doc, "q": FULLTEXT_PROBE_QUERY},
        expected_columns=6,
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
    overlap = stemmed_token_overlap(
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
        Key.CITATION: citation_block(citation_mode, policy_source),
        Key.CREATED_AT: datetime.now(timezone.utc).isoformat(),
        # Declared here and FILLED by build_package.main() from the rows the
        # dump actually wrote (manifest_rows.py), like the citation block's
        # totals beside them: this runs before the dump exists, and a live
        # count describes a package nobody has produced yet.
        Key.DOCUMENTS_COUNT: 0,
        Key.PAGES_COUNT: 0,
        # ... and, beside those two headline numbers, the per-table
        # declaration for the whole schema, filled from the same writing
        # (manifest_rows.py). The citation block carries its twin, and one
        # check reads both (table_rows_check.py): a table described by
        # nothing is one whose absence certifies green.
        Key.CORPUS: {Key.TABLE_ROWS: {}},
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
