"""Shared build/verify contract for the deploy artifact.

MANIFEST_SCHEMA_VERSION and VECTOR_PROBE_QUERY are read by both the
producer of manifest.json (manifest_probe.py, build-time only, deliberately
NOT bundled into the artifact -- see artifact_bundle.DEPLOY_FILES) and its
two verifiers (smoke_checks.py, drift_probe.py, both bundled). Previously
smoke_checks.py defined MANIFEST_SCHEMA_VERSION and manifest_probe.py
imported it from there -- the producer depending on its own verifier, and
dragging in pg_rank_probe/pg_search/blob_integrity_checks/ollama_registry
just to read one int. VECTOR_PROBE_QUERY was independently duplicated
verbatim as drift_probe.DEFAULT_QUERY, kept in sync only by a comment.

This module holds no logic and imports nothing else, so every one of its
three importers can depend on it without pulling anything else along.
"""
from __future__ import annotations


class Key:
    """manifest.json's key names, named once for its producer
    (manifest_probe.gather_manifest, build_package.py) and every consumer
    (smoke_checks.py, vector_probe_check.py, blob_integrity_checks.py,
    bundled_files_check.py, smoke_test.py, drift_probe.py) to share,
    instead of each re-typing the same string literal independently.
    Adding, renaming or nesting a field is then a single-file edit here
    that fails loudly (NameError/AttributeError) at the first stale call
    site, rather than a KeyError discovered mid-smoke-run against whichever
    consumer someone forgot to grep for.

    Plain string constants, no logic -- consistent with the rest of this
    module (see its own docstring).
    """

    # Top level.
    SCHEMA_VERSION = "schema_version"
    PROFILE = "profile"
    SCHEMAS = "schemas"
    LEGAL = "legal"
    CREATED_AT = "created_at"
    DOCUMENTS_COUNT = "documents_count"
    PAGES_COUNT = "pages_count"
    EMBEDDING_MODEL = "embedding_model"
    MEASUREMENTS_RUN_COUNT = "measurements_run_count"
    BLOB_PROBE = "blob_probe"
    FULLTEXT_PROBE = "fulltext_probe"
    VECTOR_PROBE = "vector_probe"
    FILES = "files"
    DUMP = "dump"

    # embedding_model{}.
    MODEL = "model"
    DIMS = "dims"
    DIGEST = "digest"
    SIZE_BYTES = "size_bytes"

    # blob_probe{}; document_id/sha256 also occur in vector_probe{}/dump{}
    # respectively, with the same meaning (an id, a hex digest) -- one
    # constant covers both rather than a redundant near-duplicate name.
    DOCUMENT_ID = "document_id"
    BYTE_LENGTH = "byte_length"
    SHA256 = "sha256"

    # fulltext_probe{}; query also occurs in vector_probe{} with the same
    # meaning (the probe's search string).
    QUERY = "query"
    HITS = "hits"

    # vector_probe{} (beyond QUERY/DOCUMENT_ID above).
    PAGE_NUMBER = "page_number"
    RANK = "rank"
    DISTANCE = "distance"
    RUNNER_UP_DISTANCE = "runner_up_distance"
    TOKEN_OVERLAP = "token_overlap"

    # dump{} (beyond SHA256 above).
    FILE = "file"
    BYTES = "bytes"

    # legal{} -- the classification this build applied, produced by
    # legal_profile.legal_summary() and verified statically by
    # profile_checks.py.
    VERIFY_QUERY = "verify_query"
    UNCLASSIFIED_DOCUMENTS = "unclassified_documents"
    CLASS_COUNTS = "class_counts"
    DOCUMENTS_BY_DISTRIBUTION = "documents_by_distribution"
    FULL_CONTENT_DISTRIBUTIONS = "full_content_distributions"


class Profile:
    """The two artifact profiles, named here rather than in legal_profile.py
    so that profile_checks.py -- which must stay free of any Postgres import
    to remain a purely static check -- can share the vocabulary with the
    builder instead of restating the strings.

    FULL is never published (see README and build_package.py's docstring);
    PUBLIC applies the legal filter described in public_dump.py.
    """

    FULL = "full"
    PUBLIC = "public"
    ALL = (FULL, PUBLIC)


class Distribution:
    """corpus.documents.public_distribution's vocabulary -- the DATA that
    decides what the public profile ships. Its meaning (and the refusal to
    guess about anything outside ALL) lives in legal_profile.py.
    """

    FULL_TEXT = "full-text"
    METADATA_ONLY = "metadata-only"
    INTERNAL = "internal"
    ALL = (FULL_TEXT, METADATA_ONLY, INTERNAL)
    # Distributions whose documents ship with every byte in the public
    # profile: someone else's copyright is either not involved (internal) or
    # the licence/legal basis permits redistribution (full-text).
    FULL_CONTENT = (FULL_TEXT, INTERNAL)


# 4: added profile/schemas/legal to the manifest (task 033's two-profile
# packager). A profile-unaware artifact cannot be verified by
# profile_checks.py at all, so an older manifest must fail the version gate
# rather than be read with defaults.
MANIFEST_SCHEMA_VERSION = 4

# A paraphrase of "an algebraic polynomial bounded from its values on a
# uniform grid" (the recurring theme of 1997_sm280 and related papers) with
# genuinely ZERO shared stemmed lexeme against its own nearest page's
# wording -- not merely zero shared surface forms, and not excusing the
# domain noun either: earlier wordings that kept "полином"/"величина"/
# "оценить" etc. kept landing on pages that use the exact same word, which
# the stemmed check (manifest_probe._stemmed_token_overlap) correctly
# rejected. This wording was accepted only after gather_manifest ran clean
# against the live corpus (verified: phraseto_tsquery also finds "no
# matches" for it). Nearest page can legitimately drift as the corpus/model
# change -- gather_manifest() re-checks the invariant against whatever page
# is actually nearest on every build and refuses to record a pair that
# overlaps, rather than trusting this comment to still hold.
VECTOR_PROBE_QUERY = (
    "какое предельное значение по абсолютной величине допускает "
    "рациональная форма, заданная своими данными на равноотстоящих "
    "узлах отрезка"
)
