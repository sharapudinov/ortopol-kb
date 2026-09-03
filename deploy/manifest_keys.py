"""manifest.json's own shape: the key names, and which version this is.

Read by the producer (manifest_probe.gather_manifest, build_package.py) and
by every verifier (profile_checks.py, smoke_checks.py, drift_probe.py,
blob_integrity_checks.py, vector_probe_check.py, bundled_files_check.py,
smoke_test.py) -- both sides of a file that travels, which is why neither
side may spell a key itself.

Split from manifest_contract.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): here is what the FILE looks like, there is what a BUILD makes
of the policy (which profile ships which schemas, which citation modes
carry content). The two are read together often and change apart: a new
manifest field is a key and a version bump, a new policy value is neither.

No logic and no imports, deliberately -- string constants and one int.
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
    # corpus{} -- the same per-table declaration the citation block carries
    # below, for the schema every profile ships. documents_count and
    # pages_count above are two headline numbers; everything else the
    # corpus schema holds was described by nothing, and a check that finds
    # no COPY block cannot tell "cut correctly" from "never shipped".
    CORPUS = "corpus"

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
    SHIPPED_DISTRIBUTIONS = "shipped_distributions"

    # citation{} -- describes the PACKAGE. The mode comes from
    # citation_profile.py's reading of citation.public_policy; every number
    # is stamped from the dump itself (manifest_rows.py) and verified
    # statically against its bytes by citation_content_checks.py and
    # citation_cut_checks.py. Counts are about what THIS artifact
    # carries, not the live database (MANIFEST_DESCRIBES_ARTIFACT) -- zero
    # for CitationMode.NONE, whole-corpus otherwise (full-skeleton and
    # topology-only ship every row, only some columns differ).
    # policy_source says WHOSE decision the mode is (PolicySource below):
    # the filename cannot carry that, and a recipient holding only the file
    # has no other way to ask.
    CITATION = "citation"
    CITATION_MODE = "mode"
    CITATION_POLICY_SOURCE = "policy_source"
    WORK_COUNT = "work_count"
    CITES_COUNT = "cites_count"
    WORK_BY_KIND = "work_by_kind"
    # {table: rows} for EVERY table of the block's schema the dump carries,
    # not only the ones the counts above describe. What a recipient can
    # otherwise learn about crawl_step, public_policy and schema_backfill
    # is nothing: a check that finds no such block has no way to tell "cut
    # correctly" from "never shipped", and reports a green nought either
    # way. One name for both blocks (corpus{} and citation{}), because one
    # check reads both (table_rows_check.py).
    TABLE_ROWS = "table_rows"


# 4: added profile/schemas/legal to the manifest (two-profile packager). A
# profile-unaware artifact cannot be verified by profile_checks.py at all,
# so an older manifest must fail the version gate rather than be read with
# defaults.
# 5: added legal.shipped_distributions. Without it a reader cannot tell an
# artifact that carries every classified document from one that leaves a
# whole class out on purpose -- documents_by_distribution lists the corpus,
# and only this field says which of those lists the package actually
# contains. Read with a default it would silently turn a deliberate
# exclusion into an unexplained gap, so a v4 manifest fails the gate.
# 6: added the citation{} block and, correspondingly, "citation"
# to schemas{} whenever a build ships that schema. A v5 reader has no key to
# learn the citation-graph policy from at all -- reading with a default would
# either hide a mode the artifact actually applied or invent counts the
# package does not carry, so a v5 manifest fails the gate rather than being
# read as "citation not shipped".
# 7: added citation.policy_source. A v6 reader cannot tell an artifact
# whose citation policy is the owner's row from one whose mode was forced
# by --policy-override; read with a default, an override build would be
# certified as owner-classified, which is the one thing the flag must never
# be able to produce.
# 8: citation.policy_source gained a third value, "not-applicable", and a
# full-profile manifest now carries THAT rather than "owner". A v7 full
# artifact asserts an owner decision that was never read, and a v7 reader
# would refuse the honest value as unknown, so the two must not meet: the
# version gate separates them instead of either side guessing.
# 9: added citation.table_rows. Without it the artifact-side checks assert
# the presence and the row count of exactly two citation tables and read the
# absence of every other as nothing to check -- so the journal cut, the
# schema's most intricate policy SQL, was certified by a check that passed
# vacuously on a dump carrying no journal at all. A v8 manifest names no
# tables, and read with a default that would be the same silence, so the
# version gate separates them.
# 10: added corpus.table_rows -- the same declaration v9 gave the citation
# schema, for the schema every profile ships. Without it the corpus half of
# the certification asserts the row counts of exactly two tables and reads
# the absence of every other as nothing to check, which is the polarity v9
# was added to invert. A v9 manifest names no corpus table, and read with a
# default that is the same silence, so the version gate separates them.
MANIFEST_SCHEMA_VERSION = 10
