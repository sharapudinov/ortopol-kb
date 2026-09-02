"""Shared build/verify contract for the deploy artifact.

MANIFEST_SCHEMA_VERSION and the key names are read by both the producer of
manifest.json (manifest_probe.py, build-time only, deliberately NOT bundled
into the artifact -- see artifact_bundle.DEPLOY_FILES) and its verifiers
(smoke_checks.py, profile_checks.py, drift_probe.py, all bundled).
Previously smoke_checks.py defined MANIFEST_SCHEMA_VERSION and
manifest_probe.py imported it from there -- the producer depending on its
own verifier, and dragging in pg_rank_probe/pg_search/
blob_integrity_checks/ollama_registry just to read one int. The probe's
query had the same problem and the same answer, in probe_query.py.

schemas_for() joined them for the same reason: "which schemas does this
profile ship" was answered independently by the dumper (a tuple in
artifact_bundle.py, and another in public_dump.py) and by the manifest (a
dict plus a function in manifest_probe.py), with different conditionality
on the citation mode. They agreed only because both happened to key off
the same external fact by different routes, and pg_dump tolerates a
--schema pattern that matches nothing -- so a divergence would have
produced a manifest describing a schema set its own dump lacks, which is
what MANIFEST_DESCRIBES_ARTIFACT exists to forbid.

Beyond schemas_for() and strips_content() -- each of them a question two
sides of the build would otherwise answer independently -- this module holds
no logic, and imports only citation_vocab (which imports nothing at all):
the citation mode is a DB column's closed vocabulary, and
VOCABULARY_ONE_DECLARATION puts every such vocabulary in that one root
module. Importers still pull nothing along.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from citation_vocab import PublicPolicyMode  # noqa: E402


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
    SHIPPED_DISTRIBUTIONS = "shipped_distributions"

    # citation{} -- describes the PACKAGE, produced by
    # manifest_probe.gather_manifest() from citation_profile.py's reading of
    # citation.public_policy, verified statically by
    # citation_content_checks.py. Counts are about what THIS artifact
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


class PolicySource:
    """Whose decision manifest.citation.mode records.

    OWNER: read from citation.public_policy, i.e. the corpus owner's row --
    the only provenance a publishable artifact may carry
    (PUBLIC_APPROVED_BY_OWNER, CITATION_POLICY_IS_DATA).
    OVERRIDE: forced at the command line by --policy-override, which exists
    so the packaging and smoke pipeline can be exercised before that
    decision is made. Never publishable, and profile_checks.py fails on it.
    NOT_APPLICABLE: this profile applies no citation policy at all. Only
    the public profile has one to apply -- full carries the whole schema
    whatever the owner's row says, and resolve_citation_mode() does not
    even read citation.public_policy for it. Naming the owner there would
    put a decision nobody made into the one field designed to be
    non-fabricable.

    In the manifest rather than only in the filename because the filename
    is not part of the package: it is renamed by a copy, and it is not what
    a recipient (or a later session) reads to learn what they are holding.
    The name still differs -- see build_package.py -- but the refusal rests
    on this field.
    """

    OWNER = "owner"
    OVERRIDE = "override"
    NOT_APPLICABLE = "not-applicable"
    ALL = (OWNER, OVERRIDE, NOT_APPLICABLE)


class Distribution:
    """corpus.documents.public_distribution's vocabulary -- the DATA that
    decides what the public profile ships. Its meaning (and the refusal to
    guess about anything outside ALL) lives in legal_profile.py.
    """

    FULL_TEXT = "full-text"
    METADATA_ONLY = "metadata-only"
    INTERNAL = "internal"
    EXCLUDED = "excluded"
    ALL = (FULL_TEXT, METADATA_ONLY, INTERNAL, EXCLUDED)
    # Distributions whose documents ship with every byte in the public
    # profile: someone else's copyright is either not involved (internal) or
    # the licence/legal basis permits redistribution (full-text).
    FULL_CONTENT = (FULL_TEXT, INTERNAL)
    # Distributions whose documents appear in the public artifact AT ALL.
    # Everything outside this tuple is written nowhere: not as a documents
    # row, not as a page row, not as a vector. metadata-only says "the work
    # exists, here is where to buy/read it"; excluded says the owner has not
    # established a right to say even that much from this package, so the
    # packager must not turn the absence of a decision into a publication.
    SHIPPED = (FULL_TEXT, METADATA_ONLY, INTERNAL)


class CitationMode(PublicPolicyMode):
    """The packager's view of citation.public_policy.mode. WHICH modes exist
    is the column's own vocabulary (citation_vocab.PublicPolicyMode, which
    the SQL CHECK mirrors and a live test holds it to); this class adds only
    what a BUILD makes of each. Restated here it would have been a third
    spelling of a closed DB vocabulary, outside the one mechanism that
    compares such a spelling with the database's.

    The refusal to guess about anything outside ALL lives in
    deploy/citation_profile.py; citation_dump.py applies the mode to the
    dump, citation_content_checks.py verifies it against the dump's bytes.
    """

    # Modes whose dump carries the citation schema at all -- read by
    # schemas_for() below (does the manifest declare it) and by
    # citation_dump.dump_citation() (does the dump write it), which is the
    # whole of the question in both places. Hand-written on purpose: a mode
    # added to the DB vocabulary and not to this tuple ships NOTHING and
    # declares nothing, which is the safe half of being unheard-of.
    SHIPPED = (PublicPolicyMode.FULL_SKELETON, PublicPolicyMode.TOPOLOGY_ONLY)
    # Modes whose citation.work/cites rows carry abstract/evidence. The
    # dump and the artifact-side hunt both read it through strips_content()
    # below rather than testing a mode by name.
    FULL_CONTENT = (PublicPolicyMode.FULL_SKELETON,)


def strips_content(mode: str | None) -> bool:
    """Whether a dump built under `mode` must blank the content columns.

    Asked as "is this mode declared full-content", never as "is it
    topology-only". A mode added to CitationMode and forgotten at one of
    the two call sites -- citation_dump._select_expression (what the COPY
    projects) and citation_content_checks.attach_visitors (whether the
    artifact-side hunt runs at all) -- would otherwise ship abstracts AND
    be exempt from the check that would catch it. Anti-default per mode,
    as citation_columns.blanked_cast() is per column.
    """
    return mode not in CitationMode.FULL_CONTENT


# Schemas each profile's dump carries BEFORE the citation mode is applied.
# The public profile leaves measurements behind entirely (it is the record
# of our own research, not of the corpus); the full profile is the owner's
# own backup and carries both.
_PROFILE_BASE_SCHEMAS = {
    Profile.FULL: ("corpus", "measurements"),
    Profile.PUBLIC: ("corpus",),
}


def schemas_for(profile: str, citation_mode: str) -> list[str]:
    """The schemas an artifact of this profile carries under this citation
    mode -- read by the dumpers (artifact_bundle.dump_schemas,
    public_dump._dump_ddl + citation_dump) and declared verbatim in
    manifest.json (manifest_probe.gather_manifest). One list, so the
    package and its manifest cannot describe different schema sets.

    Refuses an unknown profile or mode rather than defaulting: both
    vocabularies are closed (Profile.ALL, CitationMode.ALL), and guessing
    here would decide by omission what the owner decides by data.
    """
    if profile not in Profile.ALL:
        raise ValueError(f"unknown profile {profile!r} -- expected one of {Profile.ALL}")
    if citation_mode not in CitationMode.ALL:
        raise ValueError(
            f"unknown citation mode {citation_mode!r} -- expected one of {CitationMode.ALL}")
    schemas = list(_PROFILE_BASE_SCHEMAS[profile])
    if citation_mode in CitationMode.SHIPPED:
        schemas.append("citation")
    return schemas

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
MANIFEST_SCHEMA_VERSION = 8
