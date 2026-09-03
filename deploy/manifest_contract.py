"""Shared build/verify contract for the deploy artifact: the vocabularies
a build decides by, and the two questions asked of them.

manifest.json's own shape -- its key names and MANIFEST_SCHEMA_VERSION --
is next door in manifest_keys.py (kb/CLAUDE.md FILE_SIZE, and a different
responsibility: what the FILE looks like against what a BUILD makes of the
policy). Both are read by the producer (manifest_probe.py, build-time only,
deliberately NOT bundled into the artifact -- see
artifact_bundle.DEPLOY_FILES) and by its verifiers (smoke_checks.py,
profile_checks.py, drift_probe.py, all bundled). Previously smoke_checks.py
defined MANIFEST_SCHEMA_VERSION and manifest_probe.py imported it from
there -- the producer depending on its own verifier, and dragging in
pg_rank_probe/pg_search/blob_integrity_checks/ollama_registry just to read
one int. The probe's query had the same problem and the same answer, in
probe_query.py.

schemas_for() joined them for the same reason: "which schemas does this
profile ship" was answered independently by the dumper (a tuple in
artifact_bundle.py, and another in public_dump.py) and by the manifest (a
dict plus a function in manifest_probe.py), with different conditionality
on the citation mode. They agreed only because both happened to key off
the same external fact by different routes, and pg_dump tolerates a
--schema pattern that matches nothing -- so a divergence would have
produced a manifest describing a schema set its own dump lacks, which is
what MANIFEST_DESCRIBES_ARTIFACT exists to forbid.

Beyond schemas_for(), base_schemas_for(), required_schemas(),
ships_citation() and strips_content() -- each of
them a question two sides of the build would otherwise answer independently
-- this module holds no logic, and imports only citation_vocab and
manifest_keys, neither of which imports anything at all:
the citation mode is a DB column's closed vocabulary, and
VOCABULARY_ONE_DECLARATION puts every such vocabulary in that one root
module. Importers still pull nothing along.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from citation_vocab import PublicPolicyMode  # noqa: E402
from manifest_keys import Key  # noqa: E402


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


def ships_citation(mode: str | None) -> bool:
    """Whether an artifact built under `mode` carries the citation schema at
    all -- the one authority for that question, read by the bytes
    (citation_dump.dump_citation), by their description (schemas_for below,
    manifest_probe._citation_block) and by both verifiers
    (citation_content_checks, deploy/smoke_checks).

    Asked as "is this mode declared shipping", never as "is it NONE".
    CitationMode.ALL is INHERITED from the column's own vocabulary and grows
    with it; SHIPPED is hand-written on purpose, so a mode nobody here has
    heard of ships nothing and declares nothing. Spelled `!= NONE` at four
    of the six sites, the halves disagreed the moment a fifth mode existed:
    the dump wrote no citation byte while the manifest stamped the live
    work/cites counts into it -- MANIFEST_DESCRIBES_ARTIFACT broken by the
    packager silently, and certification failed at the recipient on a build
    reported as successful.
    """
    return mode in CitationMode.SHIPPED


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


def base_schemas_for(profile: str) -> tuple[str, ...]:
    """The profile's schemas BEFORE the citation mode is applied.

    Exported because one caller genuinely wants this and not the whole
    list: public_dump._dump_ddl() asks pg_dump for the schemas whose DDL it
    writes itself, and the citation schema's DDL is written separately by
    citation_dump.dump_ddl() under the mode. Asked for by naming a mode
    that happens to ship nothing, the dependency was invisible from this
    side -- nothing here recorded that a caller relies on that mode never
    adding `citation`, so a change to SHIPPED or to the map below would
    have put the citation DDL into the file twice, and a dump with
    duplicated CREATE statements aborts at the recipient's restore.

    Same closed-vocabulary refusal schemas_for() makes, and for the same
    reason: an unknown profile decided by omission is a schema set nobody
    chose.
    """
    if profile not in Profile.ALL:
        raise ValueError(f"unknown profile {profile!r} -- expected one of {Profile.ALL}")
    return _PROFILE_BASE_SCHEMAS[profile]


def schemas_for(profile: str, citation_mode: str) -> list[str]:
    """The schemas an artifact of this profile carries under this citation
    mode -- read by the dumpers (artifact_bundle.dump_schemas,
    public_dump._dump_ddl + citation_dump) and declared verbatim in
    manifest.json (manifest_probe.gather_manifest). One list, so the
    package and its manifest cannot describe different schema sets.

    The base half is base_schemas_for()'s answer, so the relationship
    "everything the profile carries anyway, plus citation when the mode
    ships it" is written once and holds for both callers.

    Refuses an unknown profile or mode rather than defaulting: both
    vocabularies are closed (Profile.ALL, CitationMode.ALL), and guessing
    here would decide by omission what the owner decides by data.
    """
    schemas = list(base_schemas_for(profile))
    if citation_mode not in CitationMode.ALL:
        raise ValueError(
            f"unknown citation mode {citation_mode!r} -- expected one of {CitationMode.ALL}")
    if ships_citation(citation_mode):
        schemas.append("citation")
    return schemas


def required_schemas(manifest: dict) -> tuple[frozenset[str] | None, str]:
    """schemas_for() asked of a MANIFEST's own two fields, as a verdict
    instead of an exception -- what the artifact side needs to re-derive
    the rule rather than trust the list a builder wrote down.

    manifest.json is not signed and the verifiers travel inside the package
    (ARTIFACT_SIDE_FAILS_CLOSED), so "which schemas does this profile ship"
    has to be ANSWERED on the recipient's side, not read there. Every
    producer already goes through schemas_for(); a verifier comparing the
    dump with manifest.schemas alone compares a builder's claim with the
    same builder's bytes, and a build that got the rule wrong agrees with
    itself perfectly.

    None (with the reason) rather than a raise, because both readers of
    this answer are checks that must return a row: an unknown profile or
    mode has to arrive as a red line, not as a traceback out of a pass that
    then reports nothing at all.
    """
    citation = manifest.get(Key.CITATION)
    mode = citation.get(Key.CITATION_MODE) if isinstance(citation, dict) else None
    try:
        return frozenset(schemas_for(manifest.get(Key.PROFILE), mode)), ""
    except ValueError as exc:
        return None, str(exc)
