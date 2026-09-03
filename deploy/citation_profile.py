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
packager deciding a question only the corpus owner can answer. An absent
citation SCHEMA fails a public build for the same reason and not for a
different one: "no graph travels" is the owner's mode 'none', recorded in a
schema that exists, and nothing else looks like it from the artifact side.
A FULL build is the opposite case throughout -- it applies no policy and
describes the database as it is, so a schemaless instance simply yields no
citation block.

Three modes (manifest_contract.CitationMode, mirrored by
citation.public_policy's own CHECK):

  full-skeleton   citation.work (every column, abstract/evidence included)
                  + cites (with evidence) + crawl_step + public_policy.
  topology-only   every column deploy/citation_columns.py classifies as
                  content is forced NULL -- work.abstract, work.evidence,
                  cites.evidence and crawl_step.reason -- and everything
                  else ships. That map, not a list here, is the authority:
                  it covers every column of every dumped table, and an
                  unclassified one fails the build instead of shipping by
                  default.
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

from citation_columns import JOURNAL_KEY_COLUMNS  # noqa: E402
from legal_profile import SHIPPED_SQL  # noqa: E402
from manifest_contract import CitationMode, PolicySource, Profile  # noqa: E402
from pg_common import scalar  # noqa: E402
from pg_graph_common import citation_schema_exists  # noqa: E402

# --- the per-document legal cut, as it applies to the citation slice ------
#
# Two independent policies meet here: citation.public_policy decides how much
# of the crawl record a public artifact carries at all (the modes above), and
# corpus.documents.public_distribution decides, per document, whether the
# artifact carries that document. The finer one wins on every row that names
# a document -- citation.work.document_id REFERENCES corpus.documents(id), so
# shipping a work whose document was cut away both aborts the restore on the
# FK and publishes the title (and, under full-skeleton, the abstract) of
# exactly the document whose regime the owner declined to establish.
#
# The predicate is SHIPPED_SQL, imported from legal_profile.py, never a list
# of ids restated here (LEGAL_IS_DATA). SHIPPED, not FULL_CONTENT: a
# metadata-only document DOES ship a row, so the work row naming it ships
# too -- bibliography is precisely what metadata-only means. Blanking
# document_id instead of dropping the row is not available:
# work_our_document_has_document_check (an our-document row must name its
# document -- pg_schema_citation_constraints.sql).

def shipped_work_sql(alias: str = "w") -> str:
    """SQL boolean: this citation.work row may leave in a public artifact."""
    return (
        f"({alias}.document_id IS NULL OR EXISTS ("
        f"SELECT 1 FROM corpus.documents d WHERE d.id = {alias}.document_id "
        f"AND {SHIPPED_SQL}))"
    )


# The names the cut removes, and the journal rows that mention them: four
# sets, each derived ONCE per statement. AS MATERIALIZED is not decoration
# -- a single-reference, side-effect-free CTE is INLINED by default since
# PostgreSQL 12 (the artifact image is 17), which would put each derivation
# back inside the correlated subquery it feeds.
#
# cut_keys is the works cut BECAUSE of those documents, i.e. exactly the
# rows shipped_work_sql() rejects: it rejects a row iff document_id is
# non-NULL and names an unshipped document, and citation.work.document_id
# REFERENCES corpus.documents(id), so "names an unshipped document" is the
# join below. One scan of corpus.documents for the whole statement.
#
# WHICH columns of a journal row name something is deploy/citation_columns.
# JOURNAL_KEY_COLUMNS -- one declaration, read here to BUILD the branches and
# by the bundled checker to collect the same columns off the dumped rows. A
# key column added to the schema therefore reaches the cut and its
# verification at once; spelled here as three literal joins it reached
# whichever of the two somebody remembered.
#
# The mention tests are a UNION of one branch PER KEY COLUMN rather than one
# n-way OR, and that is the whole point: an OR of three equalities is one
# non-sargable join qualifier, so none of them can reach an index. Split,
# each branch is a join the planner drives from the tiny cut_names side --
# EXPLAIN (ANALYZE) of the real COPY select on the live instance, over a
# 100k-row depth-2-sized journal inserted inside a rolled-back transaction:
# a nested loop over crawl_step_frontier_key_idx, another over
# crawl_step_candidate_key_idx and a third over crawl_step_node_key_idx, 10
# cut names and 10 loops each, 21 ms for the whole statement. At today's 604
# rows the planner hashes the table instead, as it should at that size.
#
# The node_key branch used to be strpos() over `reason`, because the node key
# lived inside that prose -- a full scan no index can serve. It is a column
# now (pg_schema_citation.sql), and matching a name means matching a name,
# not searching for its text inside a sentence: an over-matching substring
# silently dropped journal rows that named nothing sensitive.
_CUT_STEP_BRANCH = ("    SELECT j.id FROM citation.crawl_step j "
                    "JOIN cut_names r ON j.{column} = r.ref")

_CUT_CTES = """WITH cut_documents AS MATERIALIZED (
    SELECT d.id AS ref FROM corpus.documents d WHERE NOT ({shipped})
), cut_keys AS MATERIALIZED (
    SELECT w.key AS ref
    FROM citation.work w JOIN cut_documents ON cut_documents.ref = w.document_id
), cut_names AS MATERIALIZED (
    SELECT ref FROM cut_documents UNION SELECT ref FROM cut_keys
), cut_steps AS MATERIALIZED (
{steps}
)
"""


def _cut_steps() -> str:
    """One join per key column, unioned -- built per call rather than at
    import, so the branches are the declaration's answer at the moment the
    statement is written and not at the moment this module was loaded."""
    return "\n    UNION\n".join(
        _CUT_STEP_BRANCH.format(column=column) for column in JOURNAL_KEY_COLUMNS)


def crawl_step_cut_ctes() -> str:
    """The WITH clause shipped_crawl_step_sql()'s predicate reads.

    Returned separately because a COPY (SELECT ...) needs it in front of
    the SELECT, not inside the WHERE; the two belong to one statement and
    neither is valid without the other (citation_dump.copy_select pairs
    them, and its tests check the pairing).
    """
    return _CUT_CTES.format(shipped=SHIPPED_SQL, steps=_cut_steps())


def shipped_crawl_step_sql(alias: str = "s") -> str:
    """SQL boolean: this journal row names nothing the cut removed.

    Valid ONLY inside a statement prefixed with crawl_step_cut_ctes() --
    the predicate is non-membership in cut_steps, which the CTEs above
    derive once, not a re-derivation of the cut per row.
    """
    return f"(NOT EXISTS (SELECT 1 FROM cut_steps x WHERE x.id = {alias}.id))"

_POLICY_SQL = "SELECT mode FROM citation.public_policy WHERE id = 1;"


class CitationUnclassified(RuntimeError):
    """The citation schema's public-artifact policy is not decided (or not
    recognised) -- the packager must not guess (see module docstring).
    """


def citation_public_policy(env: dict) -> str | None:
    """The mode currently recorded, or None when citation.public_policy has
    no row. Callers that must not proceed on None use require_citation_mode.
    """
    return scalar(env, _POLICY_SQL) or None


def require_citation_schema(env: dict) -> None:
    """Refuses a public build against a database with no citation schema.

    Separate from the policy read below because it is asked on one more
    path: --policy-override skips the owner's row, and an override names a
    mode for a schema that exists -- it cannot conjure an absent one.
    """
    if not citation_schema_exists(env):
        raise CitationUnclassified(
            "citation schema not found in the database -- the citation graph is part of "
            "the knowledge base, not an optional extra (see kb/CLAUDE.md); apply "
            "pg_schema_citation.sql (python3 pg_graph.py init) before building a public "
            f"artifact: an artifact carrying no graph is mode {CitationMode.NONE!r} "
            "recorded by the owner, never an omission the packager arrived at"
        )


def require_citation_mode(env: dict) -> str:
    require_citation_schema(env)
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


def resolve_citation_mode(
    env: dict, profile: str, override: str | None = None
) -> tuple[str, str]:
    """The ONE reading of the citation policy per build: (mode, source).

    Resolved once by build_package.main() and threaded to every consumer --
    the manifest (manifest_probe.gather_manifest) and the dump
    (public_dump.dump_public). Neither re-derives it: two independent
    resolutions of the same policy can disagree, and the artifact would then
    describe a citation block its dump does not match, which is exactly what
    MANIFEST_DESCRIBES_ARTIFACT and profile_checks.py exist to prevent.

    The PROVENANCE travels with the mode for that same reason, and it is the
    stronger case of it: manifest.citation.policy_source is the one field
    designed to be non-fabricable (PolicySource's own docstring), so it must
    be an output of the branch that actually decided, never a second
    derivation from the same arguments. The two disagree exactly where it
    matters -- a public build against a database with no citation schema
    decides nothing, reads no owner row and honours no override, and a
    provenance computed from the profile alone would still call that the
    owner's decision.

    So: NOT_APPLICABLE only on a profile that applies no policy -- full
    carries whatever the schema holds, and holds nothing when there is no
    schema (it describes the database as it is; see build_package.py's own
    docstring), OVERRIDE only where the override picked the mode (TEST
    ONLY, see build_package.py --help), OWNER only where
    citation.public_policy was actually read.

    A PUBLIC build against a database with no citation schema is the one
    case that answers neither way: it raises. Returning NONE there would
    have the packager decide by omission that the crawl record does not
    travel -- the decision CITATION_POLICY_IS_DATA reserves for the owner,
    who records it as mode 'none' in a schema that exists. An override
    cannot conjure a schema that is absent, so it does not lift the refusal
    either; and a public build with a schema but neither row nor override
    raises for the neighbouring reason.
    """
    if profile != Profile.PUBLIC:
        if not citation_schema_exists(env):
            return CitationMode.NONE, PolicySource.NOT_APPLICABLE
        return CitationMode.FULL_SKELETON, PolicySource.NOT_APPLICABLE
    if override:
        require_citation_schema(env)
        return override, PolicySource.OVERRIDE
    return require_citation_mode(env), PolicySource.OWNER
