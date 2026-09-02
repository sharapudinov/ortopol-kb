"""The closed vocabularies of the citation schema, declared once.

citation.work.kind, citation.crawl_step.action, citation.crawl_step.
relation and citation.public_policy.mode are CHECK-constrained columns, so every value written to them is a
contract with SQL written elsewhere (pg_schema_citation.sql). Spelled as
bare literals on the Python side -- and they were, across the journal, the
writer, the completeness checks and the query modules -- the two halves can
drift apart silently in one direction and loudly in the worst possible
place in the other: a new action reaches the journal's bulk COPY, the CHECK
rejects it, and the COPY is all-or-nothing, so a whole level's audit record
is lost AFTER its work rows and edges were written.

So each vocabulary lives here, the SQL mirrors it, and a live test compares
the two in BOTH directions against pg_get_constraintdef() -- the same
holding-test discipline test_embedding_text.py applies to the works text in
its two dialects. This is the pattern deploy/manifest_contract.py already
uses for the artifact's own closed vocabularies (PolicySource,
Distribution); it is applied to the schema's own columns here.

A ROOT module, not part of the citations/ package: DEPENDENCY_DIRECTION
(kb/CLAUDE.md) forbids the corpus-wide tools from importing the crawl, and
citation_checks.py and the graph query modules need these names as much as
the crawl does -- pg_graph_candidates.py even ships in the artifact, where
citations/ deliberately does not. deploy/manifest_contract.CitationMode
reaches here for the third vocabulary rather than restating it: the
packager's view of a mode (which ones ship, which carry content) is the
artifact's business, but WHICH modes exist is the column's.
"""
from __future__ import annotations


class WorkKind:
    """citation.work.kind -- what a node in the graph IS to this corpus.

    OUR_DOCUMENT: already in corpus.documents (document_id set).
    EXTERNAL_SKELETON: a node we have metadata for but no full text -- most
    of the graph, by construction.
    INDEXED: an external skeleton promoted after being read in full; kept
    distinct from our-document, which means "one of the IIS works", not
    "we happened to read this one too".
    EXCLUDED: considered and rejected by the crawl filter.

    The reasoning behind each value, and the two conditional CHECKs that go
    with it (an our-document row must name its document, an excluded one
    must carry its reason), live beside the column in
    pg_schema_citation.sql.
    """

    OUR_DOCUMENT = "our-document"
    EXTERNAL_SKELETON = "external-skeleton"
    INDEXED = "indexed"
    EXCLUDED = "excluded"
    ALL = (OUR_DOCUMENT, EXTERNAL_SKELETON, INDEXED, EXCLUDED)
    # Kinds whose row is a claim about somebody else's work rather than
    # about our corpus, so `evidence` is the only thing standing between
    # the claim and an unverifiable assertion (citation_checks.py).
    NEED_EVIDENCE = (EXTERNAL_SKELETON, INDEXED)


class CrawlAction:
    """citation.crawl_step.action -- the kind of decision a journal row is.

    SEED / SEED_MISSING: a corpus document was matched to an OpenAlex work,
    or looked for and not found. FETCH: a frontier node was expanded.
    KEEP / DROP: a candidate passed or failed tau. HUB_SKIP: a node was not
    asked upward because its citer count is past the cap -- a decision, not
    a failure. ERROR: the crawl asked and did not find out.

    The vocabulary grows (hub-skip arrived when depth-2 turned out to pull
    >51k citers through a handful of heavily-cited classics), which is why
    the SQL side is a named constraint applied separately rather than an
    inline CHECK: see pg_schema_citation.sql's own note on widening it
    without a full validation scan.
    """

    SEED = "seed"
    SEED_MISSING = "seed-missing"
    FETCH = "fetch"
    KEEP = "keep"
    DROP = "drop"
    HUB_SKIP = "hub-skip"
    ERROR = "error"
    ALL = (SEED, SEED_MISSING, FETCH, KEEP, DROP, HUB_SKIP, ERROR)


class Relation:
    """citation.crawl_step.relation -- HOW a candidate reached the frontier.

    CITES: the candidate cites a frontier node (the crawl asked "who cites
    you"). REFERENCED: the frontier node cites the candidate (it came out
    of the node's own reference list).

    Not a label but a decision the crawl ACTS on: only a node reached as a
    citer expands at depth >= 2, because the down direction pulls in
    classics whose citers are about the field rather than about this corpus
    (kb/CLAUDE.md SNOWBALL_FRONTIER). The hub measurement groups its whole
    verdict by it, and citations/threshold_store.py mirrors the same pair on
    the measurements table a calibration writes.

    NULL is a legal value of the column and has no constant here: it means
    the row is not ABOUT a relation (seed, error, hub-skip), not that the
    relation is unknown.
    """

    CITES = "cites"
    REFERENCED = "referenced"
    ALL = (CITES, REFERENCED)


class PublicPolicyMode:
    """citation.public_policy.mode -- how much of the citation schema a
    PUBLIC artifact carries, decided by the corpus owner as a row and never
    by code (CITATION_POLICY_IS_DATA).

    FULL_SKELETON: every table, with abstracts and evidence.
    TOPOLOGY_ONLY: every table, with the content columns blanked -- keys,
    kinds, years, edges and the journal's machine-readable columns.
    NONE: the schema does not travel at all.

    Declared here rather than in deploy/manifest_contract.py, where the
    packager's reading of it lives, for the reason the other two are here:
    the SQL CHECK is the other half of the contract, and only a Python
    declaration in a module the live test can compare against
    pg_get_constraintdef() keeps the halves from drifting. What each mode
    MEANS to a build is deploy/citation_profile.py's docstring.
    """

    FULL_SKELETON = "full-skeleton"
    TOPOLOGY_ONLY = "topology-only"
    NONE = "none"
    ALL = (FULL_SKELETON, TOPOLOGY_ONLY, NONE)
