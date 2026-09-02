"""The two closed vocabularies of the citation schema, declared once.

citation.work.kind and citation.crawl_step.action are CHECK-constrained
columns, so every value the crawl writes is a contract with SQL written
elsewhere (pg_schema_citation.sql). Spelled as bare literals on the Python
side -- and they were, across the journal, the writer, the completeness
checks and the query modules -- the two halves can drift apart silently in
one direction and loudly in the worst possible place in the other: a new
action reaches the journal's bulk COPY, the CHECK rejects it, and the COPY
is all-or-nothing, so a whole level's audit record is lost AFTER its work
rows and edges were written.

So the vocabulary lives here, the SQL mirrors it, and a live test compares
the two in BOTH directions against pg_get_constraintdef() -- the same
holding-test discipline test_embedding_text.py applies to the works text in
its two dialects. This is the pattern deploy/manifest_contract.py already
uses for the artifact's own closed vocabularies (CitationMode,
PolicySource, Distribution); it is applied to the crawl's side here.

A ROOT module, not part of the citations/ package: DEPENDENCY_DIRECTION
(kb/CLAUDE.md) forbids the corpus-wide tools from importing the crawl, and
citation_checks.py and the graph query modules need these names as much as
the crawl does -- pg_graph_candidates.py even ships in the artifact, where
citations/ deliberately does not.
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
