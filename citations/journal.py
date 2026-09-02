#!/usr/bin/env python3
"""What the crawl writes into citation.crawl_step, in one place.

The journal is the only record of decisions that leave no row anywhere else:
a dropped candidate has no work row, a hub has no citer edges, a corpus
document absent from OpenAlex has nothing at all. "Why is X in the graph" and
"why isn't Y" are answerable only from here.

Every fact the pipeline READS lands in a column -- node_key, score, tau,
relation, cited_by_count beside frontier_key, candidate_key, n_found,
n_kept -- and `reason` keeps only what a human reads. So the score
distribution at depths the tau calibration never saw is a plain query over
indexed, typed values:

    SELECT depth, action, score
    FROM citation.crawl_step WHERE crawl_id = ...

Those columns were prose once (`score=0.6123 tau=0.5000 node=W123`,
`relation=cites`, `cited_by_count=5000 > cap 1000`) and separate consumers
parsed them back out with substring() and split_part(); a number a query
needs is not prose, and a name the public artifact's cut matches on cannot
sit in the one column no index can serve. `relation` was the last one to
leave: it decides whether a node expands at all (SNOWBALL_FRONTIER), and
the hub measurement had been re-deriving it from citation.work.evidence
with an 'unknown' fallback because the journal could not answer.

Centralised because which column carries what is a contract with SQL written
elsewhere: a step dict built inline at six call sites drifts at the first
edit.
"""
from __future__ import annotations


def _step(crawl_id, depth, action, **fields) -> dict:
    step = {"crawl_id": crawl_id, "depth": depth, "action": action}
    step.update({k: v for k, v in fields.items() if v is not None})
    return step


def seed(crawl_id, document_id, key) -> dict:
    return _step(crawl_id, 0, "seed", frontier_key=document_id,
                 candidate_key=key, n_found=1, n_kept=1)


def seed_missing(crawl_id, document_id) -> dict:
    return _step(crawl_id, 0, "seed-missing", frontier_key=document_id,
                 reason="not in OpenAlex (run 85)")


def seed_error(crawl_id, document_id, openalex_id) -> dict:
    return _step(crawl_id, 0, "error", frontier_key=document_id,
                 candidate_key=openalex_id,
                 reason="матч run 85 не отдан OpenAlex по openalex_id")


def zbmath_error(crawl_id, document_id, zbmath_id, reason) -> dict:
    """The zbMATH abstract fallback did not answer for this seed.

    Not the same row as "zbMATH has no review for it", which leaves no row
    at all and simply no abstract: this one says we never found out. Without
    it a transient 429 during the seeding pass is indistinguishable, forever
    after, from a work zbMATH genuinely does not review.
    """
    return _step(crawl_id, 0, "error", frontier_key=document_id,
                 candidate_key=zbmath_id, reason=f"zbmath: {reason}")


def keep(crawl_id, depth, candidate_key, node_key, score, tau, relation,
         frontier_key=None) -> dict:
    """`node_key` is the registry node the candidate was merged into -- two
    OpenAlex records of one work share a node, so it is not always the
    candidate's own key, and it is the name the graph actually carries."""
    return _step(crawl_id, depth, "keep", frontier_key=frontier_key or None,
                 candidate_key=candidate_key, node_key=node_key,
                 score=score, tau=tau, relation=relation, reason="kept")


def drop(crawl_id, depth, candidate_key, score, tau, relation,
         frontier_key=None) -> dict:
    """No node_key: a dropped candidate becomes no node, which is exactly
    what the empty column says about it."""
    return _step(crawl_id, depth, "drop", frontier_key=frontier_key or None,
                 candidate_key=candidate_key, score=score, tau=tau,
                 relation=relation, reason="below-threshold")


def fetch(crawl_id, depth, frontier_key, n_found, n_kept) -> dict:
    return _step(crawl_id, depth, "fetch", frontier_key=frontier_key,
                 n_found=n_found, n_kept=n_kept)


def hub_skip(crawl_id, depth, frontier_key, cited_by_count, cap) -> dict:
    """The node was not asked upward -- a decision, not a failure.

    The citer count is the measured quantity the decision turned on, so it
    is a column. The cap it was compared against stays in the prose: it is
    the run's own `--hub-cap`, recorded for a human reading one row, and
    nothing queries it.
    """
    return _step(crawl_id, depth, "hub-skip", frontier_key=frontier_key,
                 node_key=frontier_key, cited_by_count=cited_by_count,
                 reason=f"цитирующих больше порога хабов ({cap})")


def twin(crawl_id, candidate_key, document_id, seed_key) -> dict:
    """An external skeleton turned out to BE one of our own works.

    Depth 0: the promotion is a statement about the corpus, not about the
    level at which the crawl happened to meet the work.

    Three names, three columns: the document it belongs to (frontier_key),
    the external record promoted (candidate_key) and the seed work it turned
    out to BE (node_key) -- that last one is what "the node this decision
    resolved to" means for a promotion.
    """
    return _step(crawl_id, 0, "keep", frontier_key=document_id,
                 candidate_key=candidate_key, node_key=seed_key,
                 reason="двойник нашей работы")
