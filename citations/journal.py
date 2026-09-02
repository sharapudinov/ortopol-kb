#!/usr/bin/env python3
"""What the crawl writes into citation.crawl_step, in one place.

The journal is the only record of decisions that leave no row anywhere else:
a dropped candidate has no work row, a hub has no citer edges, a corpus
document absent from OpenAlex has nothing at all. "Why is X in the graph" and
"why isn't Y" are answerable only from here.

Every reason that carries numbers carries them in a FIXED, parseable form --
`score=0.6123 tau=0.5000 relation=cites` -- so the score distribution at
depths the tau calibration never saw is a query rather than a re-crawl:

    SELECT depth, action,
           substring(reason from 'score=(-?[0-9.]+)')::float8 AS score
    FROM citation.crawl_step WHERE crawl_id = ...

Centralised because that format is a contract with SQL written elsewhere:
a reason string built inline at six call sites drifts at the first edit.
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


def keep(crawl_id, depth, candidate_key, node_key, score, tau, relation,
         frontier_key=None) -> dict:
    return _step(crawl_id, depth, "keep", frontier_key=frontier_key or None,
                 candidate_key=candidate_key,
                 reason=f"kept; score={score:.4f} tau={tau:.4f} "
                        f"relation={relation} node={node_key}")


def drop(crawl_id, depth, candidate_key, score, tau, relation,
         frontier_key=None) -> dict:
    return _step(crawl_id, depth, "drop", frontier_key=frontier_key or None,
                 candidate_key=candidate_key,
                 reason=f"below-threshold; score={score:.4f} tau={tau:.4f} "
                        f"relation={relation}")


def fetch(crawl_id, depth, frontier_key, n_found, n_kept) -> dict:
    return _step(crawl_id, depth, "fetch", frontier_key=frontier_key,
                 n_found=n_found, n_kept=n_kept)


def hub_skip(crawl_id, depth, frontier_key, cited_by_count, cap) -> dict:
    """The node was not asked upward -- a decision, not a failure."""
    return _step(crawl_id, depth, "hub-skip", frontier_key=frontier_key,
                 reason=f"cited_by_count={cited_by_count} > cap {cap}")


def twin(crawl_id, node_key, document_id, seed_key) -> dict:
    """An external skeleton turned out to BE one of our own works.

    Depth 0: the promotion is a statement about the corpus, not about the
    level at which the crawl happened to meet the work.
    """
    return _step(crawl_id, 0, "keep", frontier_key=document_id,
                 candidate_key=node_key,
                 reason=f"twin-of={document_id} seed={seed_key}")
