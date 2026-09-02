"""Snowball crawl of the citation graph (plan 038, task 005).

Split by responsibility rather than by line count:

- openalex_client -- HTTP, quota, pagination, batching, abstract recovery;
- zbmath_client   -- the one thing zbMATH is the source of here: abstracts
                     for seeds OpenAlex has no abstract_inverted_index for
                     (survey.md verdict: 48/50 vs 30/56, and no cited-by at
                     all in its API, so it is never a source of edges);
- registry        -- node identity: a work is the union of the OpenAlex
                     records that share any id, not one record (survey.md
                     §7: before the union, Jaccard was understated threefold);
- frontier        -- the relevance filter: embeddings, seed centroid, cosine,
                     and the distribution helpers the tau calibration reports;
- store           -- Postgres: every write this crawl makes;
- crawl           -- the BFS itself, over the four above.

Nothing here imports from research/: the spike's tooling was the prototype,
this package is the shipped code and must stand on its own.
"""
