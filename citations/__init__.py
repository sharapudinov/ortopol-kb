"""Snowball crawl of the citation graph.

Split by responsibility rather than by line count:

- openalex_client -- OpenAlex's own policy: url, quota, retry set, pagination
                     and batching; openalex_records -- what the JSON MEANS
                     (abstract recovery, id shortening, a cached page's index),
                     pure and testable without a network;
- http_session    -- the request layer all three clients share: headers,
                     timeout, the polite pause, the request counter and the
                     retry policy handed in. Each client keeps only what a
                     failure MEANS to it, which is the half that genuinely
                     differs;
- http_cache      -- the disk cache the session holds, as an object the
                     mode picks: read-write for a real run, read-only under
                     --dry-run, which writes nothing into the data tree;
- zbmath_client   -- the one thing zbMATH is the source of here: abstracts
                     for seeds OpenAlex has no abstract_inverted_index for
                     (survey.md verdict: 48/50 vs 30/56, and no cited-by at
                     all in its API, so it is never a source of edges);
- registry        -- node identity: a work is the union of the OpenAlex
                     records that share any id, not one record (survey.md
                     §7: before the union, Jaccard was understated threefold);
- scoring         -- the filter's arithmetic and nothing else: seed centroid,
                     cosine, the split at tau, quantiles and histogram (no
                     dependency beyond `math`);
- frontier        -- the seam that feeds it: the embedder bound to the
                     corpus model, and the stored vectors read a chunk at a
                     time;
- inputs          -- the reads that establish a run: the seed document set,
                     the run-85 matches, the vectors and abstracts already
                     stored; store -- Postgres: every write this crawl makes;
- seed_metadata   -- what the seeds know about themselves beyond OpenAlex
                     (zbMATH abstracts, Math-Net titles), assembled from the
                     clients and inputs above rather than in the CLI;
- crawl           -- the BFS itself, over the ones above.

Nothing here imports from research/: the spike's tooling was the prototype,
this package is the shipped code and must stand on its own.
"""
