-- Citation-graph schema: one-time journal backfill.
--
-- Applied last, after pg_schema_citation.sql has created
-- citation.crawl_step and its node_key/score/tau/relation/cited_by_count
-- columns: these statements parse those columns back out of `reason` for
-- rows written before the columns existed (kb/CLAUDE.md
-- JOURNAL_FACTS_ARE_COLUMNS).

-- One-time backfill of the three columns for journal rows written before
-- they existed, parsed out of the prose that carried them. Idempotent in
-- both directions: it only ever fills a NULL, and a row whose reason never
-- carried the value keeps its NULL (a twin promotion has no score, a hub
-- skip no node). Kept in the schema (applied on every `pg_graph.py init`)
-- rather than run by hand once, because every instance this is applied to
-- -- this one, a restored artifact, a developer's fresh database -- meets
-- the same old rows.
--
-- The guard pairs each NULL column with the marker its OWN parse needs, and
-- that pairing is the difference between value-idempotent and
-- work-idempotent. "any of the three is NULL AND the reason mentions any of
-- the four markers" kept matching rows the parse can never fill any
-- further: a legacy `drop` row carries score= and tau= but no node=, so its
-- node_key stays NULL and the row matched again on every apply -- and the
-- whole schema file is applied on every `pg_graph.py init` and every
-- non-dry-run crawl. On a depth-2-sized journal (~100k rows, overwhelmingly
-- `drop`) that rewrote most of the table every time: a new tuple version,
-- WAL, and index maintenance on the three key indexes, for zero changed
-- values, on a table that is otherwise append-only.
UPDATE citation.crawl_step SET
    node_key = coalesce(node_key,
                        nullif(substring(reason from 'node=([^ ]+)'), ''),
                        nullif(substring(reason from 'seed=([^ ]+)'), '')),
    score = coalesce(score, substring(reason from 'score=(-?[0-9.]+)')::double precision),
    tau = coalesce(tau, substring(reason from 'tau=(-?[0-9.]+)')::double precision)
WHERE reason IS NOT NULL
  AND ((node_key IS NULL AND reason ~ '(node=|seed=)')
    OR (score IS NULL AND reason ~ 'score=')
    OR (tau IS NULL AND reason ~ 'tau='));

-- The same one-time parse for relation and cited_by_count, guarded the same
-- way: each NULL column paired with the marker its own parse needs, so a
-- row the parse cannot fill any further stops matching. A hub-skip row
-- carries no relation and a keep/drop row no citer count -- both keep their
-- NULL, and neither is rewritten again.
UPDATE citation.crawl_step SET
    relation = coalesce(relation, nullif(substring(reason from 'relation=([a-z]+)'), '')),
    cited_by_count = coalesce(cited_by_count,
                              substring(reason from 'cited_by_count=([0-9]+)')::bigint)
WHERE reason IS NOT NULL
  AND ((relation IS NULL AND reason ~ 'relation=')
    OR (cited_by_count IS NULL AND reason ~ 'cited_by_count='));
