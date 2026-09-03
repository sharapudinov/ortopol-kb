-- Citation-graph schema: one-time journal backfill.
--
-- Applied last, after pg_schema_citation.sql has created
-- citation.crawl_step and its node_key/score/tau/relation/cited_by_count
-- columns: these statements parse those columns back out of `reason` for
-- rows written before the columns existed (kb/CLAUDE.md
-- JOURNAL_FACTS_ARE_COLUMNS).

-- Which one-time parses this database has already run. A backfill is
-- one-time by nature and this table is how the schema knows it: without a
-- record, "already done" has to be re-derived from the data on every
-- apply, and re-deriving it means scanning the very table the parse was
-- about.
CREATE TABLE IF NOT EXISTS citation.schema_backfill (
    name       TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One-time backfill of the five columns for journal rows written before
-- they existed, parsed out of the prose that carried them. Kept in the
-- schema (applied on every `pg_graph.py init`) rather than run by hand
-- once, because every instance this is applied to -- this one, a restored
-- artifact, a developer's fresh database -- meets the same old rows.
--
-- Guarded by the registry above, and that guard is the difference between
-- value-idempotent and work-idempotent. The per-column pairing inside the
-- WHERE clauses already made the parse fill each row at most once; what it
-- could not do is stop LOOKING. Those qualifiers are regex predicates no
-- index can serve, so every apply was a sequential scan of
-- citation.crawl_step evaluating five regexes per row -- on every
-- `pg_graph.py init` AND every non-dry-run crawl, against an append-only
-- table that grows by ~100k rows per depth-2 crawl, for a set of fillable
-- rows that is a fixed legacy prefix and can only shrink. The rows the
-- parse can ever change were all written before the columns existed;
-- journal.py has not put score=/node=/relation= into `reason` since. So
-- the second apply has nothing to find, and now it does not go looking:
-- one index probe instead of a scan.
--
-- The UPDATEs are unchanged inside the block, and each still fills only a
-- NULL and only where its own marker is present -- the registry decides
-- WHETHER to run the parse, not what it means. A row whose reason never
-- carried a value keeps its NULL (a twin promotion has no score, a hub
-- skip no node).
DO $backfill$
BEGIN
    IF EXISTS (SELECT 1 FROM citation.schema_backfill
               WHERE name = 'crawl_step_reason_parse') THEN
        RETURN;
    END IF;

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

    -- The same one-time parse for relation and cited_by_count. A hub-skip
    -- row carries no relation and a keep/drop row no citer count -- both
    -- keep their NULL.
    UPDATE citation.crawl_step SET
        relation = coalesce(relation, nullif(substring(reason from 'relation=([a-z]+)'), '')),
        cited_by_count = coalesce(cited_by_count,
                                  substring(reason from 'cited_by_count=([0-9]+)')::bigint)
    WHERE reason IS NOT NULL
      AND ((relation IS NULL AND reason ~ 'relation=')
        OR (cited_by_count IS NULL AND reason ~ 'cited_by_count='));

    INSERT INTO citation.schema_backfill (name) VALUES ('crawl_step_reason_parse');
END
$backfill$;

-- One-time rewrite of the `reason` prose on drop rows that were never
-- measured at all. A candidate with no title has nothing to embed, so the
-- crawl scores it NO_TEXT_SCORE (citations/scoring.py) and never compares it
-- with tau -- but the journal called every drop "below-threshold", i.e.
-- reported a relevance verdict on a candidate no relevance was computed
-- for. citations/journal.drop_reason() tells the two apart now; these are
-- the rows written before it did.
--
-- The score COLUMN is what decides, never the prose: `score <= -1` is the
-- same test the Python side makes, and `reason = 'below-threshold'` only
-- keeps the rewrite off a row somebody has already worded differently. No
-- substring, no regex -- the fact was a column all along
-- (kb/CLAUDE.md JOURNAL_FACTS_ARE_COLUMNS).
--
-- Guarded by the same registry as the parse above, and for the same reason:
-- the rows it can change are a fixed prefix that only shrinks, so the second
-- apply has nothing to find and does not go looking.
DO $no_text_reason$
BEGIN
    IF EXISTS (SELECT 1 FROM citation.schema_backfill
               WHERE name = 'crawl_step_no_text_reason') THEN
        RETURN;
    END IF;

    UPDATE citation.crawl_step SET reason = 'no text to embed'
    WHERE action = 'drop' AND score <= -1 AND reason = 'below-threshold';

    INSERT INTO citation.schema_backfill (name) VALUES ('crawl_step_no_text_reason');
END
$no_text_reason$;
