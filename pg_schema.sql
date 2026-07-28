-- Corpus schema.
--
-- Lives in its own namespace, not in `public`. Measurements already occupy
-- `measurements`, and finding records (issue #14) are coming; a shared `public`
-- would turn into a junk drawer the moment the third record kind arrives.
--
-- documents: one row per source PDF, always — including the six unreadable
-- ones. A missing row would let a coverage query silently report a
-- complete corpus while a broken-font source (e.g. 2017_demr34) has
-- vanished from it. extraction_state is the durable form of that guarantee.
--
-- source_tier defaults to 'local_corpus' for everything today. It exists
-- now, before it has more than one value, because retrofitting it later
-- means re-labelling the whole corpus by hand once a second tier (e.g.
-- peer-reviewed vs. vendor-claimed) actually shows up.
--
-- pages: one row per extracted page, carrying BOTH retrieval keys.
--   tsv       — 'russian' text search config (Snowball stemming + stopwords),
--               so morphological variants match. See pg_search.py for why
--               'russian' beats 'simple' here.
--   embedding — bge-m3, 1024 dims, computed by pg_embed.py against the local
--               ollama. Required, not decorative: a record without a semantic
--               key is findable only by someone who already knows the exact
--               word to search for, which is precisely the case retrieval is
--               supposed to solve.

CREATE SCHEMA IF NOT EXISTS corpus;

CREATE TABLE IF NOT EXISTS corpus.documents (
    id                TEXT PRIMARY KEY,
    filename          TEXT NOT NULL,
    extraction_state  TEXT NOT NULL CHECK (extraction_state IN ('clean', 'recoded', 'degraded', 'transcribed', 'ocr', 'metadata', 'unreadable')),
    source_tier       TEXT NOT NULL DEFAULT 'local_corpus',
    pages_count       INTEGER NOT NULL DEFAULT 0,
    chars_extracted   INTEGER NOT NULL DEFAULT 0,
    note              TEXT,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Прослеживаемость: находка ведёт к исходнику в один переход, не через
    -- конвенцию. source_path generated — все источники лежат в theory/iis/
    -- плоско (инвариант classify() в corpus_completeness.py, он это проверяет).
    source_path       TEXT GENERATED ALWAYS AS ('theory/iis/' || filename) STORED,
    source_url        TEXT,  -- Math-Net.Ru из INDEX.md (pg_source_urls.py); NULL если нет
    -- Самодостаточность: сам исходник в базе (pg_load_blobs.py). Пакет
    -- разворачивается без theory/, сверка транскрипции с изображением возможна
    -- везде. sha256 делает подмену файла на диске обнаружимой.
    source_blob       BYTEA,
    source_sha256     TEXT,
    -- Правовой режим документа — ДАННЫЕ, не знание в голове сборщика:
    -- deploy/build_package.py --profile public читает public_distribution и
    -- ничего не решает сам (см. deploy/legal_profile.py). Классификацию ставит
    -- владелец по corpus/legal-regimes-dossier.md; агенту её выдумывать нельзя.
    -- Значения public_distribution: 'full-text' (весь контент уезжает в
    -- публичный артефакт), 'metadata-only' (только строка документа и векторы
    -- страниц — ни блоба, ни текста), 'internal' (наши собственные служебные
    -- документы, INDEX/THEMES), 'excluded' (в публичный артефакт не попадает
    -- ничего: ни строки документа, ни страниц — режим не установлен, и даже
    -- библиография была бы решением открытого правового вопроса).
    -- Незнакомое значение = сборка public падает, а не «на всякий случай
    -- включаем».
    legal_class          TEXT,
    public_distribution  TEXT,
    legal_note           TEXT   -- основание одной строкой (цитата/норма/дата проверки)
);

CREATE TABLE IF NOT EXISTS corpus.pages (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    body         TEXT NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('russian', body)) STORED,
    embedding    vector(1024),
    UNIQUE (document_id, page_number)
);

CREATE INDEX IF NOT EXISTS pages_tsv_idx ON corpus.pages USING GIN (tsv);
CREATE INDEX IF NOT EXISTS pages_embedding_hnsw ON corpus.pages
    USING hnsw (embedding vector_cosine_ops);

-- Which model produced corpus.pages.embedding, so a query vector is never
-- embedded with a different model than the one that produced the stored
-- vectors (see pg_search.py's embed_query -- a mismatched model still
-- yields a well-formed but meaningless cosine distance, silently).
-- CHECK (id = 1) makes the single-row assumption every reader relies on
-- (pg_search.py, deploy/manifest_probe.py, smoke_checks.py --
-- all read WHERE id = 1) structural instead of merely conventional: a
-- second row was previously possible and would have made "the" model
-- ambiguous with no error until some reader's unfiltered query hit it.
CREATE TABLE IF NOT EXISTS corpus.embedding_model (
    id           SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model        TEXT NOT NULL,
    dims         INTEGER NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
