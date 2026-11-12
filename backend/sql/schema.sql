-- Схема data/processed/faq.db: корпус FAQ + разметка тест-набора + результаты экспериментов.
-- Применяется src/db.py (init_db). Все CREATE — IF NOT EXISTS, повторный запуск безопасен.
-- Внешние ключи включаются при каждом подключении: PRAGMA foreign_keys = ON.

-- ---------------------------------------------------------------- корпус

CREATE TABLE IF NOT EXISTS topics (
    topic_id    INTEGER PRIMARY KEY,
    slug        TEXT    NOT NULL UNIQUE,          -- ключ из config.TOPICS, он же в qrels.csv
    name        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_clusters (
    cluster_id       INTEGER PRIMARY KEY,         -- = минимальный faq_id в кластере
    canonical_answer TEXT    NOT NULL,            -- ответ строки с минимальным faq_id
    n_faqs           INTEGER NOT NULL CHECK (n_faqs >= 1),
    is_dated         INTEGER NOT NULL DEFAULT 0 CHECK (is_dated IN (0, 1))
);

CREATE TABLE IF NOT EXISTS faqs (
    faq_id      INTEGER PRIMARY KEY,              -- индекс строки исходного CSV
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    orig_class  TEXT,
    topic_id    INTEGER NOT NULL REFERENCES topics (topic_id),
    cluster_id  INTEGER NOT NULL REFERENCES answer_clusters (cluster_id),
    is_dated    INTEGER NOT NULL DEFAULT 0 CHECK (is_dated IN (0, 1))
);
CREATE INDEX IF NOT EXISTS ix_faqs_topic   ON faqs (topic_id);
CREATE INDEX IF NOT EXISTS ix_faqs_cluster ON faqs (cluster_id);

CREATE TABLE IF NOT EXISTS keywords (
    faq_id      INTEGER NOT NULL REFERENCES faqs (faq_id) ON DELETE CASCADE,
    term        TEXT    NOT NULL,
    tfidf_score REAL    NOT NULL,
    PRIMARY KEY (faq_id, term)
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY,
    faq_id      INTEGER NOT NULL REFERENCES faqs (faq_id) ON DELETE CASCADE,
    text        TEXT    NOT NULL,
    type        TEXT    NOT NULL CHECK (type IN ('ORG', 'LOC', 'URL', 'PHONE', 'EMAIL', 'MONEY', 'DATE')),
    UNIQUE (faq_id, text, type)
);
CREATE INDEX IF NOT EXISTS ix_entities_type ON entities (type);

-- ---------------------------------------------------------------- тестовый набор

CREATE TABLE IF NOT EXISTS queries (
    query_id    INTEGER PRIMARY KEY,
    text        TEXT    NOT NULL,
    author      TEXT,
    split       TEXT    NOT NULL DEFAULT 'test' CHECK (split IN ('dev', 'test')),
    is_oos      INTEGER NOT NULL DEFAULT 0 CHECK (is_oos IN (0, 1))   -- запрос не по теме
);

CREATE TABLE IF NOT EXISTS qrels (
    query_id    INTEGER NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    cluster_id  INTEGER NOT NULL REFERENCES answer_clusters (cluster_id),
    topic_id    INTEGER REFERENCES topics (topic_id),
    PRIMARY KEY (query_id, cluster_id)
);

-- ---------------------------------------------------------------- эксперименты

CREATE TABLE IF NOT EXISTS methods (
    method_id   INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,                 -- имя конфигурации: tfidf_char, e5_small, ...
    params_json TEXT    NOT NULL DEFAULT '{}',
    UNIQUE (name, params_json)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY,
    method_id       INTEGER NOT NULL REFERENCES methods (method_id),
    protocol        TEXT    NOT NULL CHECK (protocol IN ('loo', 'test', 'dev', 'all')),
    created_at      TEXT    NOT NULL,
    index_time_s    REAL,
    peak_mem_mb     REAL,
    model_size_mb   REAL,
    index_size_mb   REAL,
    latency_mean_ms REAL,
    latency_p95_ms  REAL,
    metrics_json    TEXT    NOT NULL DEFAULT '{}'  -- все метрики прогона, посчитанные в Python
);

-- Топ-10 выдачи каждого запроса. Запрос — либо ручной (query_id, протокол A),
-- либо вопрос самого FAQ в leave-one-out (loo_faq_id, протокол B).
CREATE TABLE IF NOT EXISTS results (
    result_id   INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    query_id    INTEGER REFERENCES queries (query_id),
    loo_faq_id  INTEGER REFERENCES faqs (faq_id),
    rank        INTEGER NOT NULL CHECK (rank >= 1),
    faq_id      INTEGER NOT NULL REFERENCES faqs (faq_id),
    score       REAL,
    latency_ms  REAL,
    CHECK ((query_id IS NULL) <> (loo_faq_id IS NULL))
);
CREATE INDEX IF NOT EXISTS ix_results_run   ON results (run_id);
CREATE INDEX IF NOT EXISTS ix_results_query ON results (query_id);

CREATE TABLE IF NOT EXISTS chat_log (
    log_id      INTEGER PRIMARY KEY,
    ts          TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    method_id   INTEGER REFERENCES methods (method_id),
    top_faq_id  INTEGER REFERENCES faqs (faq_id) ON DELETE SET NULL,
    score       REAL,
    answered    INTEGER NOT NULL CHECK (answered IN (0, 1)),
    latency_ms  REAL,
    source      TEXT    NOT NULL DEFAULT 'cli' CHECK (source IN ('cli', 'api'))
);

-- ---------------------------------------------------------------- представления

-- Выдача, свёрнутая до кластеров ответов (как в src/evaluation.py): у каждого
-- кластера — лучший ранг его FAQ, crank — место кластера среди разных ответов,
-- is_relevant — кластер совпал с gold (qrels для ручных запросов, кластер
-- самого вопроса для leave-one-out).
DROP VIEW IF EXISTS v_result_clusters;
CREATE VIEW v_result_clusters AS
WITH hits AS (
    SELECT r.run_id, r.query_id, r.loo_faq_id,
           COALESCE('q' || r.query_id, 'f' || r.loo_faq_id) AS qkey,
           r.rank, r.score, f.cluster_id
    FROM results r
    JOIN faqs f ON f.faq_id = r.faq_id
),
by_cluster AS (
    SELECT run_id, query_id, loo_faq_id, qkey, cluster_id,
           MIN(rank) AS best_rank, MAX(score) AS best_score
    FROM hits
    GROUP BY run_id, qkey, cluster_id
)
SELECT b.*,
       ROW_NUMBER() OVER (PARTITION BY b.run_id, b.qkey ORDER BY b.best_rank) AS crank,
       CASE WHEN EXISTS (SELECT 1 FROM qrels g
                         WHERE g.query_id = b.query_id AND g.cluster_id = b.cluster_id)
              OR EXISTS (SELECT 1 FROM faqs src
                         WHERE src.faq_id = b.loo_faq_id AND src.cluster_id = b.cluster_id)
            THEN 1 ELSE 0 END AS is_relevant
FROM by_cluster b;

-- Итог по каждому запросу прогона: ранг первого правильного кластера (NULL —
-- не нашёлся в топ-10), тема и кластер топ-1, gold-тема. Запросы не по теме
-- (is_oos = 1) сюда не входят: у них нет правильного ответа.
DROP VIEW IF EXISTS v_query_outcome;
CREATE VIEW v_query_outcome AS
SELECT v.run_id, v.qkey, v.query_id, v.loo_faq_id,
       MIN(CASE WHEN v.is_relevant = 1 THEN v.crank END)            AS first_rel_rank,
       MAX(CASE WHEN v.crank = 1 THEN v.cluster_id END)               AS top1_cluster_id,
       MAX(CASE WHEN v.crank = 1 THEN v.best_score END)               AS top1_score,
       COALESCE((SELECT MIN(g.topic_id) FROM qrels g WHERE g.query_id = v.query_id),
                (SELECT src.topic_id FROM faqs src WHERE src.faq_id = v.loo_faq_id)) AS gold_topic_id
FROM v_result_clusters v
LEFT JOIN queries q ON q.query_id = v.query_id
WHERE COALESCE(q.is_oos, 0) = 0
GROUP BY v.run_id, v.qkey;
