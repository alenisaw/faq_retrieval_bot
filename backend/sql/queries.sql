-- SQL-отчёты по faq.db. Каждый запрос начинается с "-- name:" (ключ) и "-- title:"
-- (заголовок для меню и API); строки "-- " после них — описание.
-- Вызываются из пункта 6 меню main.py и через GET /api/reports/{name}.
-- "Последний прогон" = максимальный run_id для пары (метод, протокол).


-- name: mrr_by_method
-- title: MRR@10 и P@1 по методам
-- Метрики считаются заново из results, независимо от Python: выдача сворачивается
-- до кластеров ответов (v_result_clusters), берётся ранг первого правильного.
-- Учитывается топ-10 FAQ, поэтому MRR может быть чуть ниже, чем в metrics_json.
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r
    JOIN methods m ON m.method_id = r.method_id
    GROUP BY m.name, r.protocol
),
per_query AS (
    SELECT o.run_id, o.first_rel_rank
    FROM v_query_outcome o
    JOIN latest l ON l.run_id = o.run_id
)
SELECT m.name                                                        AS method,
       r.protocol,
       COUNT(*)                                                      AS n_queries,
       ROUND(AVG(CASE WHEN pq.first_rel_rank = 1 THEN 1.0 ELSE 0 END), 3)  AS p_at_1,
       ROUND(AVG(CASE WHEN pq.first_rel_rank <= 3 THEN 1.0 ELSE 0 END), 3) AS hit_at_3,
       ROUND(AVG(CASE WHEN pq.first_rel_rank <= 5 THEN 1.0 ELSE 0 END), 3) AS hit_at_5,
       ROUND(AVG(COALESCE(1.0 / pq.first_rel_rank, 0)), 3)          AS mrr_at_10
FROM per_query pq
JOIN runs r    ON r.run_id = pq.run_id
JOIN methods m ON m.method_id = r.method_id
GROUP BY r.run_id
ORDER BY r.protocol, mrr_at_10 DESC;


-- name: p1_by_topic
-- title: P@1 по темам и методам
-- Темы, в которых у метода меньше 5 запросов, не показываются (HAVING).
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r JOIN methods m ON m.method_id = r.method_id
    GROUP BY m.name, r.protocol
)
SELECT m.name    AS method,
       r.protocol,
       t.name    AS topic,
       COUNT(*)  AS n_queries,
       ROUND(AVG(CASE WHEN o.first_rel_rank = 1 THEN 1.0 ELSE 0 END), 3) AS p_at_1
FROM v_query_outcome o
JOIN latest l  ON l.run_id = o.run_id
JOIN runs r    ON r.run_id = o.run_id
JOIN methods m ON m.method_id = r.method_id
JOIN topics t  ON t.topic_id = o.gold_topic_id
GROUP BY r.run_id, t.topic_id
HAVING COUNT(*) >= 5
ORDER BY r.protocol, t.name, p_at_1 DESC;


-- name: method_rank_in_topic
-- title: Место метода внутри каждой темы
-- RANK() по P@1 внутри темы: видно, что лучший в среднем метод не везде первый.
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r JOIN methods m ON m.method_id = r.method_id
    GROUP BY m.name, r.protocol
),
topic_p1 AS (
    SELECT m.name AS method, r.protocol, t.name AS topic, COUNT(*) AS n_queries,
           AVG(CASE WHEN o.first_rel_rank = 1 THEN 1.0 ELSE 0 END) AS p1
    FROM v_query_outcome o
    JOIN latest l  ON l.run_id = o.run_id
    JOIN runs r    ON r.run_id = o.run_id
    JOIN methods m ON m.method_id = r.method_id
    JOIN topics t  ON t.topic_id = o.gold_topic_id
    GROUP BY r.run_id, t.topic_id
)
SELECT protocol, topic, method, n_queries, ROUND(p1, 3) AS p_at_1,
       RANK() OVER (PARTITION BY protocol, topic ORDER BY p1 DESC) AS place
FROM topic_p1
ORDER BY protocol, topic, place, method;


-- name: top_keywords_per_topic
-- title: Топ-5 ключевых слов каждой темы
-- Сумма TF-IDF терма по всем FAQ темы, ROW_NUMBER() внутри темы.
WITH term_weight AS (
    SELECT f.topic_id, k.term, SUM(k.tfidf_score) AS weight, COUNT(*) AS n_faqs
    FROM keywords k
    JOIN faqs f ON f.faq_id = k.faq_id
    GROUP BY f.topic_id, k.term
),
ranked AS (
    SELECT tw.*, ROW_NUMBER() OVER (PARTITION BY tw.topic_id ORDER BY tw.weight DESC, tw.term) AS rn
    FROM term_weight tw
)
SELECT t.name AS topic, r.rn AS place, r.term, ROUND(r.weight, 3) AS tfidf_sum, r.n_faqs
FROM ranked r
JOIN topics t ON t.topic_id = r.topic_id
WHERE r.rn <= 5
ORDER BY t.name, r.rn;


-- name: duplicate_answers
-- title: Один ответ на несколько вопросов (кластеры 2+)
SELECT c.cluster_id,
       t.name                                  AS topic,
       COUNT(f.faq_id)                         AS n_questions,
       GROUP_CONCAT(f.faq_id, ', ')            AS faq_ids,
       SUBSTR(c.canonical_answer, 1, 120)      AS answer_start
FROM answer_clusters c
JOIN faqs f   ON f.cluster_id = c.cluster_id
JOIN topics t ON t.topic_id = f.topic_id
GROUP BY c.cluster_id
HAVING COUNT(f.faq_id) > 1
ORDER BY n_questions DESC, c.cluster_id;


-- name: hard_queries
-- title: Трудные запросы: топ-1 неверный у всех методов
-- Запрос попадает сюда, если НЕ существует последнего прогона, где правильный
-- ответ стоял первым. Только ручные запросы (протокол A).
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r JOIN methods m ON m.method_id = r.method_id
    WHERE r.protocol IN ('test', 'dev', 'all')
    GROUP BY m.name, r.protocol
),
evaluated AS (
    SELECT DISTINCT o.query_id
    FROM v_query_outcome o JOIN latest l ON l.run_id = o.run_id
    WHERE o.query_id IS NOT NULL
)
SELECT q.query_id, q.text, q.author, t.name AS gold_topic,
       (SELECT COUNT(*) FROM latest) AS n_runs
FROM queries q
JOIN evaluated e ON e.query_id = q.query_id
LEFT JOIN qrels g ON g.query_id = q.query_id
LEFT JOIN topics t ON t.topic_id = g.topic_id
WHERE NOT EXISTS (
    SELECT 1
    FROM v_query_outcome o
    JOIN latest l ON l.run_id = o.run_id
    WHERE o.query_id = q.query_id AND o.first_rel_rank = 1
)
GROUP BY q.query_id
ORDER BY gold_topic, q.query_id;


-- name: dated_answers
-- title: Ответы с датами и суммами, которые могли устареть
SELECT f.faq_id,
       t.name                                                                   AS topic,
       f.is_dated,
       GROUP_CONCAT(CASE WHEN e.type = 'DATE'  THEN e.text END, '; ')            AS dates,
       GROUP_CONCAT(CASE WHEN e.type = 'MONEY' THEN e.text END, '; ')            AS amounts,
       SUBSTR(f.question, 1, 90)                                                AS question
FROM faqs f
JOIN topics t   ON t.topic_id = f.topic_id
JOIN entities e ON e.faq_id = f.faq_id AND e.type IN ('DATE', 'MONEY')
GROUP BY f.faq_id
ORDER BY f.is_dated DESC, t.name, f.faq_id;


-- name: latency_by_method
-- title: Средняя и p95 задержка по методам
-- Задержка одного запроса записана в каждой строке его выдачи, поэтому берём rank = 1.
-- NTILE(20) делит запросы на 20 групп по задержке: нижняя граница 20-й — оценка p95.
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r JOIN methods m ON m.method_id = r.method_id
    GROUP BY m.name, r.protocol
),
per_query AS (
    SELECT res.run_id, res.latency_ms,
           NTILE(20) OVER (PARTITION BY res.run_id ORDER BY res.latency_ms) AS bucket
    FROM results res
    JOIN latest l ON l.run_id = res.run_id
    WHERE res.rank = 1
)
SELECT m.name AS method, r.protocol,
       COUNT(*)                                                    AS n_queries,
       ROUND(AVG(pq.latency_ms), 3)                                AS mean_ms,
       ROUND(MIN(CASE WHEN pq.bucket = 20 THEN pq.latency_ms END), 3) AS p95_ms,
       ROUND(MAX(pq.latency_ms), 3)                                AS max_ms,
       ROUND(r.index_time_s, 3)                                    AS index_time_s
FROM per_query pq
JOIN runs r    ON r.run_id = pq.run_id
JOIN methods m ON m.method_id = r.method_id
GROUP BY r.run_id
ORDER BY mean_ms;


-- name: mrr_change_between_runs
-- title: Изменение MRR между соседними прогонами одного метода
-- LAG() сравнивает прогон с предыдущим прогоном того же метода и протокола:
-- видно, как менялось качество после правок предобработки или данных.
WITH run_mrr AS (
    SELECT o.run_id, AVG(COALESCE(1.0 / o.first_rel_rank, 0)) AS mrr, COUNT(*) AS n_queries
    FROM v_query_outcome o
    GROUP BY o.run_id
)
SELECT m.name AS method, r.protocol, r.run_id, r.created_at, rm.n_queries,
       ROUND(rm.mrr, 3) AS mrr_at_10,
       ROUND(rm.mrr - LAG(rm.mrr) OVER (PARTITION BY m.name, r.protocol ORDER BY r.run_id), 3) AS delta
FROM run_mrr rm
JOIN runs r    ON r.run_id = rm.run_id
JOIN methods m ON m.method_id = r.method_id
ORDER BY m.name, r.protocol, r.run_id;


-- name: chat_topics
-- title: Частые темы и доля отказов в журнале чат-бота
-- Тема берётся по топ-1 ответу; share — доля темы среди всех вопросов (оконная SUM).
SELECT COALESCE(t.name, '— нет ответа —')                         AS topic,
       COUNT(*)                                                   AS n_questions,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)         AS share_pct,
       ROUND(100.0 * SUM(1 - c.answered) / COUNT(*), 1)           AS refused_pct,
       ROUND(AVG(c.score), 3)                                     AS mean_score
FROM chat_log c
LEFT JOIN faqs f   ON f.faq_id = c.top_faq_id
LEFT JOIN topics t ON t.topic_id = f.topic_id
GROUP BY t.topic_id
ORDER BY n_questions DESC;


-- name: resources_by_method
-- title: Ресурсы методов: индексация, память, размер
WITH latest AS (
    SELECT MAX(r.run_id) AS run_id
    FROM runs r JOIN methods m ON m.method_id = r.method_id
    GROUP BY m.name
)
SELECT m.name AS method, r.protocol, r.created_at,
       ROUND(r.index_time_s, 3)    AS index_time_s,
       ROUND(r.peak_mem_mb, 2)     AS peak_mem_mb,
       ROUND(r.index_size_mb, 3)   AS index_size_mb,
       ROUND(r.model_size_mb, 1)   AS model_size_mb,
       ROUND(r.latency_mean_ms, 3) AS latency_mean_ms,
       ROUND(r.latency_p95_ms, 3)  AS latency_p95_ms
FROM runs r
JOIN latest l  ON l.run_id = r.run_id
JOIN methods m ON m.method_id = r.method_id
ORDER BY r.index_time_s;
