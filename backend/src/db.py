"""SQLite-база data/processed/faq.db: корпус, тест-набор, прогоны, журнал бота.

    from src import db
    with db.connect() as conn:
        db.init_db(conn)
        db.load_corpus(conn, faq_df, keywords=..., entities=...)
        db.save_run(conn, run)                       # RunResult из src.evaluation
        df = db.run_report(conn, "mrr_by_method")   # запрос из sql/queries.sql

Схема — sql/schema.sql, отчёты — sql/queries.sql. Итоговые таблицы отчёта
строятся SQL-запросами через pandas.read_sql.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from . import config

CORPUS_TABLES = ("keywords", "entities", "faqs", "answer_clusters", "topics")


# --- Подключение ------------------------------------------------------------

@contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Соединение с включёнными внешними ключами; commit при выходе без ошибки."""
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection, schema_path: str | Path | None = None) -> None:
    script = Path(schema_path or config.SCHEMA_SQL_PATH).read_text(encoding="utf-8")
    conn.executescript(script)
    conn.execute("PRAGMA foreign_keys = ON")  # executescript делает COMMIT, настройка сохраняется


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}


def _none_if_nan(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --- Корпус -------------------------------------------------------------------

def load_corpus(conn: sqlite3.Connection, faq_df: pd.DataFrame,
                keywords: pd.DataFrame | None = None,
                entities: pd.DataFrame | None = None) -> dict[str, int]:
    """Заменить корпус (topics, answer_clusters, faqs, keywords, entities).

    Прогоны, разметка и журнал бота сохраняются: faq_id и cluster_id стабильны
    (индекс строки исходника и минимальный faq_id кластера). Если после
    пересборки данных на какие-то FAQ ссылаться стало нечем, такие строки
    results/qrels удаляются, а в chat_log ссылка обнуляется — счётчики в ответе.
    """
    required = {"faq_id", "question", "answer", "topic", "answer_cluster"}
    missing = required - set(faq_df.columns)
    if missing:
        raise ValueError(f"В FAQ нет колонок {sorted(missing)}")
    unknown = set(faq_df["topic"].astype(str)) - set(config.TOPICS)
    if unknown:
        raise ValueError(f"Темы не из config.TOPICS: {sorted(unknown)}")

    df = faq_df.sort_values("faq_id")
    is_dated = df["is_dated"].astype(bool) if "is_dated" in df else pd.Series(False, index=df.index)
    topic_ids = {slug: i for i, slug in enumerate(config.TOPICS, start=1)}

    conn.commit()                               # PRAGMA внутри открытой транзакции игнорируется
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in CORPUS_TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.executemany("INSERT INTO topics (topic_id, slug, name) VALUES (?, ?, ?)",
                         [(i, slug, config.TOPICS[slug]) for slug, i in topic_ids.items()])
        clusters = (df.assign(is_dated=is_dated)
                    .groupby("answer_cluster", sort=True)
                    .agg(canonical_answer=("answer", "first"), n_faqs=("faq_id", "size"),
                         is_dated=("is_dated", "max")))
        conn.executemany(
            "INSERT INTO answer_clusters (cluster_id, canonical_answer, n_faqs, is_dated) VALUES (?, ?, ?, ?)",
            [(int(cid), r.canonical_answer, int(r.n_faqs), int(bool(r.is_dated)))
             for cid, r in clusters.iterrows()])
        conn.executemany(
            "INSERT INTO faqs (faq_id, question, answer, orig_class, topic_id, cluster_id, is_dated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(int(r.faq_id), r.question, r.answer, _none_if_nan(getattr(r, "orig_class", None)) or None,
              topic_ids[str(r.topic)], int(r.answer_cluster), int(bool(d)))
             for r, d in zip(df.itertuples(index=False), is_dated)])
        if keywords is not None and len(keywords):
            conn.executemany("INSERT OR IGNORE INTO keywords (faq_id, term, tfidf_score) VALUES (?, ?, ?)",
                             [(int(f), str(t), float(s)) for f, t, s in
                              keywords[["faq_id", "term", "tfidf_score"]].itertuples(index=False)])
        if entities is not None and len(entities):
            conn.executemany("INSERT OR IGNORE INTO entities (faq_id, text, type) VALUES (?, ?, ?)",
                             [(int(f), str(t), str(e)) for f, t, e in
                              entities[["faq_id", "text", "type"]].itertuples(index=False)])
        orphans = _repair_orphans(conn)
        broken = conn.execute("PRAGMA foreign_key_check").fetchall()
        if broken:
            raise sqlite3.IntegrityError(f"Нарушены внешние ключи после загрузки корпуса: {broken[:5]}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    return {**orphans, **{t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                          for t in reversed(CORPUS_TABLES)}}


def _repair_orphans(conn: sqlite3.Connection) -> dict[str, int]:
    out = {}
    cur = conn.execute("DELETE FROM results WHERE faq_id NOT IN (SELECT faq_id FROM faqs) "
                       "OR (loo_faq_id IS NOT NULL AND loo_faq_id NOT IN (SELECT faq_id FROM faqs))")
    out["orphan_results_deleted"] = cur.rowcount
    cur = conn.execute("DELETE FROM qrels WHERE cluster_id NOT IN (SELECT cluster_id FROM answer_clusters)")
    out["orphan_qrels_deleted"] = cur.rowcount
    cur = conn.execute("UPDATE chat_log SET top_faq_id = NULL "
                       "WHERE top_faq_id IS NOT NULL AND top_faq_id NOT IN (SELECT faq_id FROM faqs)")
    out["chat_log_unlinked"] = cur.rowcount
    return out


def load_eval_set(conn: sqlite3.Connection, queries: pd.DataFrame, qrels: pd.DataFrame) -> dict[str, int]:
    """Заменить таблицы queries и qrels содержимым eval/*.csv.

    Прогоны протокола A ссылаются на query_id, поэтому их результаты
    удаляются вместе со старым набором (набор замораживается один раз).
    """
    topic_ids = dict(conn.execute("SELECT slug, topic_id FROM topics"))
    if not topic_ids:
        raise RuntimeError("Корпус не загружен: сначала load_corpus (пункт 1 меню)")
    is_oos = queries["is_oos"] if "is_oos" in queries else pd.Series(False, index=queries.index)
    is_oos = is_oos.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "да"})
    split = queries["split"] if "split" in queries else pd.Series("test", index=queries.index)
    author = queries["author"] if "author" in queries else pd.Series(None, index=queries.index)

    unknown_topics = set(qrels.get("topic", pd.Series(dtype=str)).dropna()) - set(topic_ids)
    if unknown_topics:
        raise ValueError(f"В qrels темы не из config.TOPICS: {sorted(unknown_topics)}")

    cur = conn.execute("DELETE FROM results WHERE query_id IS NOT NULL")
    dropped_results = cur.rowcount
    conn.execute("DELETE FROM qrels")
    conn.execute("DELETE FROM queries")
    conn.executemany(
        "INSERT INTO queries (query_id, text, author, split, is_oos) VALUES (?, ?, ?, ?, ?)",
        [(int(q), str(t), _none_if_nan(a), str(s), int(o))
         for q, t, a, s, o in zip(queries["query_id"], queries["text"], author, split, is_oos)])
    conn.executemany(
        "INSERT OR IGNORE INTO qrels (query_id, cluster_id, topic_id) VALUES (?, ?, ?)",
        [(int(q), int(c), topic_ids.get(t) if isinstance(t, str) else None)
         for q, c, t in zip(qrels["query_id"], qrels["answer_cluster"],
                            qrels["topic"] if "topic" in qrels else [None] * len(qrels))])
    return {"queries": len(queries), "qrels": len(qrels), "results_deleted": dropped_results}


# --- Прогоны ------------------------------------------------------------------

def get_or_create_method(conn: sqlite3.Connection, name: str, params: dict | str | None = None) -> int:
    params_json = params if isinstance(params, str) else json.dumps(
        params or {}, ensure_ascii=False, sort_keys=True, default=str)
    row = conn.execute("SELECT method_id FROM methods WHERE name = ? AND params_json = ?",
                       (name, params_json)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO methods (name, params_json) VALUES (?, ?)",
                        (name, params_json)).lastrowid


def save_run(conn: sqlite3.Connection, run, store_results: bool = True) -> int:
    """RunResult (src.evaluation) -> строка runs + топ-10 каждого запроса в results."""
    rec = run.run_record()
    params = json.loads(rec["params_json"]) if rec.get("params_json") else {}
    method_id = get_or_create_method(conn, rec["method"], params)
    protocol = rec["protocol"] if rec["protocol"] in ("loo", "test", "dev", "all") else "all"
    m = run.metrics
    metrics = {k: _none_if_nan(v) for k, v in m.items()}
    run_id = conn.execute(
        "INSERT INTO runs (method_id, protocol, created_at, index_time_s, peak_mem_mb, model_size_mb, "
        "index_size_mb, latency_mean_ms, latency_p95_ms, metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (method_id, protocol, rec["created_at"], _none_if_nan(m.get("index_time_s")),
         _none_if_nan(m.get("peak_mem_mb")), _none_if_nan(m.get("model_size_mb")),
         _none_if_nan(m.get("index_size_mb")), _none_if_nan(m.get("latency_mean_ms")),
         _none_if_nan(m.get("latency_p95_ms")), json.dumps(metrics, ensure_ascii=False, default=str)),
    ).lastrowid

    if store_results and len(run.results):
        res = run.results
        if protocol == "loo":
            rows = [(run_id, None, int(q), int(r), int(f), _none_if_nan(s), _none_if_nan(lat))
                    for q, r, f, s, lat in res[["query_id", "rank", "faq_id", "score", "latency_ms"]]
                    .itertuples(index=False)]
        else:
            known = {r[0] for r in conn.execute("SELECT query_id FROM queries")}
            missing = set(res["query_id"].astype(int)) - known
            if missing:
                raise ValueError(f"Запросов {sorted(missing)[:5]}... нет в таблице queries: сначала "
                                 f"загрузите тест-набор (db.load_eval_set)")
            rows = [(run_id, int(q), None, int(r), int(f), _none_if_nan(s), _none_if_nan(lat))
                    for q, r, f, s, lat in res[["query_id", "rank", "faq_id", "score", "latency_ms"]]
                    .itertuples(index=False)]
        conn.executemany(
            "INSERT INTO results (run_id, query_id, loo_faq_id, rank, faq_id, score, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return run_id


def latest_metrics(conn: sqlite3.Connection, protocol: str | None = None) -> pd.DataFrame:
    """Метрики последнего прогона каждого метода (из metrics_json) — для API и меню."""
    sql = """
        SELECT m.name AS method, r.protocol, r.run_id, r.created_at, r.metrics_json
        FROM runs r JOIN methods m ON m.method_id = r.method_id
        WHERE r.run_id IN (SELECT MAX(r2.run_id) FROM runs r2 JOIN methods m2 ON m2.method_id = r2.method_id
                           GROUP BY m2.name, r2.protocol)
    """
    params: tuple = ()
    if protocol:
        sql += " AND r.protocol = ?"
        params = (protocol,)
    df = pd.read_sql(sql + " ORDER BY r.protocol, r.run_id", conn, params=params)
    if df.empty:
        return df.drop(columns="metrics_json")
    metrics = pd.DataFrame([json.loads(s) for s in df.pop("metrics_json")])
    out = pd.concat([df.reset_index(drop=True), metrics], axis=1)
    if "mrr@10" in out:
        out = out.sort_values(["protocol", "mrr@10"], ascending=[True, False], kind="stable")
    return out.reset_index(drop=True)


# --- Журнал чат-бота ----------------------------------------------------------

def log_chat(conn: sqlite3.Connection, resp, method_params: dict | None = None,
             source: str = "cli") -> int:
    """ChatResponse (src.chatbot) -> строка chat_log."""
    method_id = get_or_create_method(conn, resp.method or "unknown", method_params or {})
    top = resp.top
    return conn.execute(
        "INSERT INTO chat_log (ts, query, method_id, top_faq_id, score, answered, latency_ms, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), resp.query, method_id,
         int(_none_if_nan(top.faq_id)) if top else None, float(top.score) if top else None,
         int(bool(resp.answered)), float(resp.latency_ms), source),
    ).lastrowid


def make_chat_logger(db_path: str | Path | None = None, method_params: dict | None = None,
                     source: str = "cli") -> Callable:
    """Колбэк для ChatBot(on_answer=...): каждый ответ — строка в chat_log.

    Ошибка записи не должна ронять диалог: она печатается и пропускается.
    """
    def _log(resp) -> None:
        try:
            with connect(db_path) as conn:
                log_chat(conn, resp, method_params, source)
        except sqlite3.Error as exc:
            print(f"[chat_log] не записано: {exc}")
    return _log


# --- SQL-отчёты -----------------------------------------------------------------

@dataclass(frozen=True)
class Report:
    name: str
    title: str
    description: str
    sql: str


_NAME_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)


def load_reports(path: str | Path | None = None) -> dict[str, Report]:
    """Разобрать sql/queries.sql на именованные запросы (в порядке файла)."""
    text = Path(path or config.QUERIES_SQL_PATH).read_text(encoding="utf-8")
    marks = list(_NAME_RE.finditer(text))
    reports: dict[str, Report] = {}
    for i, m in enumerate(marks):
        block = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        title, desc, sql_lines = m.group(1), [], []
        for line in block.strip("\n").splitlines():
            stripped = line.strip()
            if not sql_lines and stripped.startswith("--"):
                body = stripped[2:].strip()
                if body.lower().startswith("title:"):
                    title = body[6:].strip()
                elif body:
                    desc.append(body)
            elif stripped or sql_lines:
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip().rstrip(";")
        reports[m.group(1)] = Report(m.group(1), title, " ".join(desc), sql)
    return reports


def run_report(conn: sqlite3.Connection, name: str, path: str | Path | None = None) -> pd.DataFrame:
    reports = load_reports(path)
    if name not in reports:
        raise KeyError(f"Нет запроса {name!r}. Доступны: {', '.join(reports)}")
    return pd.read_sql(reports[name].sql, conn)
