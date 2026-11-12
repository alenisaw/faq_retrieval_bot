import sqlite3

import pandas as pd
import pytest

from src import db
from src.chatbot import ChatBot
from src.evaluation import evaluate, leave_one_out
from src.retrievers import build_retriever


@pytest.fixture()
def conn(tmp_path, faq_df):
    with db.connect(tmp_path / "test.db") as c:
        db.init_db(c)
        db.load_corpus(c, faq_df, keywords=pd.DataFrame({"faq_id": [1, 1, 4], "term": ["общежитие", "есть", "стоимость"],
                                                          "tfidf_score": [0.8, 0.2, 0.9]}),
                       entities=pd.DataFrame({"faq_id": [3, 4], "text": ["50 000 тенге", "2 000 000 тенге"],
                                              "type": ["MONEY", "MONEY"]}))
        yield c


def test_schema_and_corpus(conn, faq_df):
    counts = db.table_counts(conn)
    assert counts["faqs"] == len(faq_df)
    assert counts["answer_clusters"] == faq_df["answer_cluster"].nunique()
    assert counts["topics"] == 12
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("SELECT n_faqs FROM answer_clusters WHERE cluster_id = 10").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO faqs (faq_id, question, answer, topic_id, cluster_id) VALUES (99, 'q', 'a', 1, 12345)")


def test_init_db_is_idempotent(conn):
    db.init_db(conn)
    assert db.table_counts(conn)["faqs"] > 0


def test_unknown_topic_rejected(conn, faq_df):
    bad = faq_df.copy()
    bad.loc[0, "topic"] = "nonexistent"
    with pytest.raises(ValueError):
        db.load_corpus(conn, bad)


def test_loo_run_sql_matches_python(conn, faq_df):
    run = leave_one_out(build_retriever("tfidf_char"), faq_df, measure_memory=False)
    run_id = db.save_run(conn, run)
    sql = db.run_report(conn, "mrr_by_method").set_index("method").loc["tfidf_char"]
    assert sql["n_queries"] == run.metrics["n_queries"]
    assert sql["mrr_at_10"] == pytest.approx(run.metrics["mrr@10"], abs=1e-3)
    assert sql["p_at_1"] == pytest.approx(run.metrics["p@1"], abs=1e-3)
    stored = conn.execute("SELECT COUNT(*) FROM results WHERE run_id = ? AND loo_faq_id IS NOT NULL",
                          (run_id,)).fetchone()[0]
    assert stored == len(run.results)


def test_test_protocol_requires_eval_set(conn, faq_df, eval_set):
    queries, qrels = eval_set
    run = evaluate(build_retriever("tfidf_char"), faq_df, queries, qrels, split=None, measure_memory=False)
    with pytest.raises(ValueError, match="queries"):
        db.save_run(conn, run)
    db.load_eval_set(conn, queries, qrels)
    db.save_run(conn, run)
    sql = db.run_report(conn, "mrr_by_method").set_index("method").loc["tfidf_char"]
    assert sql["n_queries"] == 4                                   # OOS-запросы не считаются
    assert sql["mrr_at_10"] == pytest.approx(run.metrics["mrr@10"], abs=1e-3)


def test_all_reports_execute(conn, faq_df, eval_set):
    queries, qrels = eval_set
    db.load_eval_set(conn, queries, qrels)
    for name in ("tfidf_char", "bm25"):
        db.save_run(conn, leave_one_out(build_retriever(name), faq_df, measure_memory=False))
        db.save_run(conn, evaluate(build_retriever(name), faq_df, queries, qrels, split=None, measure_memory=False))
    bot = ChatBot(build_retriever("tfidf_char"), faq_df, threshold=0.9)
    db.log_chat(conn, bot.answer("общага есть?"), source="api")
    db.log_chat(conn, bot.answer("рецепт плова"), source="cli")

    reports = db.load_reports()
    assert len(reports) >= 10
    for name in reports:
        df = db.run_report(conn, name)
        assert isinstance(df, pd.DataFrame), name
    assert len(db.run_report(conn, "duplicate_answers")) == faq_df.groupby("answer_cluster").size().gt(1).sum()
    assert not db.run_report(conn, "dated_answers").empty
    assert db.run_report(conn, "chat_topics")["n_questions"].sum() == 2
    assert db.run_report(conn, "top_keywords_per_topic").iloc[0]["term"] in {"общежитие", "стоимость", "есть"}


def test_reports_parsed_with_titles():
    reports = db.load_reports()
    assert list(reports)[0] == "mrr_by_method"
    for r in reports.values():
        assert r.title and r.sql and not r.sql.rstrip().endswith(";")
    with pytest.raises(KeyError):
        db.run_report(None, "no_such_report")


def test_reload_corpus_keeps_runs_and_repairs_orphans(conn, faq_df):
    db.save_run(conn, leave_one_out(build_retriever("bm25"), faq_df, measure_memory=False))
    n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    counts = db.load_corpus(conn, faq_df)                         # те же faq_id -> ничего не теряется
    assert counts["orphan_results_deleted"] == 0
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == n_results
    counts = db.load_corpus(conn, faq_df[faq_df["faq_id"] != 1])   # FAQ 1 исчез из данных
    assert counts["orphan_results_deleted"] > 0
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_method_dedup_and_latest_metrics(conn, faq_df):
    a = db.get_or_create_method(conn, "x", {"b": 1, "a": 2})
    assert db.get_or_create_method(conn, "x", {"a": 2, "b": 1}) == a
    assert db.get_or_create_method(conn, "x", {"a": 3}) != a
    for _ in range(2):
        db.save_run(conn, leave_one_out(build_retriever("bow"), faq_df, measure_memory=False))
    latest = db.latest_metrics(conn)
    assert latest["method"].tolist() == ["bow"] and "mrr@10" in latest


def test_chat_logger_survives_db_errors(tmp_path, faq_df, capsys):
    bot = ChatBot(build_retriever("bow"), faq_df,
                  on_answer=db.make_chat_logger(tmp_path / "missing_dir" / "nope" / "x.db"))
    bot.answer("общага есть?")   # базы нет -> схемы нет -> ошибка печатается, ответ возвращается
    assert "chat_log" in capsys.readouterr().out
