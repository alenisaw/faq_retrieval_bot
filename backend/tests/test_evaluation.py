import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    annotator_agreement,
    assign_split,
    best_method,
    collapse_to_clusters,
    evaluate,
    hit_at_k,
    leave_one_out,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
    threshold_report,
    topic_confusion,
    tune_threshold,
    validate_eval_set,
    validate_faq,
)
from src.retrievers import build_retriever


def test_metric_functions():
    ranked = ["a", "b", "c", "d"]
    rel = {"c"}
    assert hit_at_k(ranked, rel, 1) == 0.0
    assert hit_at_k(ranked, rel, 3) == 1.0
    assert precision_at_k(ranked, rel, 3) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert recall_at_k(ranked, {"c", "z"}, 3) == 0.5
    assert reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, rel, k=2) == 0.0
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_collapse_to_clusters_keeps_first_occurrence():
    faq_to_cluster = {1: "A", 2: "A", 3: "B", 4: "A", 5: "C"}
    assert collapse_to_clusters([2, 1, 3, 4, 5], faq_to_cluster) == ["A", "B", "C"]


def test_validate_faq_requires_schema(faq_df):
    with pytest.raises(ValueError, match="answer_cluster"):
        validate_faq(faq_df.drop(columns=["answer_cluster"]))
    with pytest.raises(ValueError, match="уникальным"):
        validate_faq(pd.concat([faq_df, faq_df.head(1)]))


def test_validate_eval_set_errors(faq_df, eval_set):
    queries, qrels = eval_set
    with pytest.raises(ValueError, match="Нет разметки"):
        validate_eval_set(queries, qrels[qrels["query_id"] != 1])
    bad = pd.concat([qrels, pd.DataFrame([{"query_id": 5, "answer_cluster": 10, "topic": "x"}])])
    with pytest.raises(ValueError, match="не по теме"):
        validate_eval_set(queries, bad)
    unknown = qrels.assign(answer_cluster=qrels["answer_cluster"].replace({10: 999}))
    with pytest.raises(ValueError, match="кластеры"):
        validate_eval_set(queries, unknown, faq_df)


def test_evaluate_protocol_a(faq_df, eval_set):
    queries, qrels = eval_set
    run = evaluate(build_retriever("tfidf_char"), faq_df, queries, qrels, split=None,
                   measure_memory=False)
    m = run.metrics
    assert m["n_queries"] == 4                       # OOS не входят в метрики качества
    assert 0 <= m["p@1"] <= m["hit@3"] <= m["hit@5"] <= 1
    assert 0 <= m["mrr@10"] <= 1
    assert m["index_time_s"] >= 0 and m["latency_p95_ms"] >= m["latency_mean_ms"] * 0
    assert len(run.per_query) == 6                   # но задержка и score есть для всех
    assert set(run.results.columns) == {"query_id", "rank", "faq_id", "score", "latency_ms"}
    assert run.results.groupby("query_id")["rank"].max().max() <= 10
    record = run.run_record()
    assert record["method"] == "tfidf_char" and "params_json" in record


def test_evaluate_split_filter(faq_df, eval_set):
    queries, qrels = eval_set
    run = evaluate(build_retriever("bm25"), faq_df, queries, qrels, split="dev", measure_memory=False)
    assert set(run.per_query["query_id"]) == {3, 4, 5}


def test_leave_one_out_masks_exact_duplicates(faq_df):
    run = leave_one_out(build_retriever("tfidf_char"), faq_df, measure_memory=False)
    qids = set(run.per_query["query_id"])
    # кластер 40 — один и тот же вопрос дважды: после маскирования дубля
    # искать нечего, поэтому faq 8 и 9 в LOO не участвуют
    assert not qids & {8, 9}
    # кластеры 10, 20, 30, 50 по 2 разных вопроса -> 8 запросов; 11 и 60 одиночные
    assert qids == {1, 2, 4, 5, 6, 7, 10, 11}
    assert run.protocol == "loo"


def test_run_benchmark_and_best_method(faq_df):
    table, runs = run_benchmark(["bow", "tfidf_char"], faq_df, protocol="loo",
                                measure_memory=False, verbose=False)
    assert set(table["method"]) == {"bow", "tfidf_char"}
    assert table["mrr@10"].is_monotonic_decreasing
    assert best_method(table) == table.iloc[0]["method"]
    assert set(runs) == {"bow", "tfidf_char"}


def test_assign_split_is_deterministic_and_stratified():
    queries = pd.DataFrame({"query_id": range(20), "text": "q", "is_oos": [False] * 16 + [True] * 4})
    qrels = pd.DataFrame({"query_id": range(16), "answer_cluster": 1,
                          "topic": ["a"] * 10 + ["b"] * 5 + ["c"]})
    s1 = assign_split(queries, qrels, dev_frac=0.3, seed=1)
    s2 = assign_split(queries, qrels, dev_frac=0.3, seed=1)
    assert s1["split"].tolist() == s2["split"].tolist()
    dev = s1[s1["split"] == "dev"]
    assert (dev["query_id"] < 10).sum() == 3          # 30% темы "a"
    assert ((dev["query_id"] >= 10) & (dev["query_id"] < 15)).sum() == 2  # round(1.5) = 2
    assert 15 not in set(dev["query_id"])               # одиночная тема — в test
    assert dev["is_oos"].sum() == 1                    # OOS — своя страта


def test_threshold_tuning_separates_oos():
    pq = pd.DataFrame({
        "top1_score": [0.9, 0.8, 0.7, 0.3, 0.2],
        "is_oos": [False, False, False, True, True],
        "hit@1": [1.0, 1.0, 0.0, np.nan, np.nan],
    })
    best = tune_threshold(pq)
    assert 0.3 < best["threshold"] <= 0.7
    assert best["balanced_accuracy"] == 1.0
    assert best["refuse_recall"] == 1.0 and best["coverage"] == 1.0
    assert best["answered_correct"] == pytest.approx(2 / 3)
    report = threshold_report(pq, 0.0)
    assert report["refuse_recall"] == 0.0 and report["coverage"] == 1.0
    with pytest.raises(ValueError):
        tune_threshold(pq[~pq["is_oos"]])


def test_annotator_agreement():
    a = pd.DataFrame({"query_id": [1, 2, 3, 3], "answer_cluster": [10, 20, 30, 31],
                      "topic": ["dormitory", "tuition", "programs", "programs"]})
    b = pd.DataFrame({"query_id": [1, 2, 3], "answer_cluster": [10, 21, 31],
                      "topic": ["dormitory", "tuition", "scores_grants"]})
    m, diff = annotator_agreement(a, b, query_ids=[1, 2, 3, 4])   # 4 — оба считают "не по теме"
    assert m["n"] == 4
    assert m["topic_agreement"] == 0.75
    assert m["cluster_overlap"] == 0.75                 # у запроса 3 есть общий кластер 31
    assert m["cluster_agreement"] == 0.5                # по наименьшему id: 30 vs 31, 20 vs 21
    assert -1 <= m["topic_kappa"] <= 1 and -1 <= m["cluster_kappa"] <= 1
    assert set(diff["query_id"]) == {2, 3}
    same, none = annotator_agreement(a, a)
    assert same["topic_kappa"] == 1.0 and same["cluster_kappa"] == 1.0 and none.empty


def test_topic_confusion(faq_df, eval_set):
    queries, qrels = eval_set
    run = evaluate(build_retriever("tfidf_char"), faq_df, queries, qrels, split=None,
                   measure_memory=False)
    cm, labels = topic_confusion(run.per_query)
    assert cm.shape == (len(labels), len(labels))
    assert cm.sum() == 4
    assert labels.index("dormitory") < labels.index("contacts")  # порядок из config.TOPICS
