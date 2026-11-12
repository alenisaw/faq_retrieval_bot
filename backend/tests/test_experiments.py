import json

import pytest

from src import config
from src.chatbot import ChatBot
from src.evaluation import best_method
from src.experiments import run_all


@pytest.fixture()
def sandbox(tmp_path, monkeypatch, faq_df, eval_set):
    """Все пути проекта -> временная папка; игрушечные FAQ и тестовый набор."""
    for name in ("METRICS_DIR", "FIGURES_DIR", "MODELS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "faq.db")  # не трогать настоящую базу
    queries, qrels = eval_set
    monkeypatch.setattr(config, "TEST_QUERIES_PATH", tmp_path / "test_queries.csv")
    monkeypatch.setattr(config, "QRELS_PATH", tmp_path / "qrels.csv")
    queries.to_csv(config.TEST_QUERIES_PATH, index=False)
    qrels.to_csv(config.QRELS_PATH, index=False)
    faq_path = tmp_path / "faq_clean.parquet"
    faq_df.to_parquet(faq_path, index=False)
    return faq_path


def test_run_all_loo(sandbox):
    out = run_all("loo", include_neural=False, faq_path=sandbox)
    assert {"preprocessing", "methods", "ablations"} <= set(out)
    assert (config.METRICS_DIR / "metrics_loo.csv").exists()
    assert list(config.FIGURES_DIR.glob("cm_topics_*_loo.png"))
    assert not config.DB_PATH.exists()               # без собранной базы — только CSV


def test_hybrid_components_chosen_on_dev(faq_df, eval_set, monkeypatch):
    from src import experiments

    seen_splits = []
    real = experiments.run_benchmark

    def spy(*args, **kwargs):
        seen_splits.append(kwargs.get("split"))
        return real(*args, **kwargs)

    monkeypatch.setattr(experiments, "run_benchmark", spy)
    queries, qrels = eval_set
    lexical, neural = experiments.choose_hybrid_components(
        faq_df, queries, qrels, neural_candidates=("bm25", "bow"))  # без скачивания моделей
    assert seen_splits == ["dev"]                    # выбор компонент не смотрит на test
    assert lexical not in {"rule_based", "bm25", "bow"} and neural in {"bm25", "bow"}


def test_runs_are_saved_to_db_and_eval_set_not_reloaded(sandbox, faq_df, eval_set):
    from src import db
    from src.evaluation import run_benchmark
    from src.experiments import save_runs_to_db

    with db.connect() as conn:
        db.init_db(conn)
        db.load_corpus(conn, faq_df)
    queries, qrels = eval_set
    _, runs = run_benchmark(["bm25"], faq_df, protocol="test", queries=queries, qrels=qrels,
                            measure_memory=False, verbose=False)
    first = save_runs_to_db(runs, queries, qrels)
    second = save_runs_to_db(runs, queries, qrels)   # тот же набор: прошлые результаты не стираются
    with db.connect() as conn:
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        run_ids = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM results")}
    assert n_runs == 2 and run_ids == set(first + second)


def test_run_all_test_protocol_selects_bot_method(sandbox, faq_df):
    out = run_all("test", include_neural=False, faq_path=sandbox)
    assert {"thresholds", "classifiers", "topic_filter"} <= set(out)
    for name in ("metrics_test", "ablations_test", "thresholds_test", "topic_classifiers_test",
                 "topic_filter_test", "preprocessing_modes"):
        assert (config.METRICS_DIR / f"{name}.csv").exists(), name
    assert (config.MODELS_DIR / "topic_classifier.joblib").exists()

    cfg = json.loads((config.MODELS_DIR / "chatbot.json").read_text(encoding="utf-8"))
    assert cfg["method"] == best_method(out["methods"])  # ничья по MRR -> быстрее по p95
    assert cfg["threshold"] is not None
    bot = ChatBot.from_saved(faq_df)               # бот поднимается из сохранённого выбора
    assert bot.retriever.name == cfg["method"]
