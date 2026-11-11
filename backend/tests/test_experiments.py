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
