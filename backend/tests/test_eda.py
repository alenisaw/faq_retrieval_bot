"""EDA строит все графики и сводку на игрушечной базе."""

import json

import pytest

from src import config, eda


@pytest.fixture()
def tmp_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    return tmp_path


def test_run_all_writes_figures_and_summary(faq_df, tmp_reports):
    out = tmp_reports / "figures"
    summary = eda.run_all(faq_df, out_dir=out)

    expected = {"eda_topics.png", "eda_lengths.png", "eda_top_words.png", "eda_ngrams.png",
                "eda_vocab.png", "eda_clusters.png"}
    assert expected <= {p.name for p in out.iterdir()}
    assert summary["n_faqs"] == len(faq_df)
    assert summary["clusters"]["n_clusters"] == faq_df["answer_cluster"].nunique()
    saved = json.loads((tmp_reports / "eda_summary.json").read_text(encoding="utf-8"))
    assert set(saved["figures"]) >= expected


def test_iqr_outliers():
    import pandas as pd

    upper, n = eda.iqr_outliers(pd.Series([1, 2, 2, 3, 3, 3, 4, 100]))
    assert n == 1 and upper < 100


def test_top_ngrams_counts_lemmas():
    grams = dict(eda.top_ngrams(["военная кафедра есть", "военной кафедры нет"], n=2))
    assert grams.get("военный кафедра") == 2
