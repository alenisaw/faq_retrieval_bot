from src.classifier import (
    evaluate_classifiers,
    load_classifier,
    make_classifiers,
    save_classifier,
    topic_eval_data,
)


def test_classifiers_train_on_faq_and_test_on_queries(faq_df):
    texts = ["общага есть?", "сколько стоит учеба", "проходной балл", "военная кафедра есть?"]
    topics = ["dormitory", "tuition", "scores_grants", "military"]
    table, details = evaluate_classifiers(faq_df, texts, topics,
                                          models=make_classifiers(include_neural=False))
    assert set(table["model"]) == {"dummy", "tfidf_lr"}
    by_model = table.set_index("model")
    assert by_model.loc["tfidf_lr", "f1_macro"] > by_model.loc["dummy", "f1_macro"]
    d = details["tfidf_lr"]
    assert d["cm"].shape == (len(d["labels"]), len(d["labels"]))
    assert "f1-score" in d["report"].columns


def test_topic_eval_data_skips_oos(eval_set):
    queries, qrels = eval_set
    texts, topics = topic_eval_data(queries, qrels, split=None)
    assert len(texts) == 4 and "рецепт плова" not in texts
    texts_test, _ = topic_eval_data(queries, qrels, split="test")
    assert texts_test == ["общага есть?", "сколько стоит обучение"]


def test_save_and_load_classifier(faq_df, tmp_path):
    model = make_classifiers(include_neural=False)["tfidf_lr"].fit(faq_df["question"], faq_df["topic"])
    path = save_classifier(model, tmp_path / "clf.joblib")
    restored = load_classifier(path)
    assert list(restored.predict(["общежитие"])) == list(model.predict(["общежитие"]))
