"""Классификатор темы (интента) вопроса.

Обучение — на вопросах FAQ (колонка topic), тест — на ручных запросах
абитуриентов из eval/: это разные тексты, утечки нет.

Модели (плюс baseline):
    dummy     DummyClassifier(most_frequent)
    tfidf_lr  TF-IDF (леммы 1–2 + символы 2–5) + LogisticRegression
    e5_lr     эмбеддинги multilingual-e5-small + LogisticRegression

Классов мало и они несбалансированы (творческий экзамен — единицы
вопросов), поэтому class_weight="balanced" и F1 по каждой теме в отчёте.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config
from .preprocessing import CharPreprocessor, TextAnalyzer
from .retrievers.dense import E5_SMALL, load_model


# (модель, текст) -> эмбеддинг. На уровне модуля, а не объекта: GridSearchCV
# клонирует пайплайн на каждый фолд, и кэш внутри объекта терялся бы.
_EMBEDDING_MEMO: dict[tuple[str, str], np.ndarray] = {}


class EmbeddingFeaturizer(BaseEstimator, TransformerMixin):
    """Тексты -> нормированные эмбеддинги sentence-transformers.

    Веса модели в pickle не попадают (берутся из общего кэша моделей),
    сохраняется только имя модели.
    """

    def __init__(self, model_name: str = E5_SMALL, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        texts = [str(t) for t in X]
        missing = [t for t in dict.fromkeys(texts) if (self.model_name, t) not in _EMBEDDING_MEMO]
        if missing:
            prefix = "query: " if "e5" in self.model_name.lower() else ""
            emb = load_model(self.model_name).encode(
                [prefix + t for t in missing], batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            _EMBEDDING_MEMO.update(((self.model_name, t), e) for t, e in zip(missing, emb))
        return np.vstack([_EMBEDDING_MEMO[(self.model_name, t)] for t in texts])


def make_classifiers(include_neural: bool = True) -> dict[str, Pipeline]:
    lr = dict(max_iter=3000, class_weight="balanced")
    models = {
        "dummy": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "tfidf_lr": Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(analyzer=TextAnalyzer("lemma", ngram_range=(1, 2)),
                                         sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         preprocessor=CharPreprocessor(), sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=10.0, **lr)),
        ]),
    }
    if include_neural:
        models["e5_lr"] = Pipeline([("emb", EmbeddingFeaturizer()),
                                    ("clf", LogisticRegression(C=10.0, **lr))])
    return models


def tune(model: Pipeline, X: Sequence[str], y: Sequence[str],
         grid: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0)) -> Pipeline:
    """Подбор C логистической регрессии кросс-валидацией по F1-macro на FAQ."""
    if "clf" not in model.named_steps or not isinstance(model.named_steps["clf"], LogisticRegression):
        return model.fit(X, y)
    n_splits = int(min(3, pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return model.fit(X, y)
    search = GridSearchCV(model, {"clf__C": list(grid)}, scoring="f1_macro",
                          cv=StratifiedKFold(n_splits, shuffle=True, random_state=config.RANDOM_SEED))
    search.fit(list(X), list(y))
    return search.best_estimator_


def topic_eval_data(queries: pd.DataFrame, qrels: pd.DataFrame,
                    split: str | None = "test") -> tuple[list[str], list[str]]:
    """Ручные запросы по теме -> (тексты, gold-темы)."""
    q = queries[~queries["is_oos"].astype(bool)]
    if split is not None:
        q = q[q["split"] == split]
    topic = qrels.drop_duplicates("query_id").set_index("query_id")["topic"]
    q = q.assign(topic=q["query_id"].map(topic)).dropna(subset=["topic"])
    return q["text"].tolist(), q["topic"].tolist()


def evaluate_classifiers(faq_df: pd.DataFrame, test_texts: Sequence[str],
                         test_topics: Sequence[str], models: dict[str, Pipeline] | None = None,
                         do_tune: bool = False) -> tuple[pd.DataFrame, dict]:
    """Обучить на вопросах FAQ, проверить на ручных запросах.

    Returns:
        таблица (model, accuracy, f1_macro, f1_weighted, train_time_s, predict_ms),
        details[name] = {"model", "y_pred", "report" (F1 по темам), "cm", "labels"}.
    """
    if "topic" not in faq_df:
        raise ValueError("В FAQ нет колонки topic")
    train = faq_df.dropna(subset=["topic"])
    X_train, y_train = train["question"].tolist(), train["topic"].tolist()
    labels = [t for t in config.TOPICS if t in set(y_train) | set(test_topics)]
    labels += sorted((set(y_train) | set(test_topics)) - set(labels))

    rows, details = [], {}
    for name, model in (models or make_classifiers()).items():
        start = time.perf_counter()
        model = tune(model, X_train, y_train) if do_tune else model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(list(test_texts))
        predict_ms = 1000 * (time.perf_counter() - start) / max(len(test_texts), 1)

        report = pd.DataFrame(classification_report(
            test_topics, y_pred, labels=labels, output_dict=True, zero_division=0)).T
        rows.append({
            "model": name,
            "accuracy": accuracy_score(test_topics, y_pred),
            "f1_macro": f1_score(test_topics, y_pred, labels=labels, average="macro", zero_division=0),
            "f1_weighted": f1_score(test_topics, y_pred, labels=labels, average="weighted",
                                    zero_division=0),
            "train_time_s": train_time,
            "predict_ms": predict_ms,
        })
        details[name] = {"model": model, "y_pred": list(y_pred), "report": report,
                         "cm": confusion_matrix(test_topics, y_pred, labels=labels),
                         "labels": labels}
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False, kind="stable")
    return table.reset_index(drop=True), details


CLASSIFIER_NAME = "topic_classifier.joblib"  # в config.MODELS_DIR


def save_classifier(model: Pipeline, path: str | Path | None = None) -> Path:
    path = Path(path or config.MODELS_DIR / CLASSIFIER_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_classifier(path: str | Path | None = None) -> Pipeline:
    return joblib.load(path or config.MODELS_DIR / CLASSIFIER_NAME)
