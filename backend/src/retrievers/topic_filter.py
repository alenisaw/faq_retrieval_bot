"""Поиск с фильтром по теме: классификатор сужает выдачу до предсказанной темы.

Фильтр мягкий: FAQ предсказанной темы поднимаются над остальными, но не
удаляются. Если классификатор ошибся, нужный ответ всё равно останется в
выдаче, только ниже. Фильтр включается, только если вероятность темы
не ниже min_proba.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseRetriever


class TopicFilteredRetriever(BaseRetriever):
    name = "topic_filtered"

    def __init__(self, base: BaseRetriever, classifier, min_proba: float = 0.5) -> None:
        super().__init__(base.index_field)
        self.base = base
        self.classifier = classifier  # любой объект с predict_proba и classes_
        self.min_proba = min_proba
        self.name = f"{base.name}+topic"

    def fit(self, faq_df: pd.DataFrame) -> "TopicFilteredRetriever":
        if "topic" not in faq_df:
            raise ValueError("Фильтру по теме нужна колонка topic в FAQ")
        self.base.fit(faq_df)
        return super().fit(faq_df)

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        pass

    def predict_topics(self, queries: Sequence[str]) -> list[str | None]:
        proba = self.classifier.predict_proba(list(queries))
        classes = np.asarray(self.classifier.classes_)
        best = proba.argmax(axis=1)
        return [classes[b] if proba[i, b] >= self.min_proba else None for i, b in enumerate(best)]

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        return self._filtered(queries, mask=None)[0]

    def score_matrix(self, queries: Sequence[str], mask: np.ndarray | None = None) -> np.ndarray:
        return self.score_with_confidence(queries, mask)[0]

    def score_with_confidence(self, queries: Sequence[str],
                              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Сдвиг по теме меняет только порядок; уверенность — у базового метода.
        self._check_fitted()
        scores, confidence = self._filtered(list(queries), mask)
        if mask is not None:
            scores[mask] = -np.inf
        return scores, confidence

    def _filtered(self, queries: Sequence[str], mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        scores, confidence = self.base.score_with_confidence(queries, mask)
        scores = scores.copy()
        for i, topic in enumerate(self.predict_topics(queries)):
            if topic is None:
                continue
            finite = np.isfinite(scores[i])
            if not finite.any():
                continue
            # сдвиг больше размаха score: FAQ темы гарантированно выше остальных
            shift = scores[i, finite].max() - scores[i, finite].min() + 1.0
            scores[i] += shift * (self.topics == topic)
        return scores, confidence

    def params(self) -> dict:
        return {"base": {"name": self.base.name, **self.base.params()},
                "min_proba": self.min_proba}

    def index_size_mb(self) -> float:
        return self.base.index_size_mb()

    def model_size_mb(self) -> float:
        return self.base.model_size_mb()
