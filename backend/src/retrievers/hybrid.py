"""Гибрид: Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum_i w_i / (k + rank_i(d)), где rank_i — место документа в
выдаче i-го метода. RRF смотрит только на места, а не на сами score, поэтому
не нужно приводить косинус TF-IDF и косинус эмбеддингов к одной шкале.

Обратная сторона: score RRF у топ-1 почти одинаковый и для хорошего запроса,
и для запроса не по теме, поэтому для порога отказа он не годится. Уверенность
гибрида — максимальный score одной из компонент (по умолчанию последней,
нейросетевой: её косинус сравним между запросами).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseRetriever, _row_max


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Места документов по убыванию score (1 = лучший), построчно."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


class HybridRRFRetriever(BaseRetriever):
    name = "hybrid_rrf"

    def __init__(self, components: Sequence[BaseRetriever | str] = ("tfidf_char", "e5_small"),
                 weights: Sequence[float] | None = None, k_rrf: int = 60,
                 confidence_component: int = -1) -> None:
        super().__init__("question")
        from . import build_retriever  # поздний импорт: реестр импортирует этот модуль

        self.components = [build_retriever(c) if isinstance(c, str) else c for c in components]
        if len(self.components) < 2:
            raise ValueError("Гибриду нужно минимум два метода")
        self.weights = list(weights) if weights is not None else [1.0] * len(self.components)
        if len(self.weights) != len(self.components):
            raise ValueError("Число весов должно совпадать с числом методов")
        self.k_rrf = k_rrf
        self.confidence_component = confidence_component
        self.components[confidence_component]  # IndexError сразу, а не при первом запросе

    def fit(self, faq_df: pd.DataFrame) -> "HybridRRFRetriever":
        for comp in self.components:
            comp.fit(faq_df)
        return super().fit(faq_df)

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        pass  # индексы уже построены компонентами

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        return self._fuse(queries, mask=None)[0]

    def score_matrix(self, queries: Sequence[str], mask: np.ndarray | None = None) -> np.ndarray:
        return self.score_with_confidence(queries, mask)[0]

    def score_with_confidence(self, queries: Sequence[str],
                              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Маску передаём внутрь: исключённый документ не должен занимать
        # место в выдаче компонент и сдвигать ранги остальных.
        self._check_fitted()
        fused, confidence = self._fuse(list(queries), mask)
        if mask is not None:
            fused[mask] = -np.inf
        return fused, confidence

    def _fuse(self, queries: Sequence[str], mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        fused = np.zeros((len(queries), len(self.faq_ids)))
        conf_idx = self.confidence_component % len(self.components)
        confidence = None
        for i, (w, comp) in enumerate(zip(self.weights, self.components)):
            comp_scores = comp.score_matrix(queries, mask)
            if i == conf_idx:
                confidence = _row_max(comp_scores)
            fused += w / (self.k_rrf + ranks_from_scores(comp_scores))
        return fused, confidence

    def params(self) -> dict:
        return {"components": [{"name": c.name, **c.params()} for c in self.components],
                "weights": self.weights, "k_rrf": self.k_rrf,
                "confidence_component": self.confidence_component}

    def index_size_mb(self) -> float:
        return sum(c.index_size_mb() for c in self.components)

    def model_size_mb(self) -> float:
        return sum(c.model_size_mb() for c in self.components)
