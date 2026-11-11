"""Общий интерфейс всех методов поиска.

Каждый метод умеет одно главное: посчитать score запроса против всех
документов индекса (`score_matrix`). Из этого строятся `search`, оценка
leave-one-out (маскирование своего же вопроса) и гибрид RRF, поэтому
оценка и чат-бот работают с любым методом одинаково.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import pandas as pd

INDEX_FIELDS = ("question", "question_answer")


class BaseRetriever(ABC):
    """fit(faq_df) / search(query, k) -> [(faq_id, score)]."""

    name: str = "base"

    def __init__(self, index_field: str = "question") -> None:
        if index_field not in INDEX_FIELDS:
            raise ValueError(f"index_field должен быть одним из {INDEX_FIELDS}, получено {index_field!r}")
        self.index_field = index_field
        self.faq_ids: np.ndarray | None = None
        self.topics: np.ndarray | None = None

    # --- обучение -------------------------------------------------------
    def documents(self, faq_df: pd.DataFrame) -> list[str]:
        """Тексты, которые попадают в индекс: вопрос или вопрос + ответ."""
        questions = faq_df["question"].fillna("").astype(str)
        if self.index_field == "question":
            return questions.tolist()
        answers = faq_df["answer"].fillna("").astype(str)
        return (questions + " " + answers).tolist()

    def fit(self, faq_df: pd.DataFrame) -> "BaseRetriever":
        if len(faq_df) == 0:
            raise ValueError("Пустая база FAQ: нечего индексировать")
        self.faq_ids = faq_df["faq_id"].to_numpy()
        self.topics = faq_df["topic"].to_numpy() if "topic" in faq_df.columns else None
        self._fit(self.documents(faq_df), faq_df)
        return self

    @abstractmethod
    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        """Построить индекс по текстам документов."""

    # --- поиск ------------------------------------------------------------
    @abstractmethod
    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        """Матрица score (n_queries, n_docs); больше — релевантнее."""

    def score_matrix(self, queries: Sequence[str], mask: np.ndarray | None = None) -> np.ndarray:
        """Score всех документов для каждого запроса.

        mask: bool-матрица (n_queries, n_docs); True = документ исключить
        (так leave-one-out прячет сам вопрос и его точные дубли).
        """
        self._check_fitted()
        scores = np.asarray(self._raw_scores(list(queries)), dtype=np.float64)
        if mask is not None:
            scores = scores.copy()
            scores[mask] = -np.inf
        return scores

    def scores(self, query: str) -> np.ndarray:
        return self.score_matrix([query])[0]

    def search(self, query: str, k: int = 5) -> list[tuple[Any, float]]:
        """Топ-k документов: [(faq_id, score), ...] по убыванию score."""
        return self.rank(self.scores(query), k)

    def rank(self, scores: np.ndarray, k: int) -> list[tuple[Any, float]]:
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.faq_ids[i].item() if hasattr(self.faq_ids[i], "item") else self.faq_ids[i],
                 float(scores[i])) for i in top if np.isfinite(scores[i])]

    # --- описание и ресурсы ---------------------------------------------
    def params(self) -> dict:
        """Параметры метода для таблицы methods.params_json."""
        return {"index_field": self.index_field}

    def _index_state(self) -> Any:
        """Объекты, составляющие индекс (для замера размера)."""
        return None

    def index_size_mb(self) -> float:
        state = self._index_state()
        if state is None:
            return 0.0
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20

    def model_size_mb(self) -> float:
        """Размер весов модели (0 для методов без нейросети)."""
        return 0.0

    def _check_fitted(self) -> None:
        if self.faq_ids is None:
            raise RuntimeError(f"{self.name}: сначала вызовите fit(faq_df)")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"
