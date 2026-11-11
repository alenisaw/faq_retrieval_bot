"""Плотный поиск на sentence embeddings (sentence-transformers).

Основная модель: intfloat/multilingual-e5-small (118M параметров, 384-мерные
векторы). Для сравнения: paraphrase-multilingual-MiniLM-L12-v2.

Модели семейства E5 обучены с префиксами. Сопоставление вопроса пользователя
с вопросом FAQ — симметричная задача (поиск перефраза), поэтому по
рекомендации авторов E5 обе стороны получают "query: ". Если в индекс идёт
вопрос + ответ, задача асимметричная и документы получают "passage: ".
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseRetriever

E5_SMALL = "intfloat/multilingual-e5-small"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Одна загруженная модель на процесс: e5 в отдельном методе и e5 внутри
# гибрида не должны грузить веса дважды.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_model(model_name: str, device: str | None = None):
    device = device or "cpu"
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def _clean(text: str) -> str:
    return " ".join(str(text).replace("\\n", " ").split())


class EmbeddingRetriever(BaseRetriever):
    """Косинус между нормированными эмбеддингами запроса и документов."""

    name = "embedding"

    def __init__(self, model_name: str = E5_SMALL, index_field: str = "question",
                 batch_size: int = 64, device: str | None = None,
                 cache_dir: str | Path | None = None) -> None:
        super().__init__(index_field)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.model_load_s = 0.0
        self.encode_s = 0.0

    # --- префиксы E5 --------------------------------------------------------
    @property
    def is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    @property
    def query_prefix(self) -> str:
        return "query: " if self.is_e5 else ""

    @property
    def doc_prefix(self) -> str:
        if not self.is_e5:
            return ""
        return "query: " if self.index_field == "question" else "passage: "

    # --- модель -------------------------------------------------------------
    @property
    def model(self):
        start = time.perf_counter()
        model = load_model(self.model_name, self.device)
        self.model_load_s += time.perf_counter() - start
        return model

    def encode(self, texts: Sequence[str], prefix: str = "") -> np.ndarray:
        """Нормированные эмбеддинги float32, shape (n, dim)."""
        texts = [prefix + _clean(t) for t in texts]
        emb = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        # Без кэша запросов: иначе повторный прогон занижал бы замер задержки.
        return self.encode(queries, self.query_prefix)

    # --- индекс -------------------------------------------------------------
    def _cache_path(self, docs: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(
            "\n".join([self.model_name, self.doc_prefix, *docs]).encode("utf-8")
        ).hexdigest()[:16]
        slug = self.model_name.split("/")[-1]
        return self.cache_dir / f"emb_{slug}_{digest}.npy"

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        self.model_load_s = 0.0
        path = self._cache_path(docs)
        if path is not None and path.exists():
            self.doc_embeddings = np.load(path)
            self.encode_s = 0.0
            return
        start = time.perf_counter()
        self.doc_embeddings = self.encode(docs, self.doc_prefix)
        self.encode_s = time.perf_counter() - start - self.model_load_s
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.doc_embeddings)

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        return self.encode_queries(queries) @ self.doc_embeddings.T

    # --- описание -----------------------------------------------------------
    def params(self) -> dict:
        return {**super().params(), "model_name": self.model_name,
                "query_prefix": self.query_prefix, "doc_prefix": self.doc_prefix}

    def _index_state(self):
        return self.doc_embeddings

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def model_size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 2**20
