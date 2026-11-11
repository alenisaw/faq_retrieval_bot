"""Методы поиска по FAQ и реестр готовых конфигураций.

    from src.retrievers import build_retriever
    r = build_retriever("tfidf_char").fit(faq_df)
    r.search("общага есть?", k=3)   # [(faq_id, score), ...]

Номера методов совпадают с таблицей в плане проекта (0 — baseline).
"""

from __future__ import annotations

from .base import BaseRetriever
from .dense import E5_SMALL, MINILM, EmbeddingRetriever
from .hybrid import HybridRRFRetriever
from .rule_based import RuleBasedRetriever
from .sparse import BM25Retriever, BoWRetriever, TfidfCharRetriever, TfidfWordRetriever
from .topic_filter import TopicFilteredRetriever

# имя -> (класс, параметры по умолчанию)
METHODS: dict[str, tuple[type[BaseRetriever], dict]] = {
    "rule_based": (RuleBasedRetriever, {}),                                          # 0
    "bow": (BoWRetriever, {"mode": "lemma"}),                                        # 1
    "tfidf_word_lemma": (TfidfWordRetriever, {"mode": "lemma"}),                     # 2
    "tfidf_word_stem": (TfidfWordRetriever, {"mode": "stem"}),                       # 2
    "tfidf_char": (TfidfCharRetriever, {}),                                          # 3
    "e5_small": (EmbeddingRetriever, {"model_name": E5_SMALL}),                      # 4
    "minilm": (EmbeddingRetriever, {"model_name": MINILM}),                          # 4
    "hybrid_rrf": (HybridRRFRetriever, {"components": ("tfidf_char", "e5_small")}),  # 5
    "bm25": (BM25Retriever, {"mode": "lemma"}),                                      # доп.
}

# Методы без нейросетей: строятся за секунды, не требуют скачивания моделей.
LEXICAL_METHODS = ("rule_based", "bow", "tfidf_word_lemma", "tfidf_word_stem", "tfidf_char", "bm25")
NEURAL_METHODS = ("e5_small", "minilm", "hybrid_rrf")


def build_retriever(name: str, **overrides) -> BaseRetriever:
    """Создать метод по имени из METHODS; overrides меняют параметры по умолчанию."""
    if name not in METHODS:
        raise KeyError(f"Неизвестный метод {name!r}. Доступны: {', '.join(METHODS)}")
    cls, defaults = METHODS[name]
    retriever = cls(**{**defaults, **overrides})
    retriever.name = name  # в таблицах метрик — имя конфигурации, а не класса
    return retriever


__all__ = [
    "BaseRetriever", "BoWRetriever", "BM25Retriever", "EmbeddingRetriever",
    "HybridRRFRetriever", "RuleBasedRetriever", "TfidfCharRetriever", "TfidfWordRetriever",
    "TopicFilteredRetriever",
    "METHODS", "LEXICAL_METHODS", "NEURAL_METHODS", "build_retriever", "E5_SMALL", "MINILM",
]
