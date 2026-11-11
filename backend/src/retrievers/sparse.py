"""Разреженные (лексические) методы: BoW, TF-IDF по словам и символам, BM25."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from ..preprocessing import CharPreprocessor, TextAnalyzer, preprocess
from .base import BaseRetriever


class _VectorizerRetriever(BaseRetriever):
    """Косинус между L2-нормированными векторами запроса и документов."""

    def _make_vectorizer(self):
        raise NotImplementedError

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        self.vectorizer = self._make_vectorizer()
        self.doc_matrix = l2_normalize(self.vectorizer.fit_transform(docs))

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        q = l2_normalize(self.vectorizer.transform(queries))
        return (q @ self.doc_matrix.T).toarray()

    def _index_state(self):
        m = self.doc_matrix
        return (m.data, m.indices, m.indptr, self.vectorizer.vocabulary_)

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


class BoWRetriever(_VectorizerRetriever):
    """Мешок слов (частоты лемм) + косинус."""

    name = "bow"

    def __init__(self, mode: str = "lemma", remove_stopwords: bool = True,
                 keep_question_words: bool = False, index_field: str = "question") -> None:
        super().__init__(index_field)
        self.mode = mode
        self.remove_stopwords = remove_stopwords
        self.keep_question_words = keep_question_words

    def _make_vectorizer(self):
        return CountVectorizer(analyzer=TextAnalyzer(self.mode, self.remove_stopwords,
                                                     self.keep_question_words))

    def params(self) -> dict:
        return {**super().params(), "mode": self.mode, "remove_stopwords": self.remove_stopwords,
                "keep_question_words": self.keep_question_words}


class TfidfWordRetriever(_VectorizerRetriever):
    """TF-IDF по словам (1–2-граммы) после стемминга или лемматизации."""

    name = "tfidf_word"

    def __init__(self, mode: str = "lemma", ngram_range: tuple[int, int] = (1, 2),
                 remove_stopwords: bool = True, keep_question_words: bool = False,
                 sublinear_tf: bool = True, index_field: str = "question") -> None:
        super().__init__(index_field)
        self.mode = mode
        self.ngram_range = tuple(ngram_range)
        self.remove_stopwords = remove_stopwords
        self.keep_question_words = keep_question_words
        self.sublinear_tf = sublinear_tf

    def _make_vectorizer(self):
        analyzer = TextAnalyzer(self.mode, self.remove_stopwords, self.keep_question_words,
                                self.ngram_range)
        return TfidfVectorizer(analyzer=analyzer, sublinear_tf=self.sublinear_tf)

    def params(self) -> dict:
        return {**super().params(), "mode": self.mode, "ngram_range": list(self.ngram_range),
                "remove_stopwords": self.remove_stopwords,
                "keep_question_words": self.keep_question_words, "sublinear_tf": self.sublinear_tf}

    def top_terms(self, n: int = 10) -> pd.DataFrame:
        """Топ-n термов каждого FAQ по TF-IDF -> (faq_id, term, tfidf_score).

        Формат таблицы `keywords` в faq.db.
        """
        self._check_fitted()
        terms = np.array(self.vectorizer.get_feature_names_out())
        rows = []
        m = self.doc_matrix.tocsr()
        for i, faq_id in enumerate(self.faq_ids):
            start, end = m.indptr[i], m.indptr[i + 1]
            if start == end:
                continue
            idx, vals = m.indices[start:end], m.data[start:end]
            order = np.argsort(-vals)[:n]
            rows.extend({"faq_id": faq_id, "term": terms[idx[j]], "tfidf_score": float(vals[j])}
                        for j in order)
        return pd.DataFrame(rows, columns=["faq_id", "term", "tfidf_score"])


class TfidfCharRetriever(_VectorizerRetriever):
    """TF-IDF по символьным n-граммам внутри слов (char_wb).

    Не требует стемминга: общие подстроки ловят и морфологию ("общежитие" /
    "общежития"), и опечатки ("общага" частично совпадает с "общежитие").
    """

    name = "tfidf_char"

    def __init__(self, ngram_range: tuple[int, int] = (2, 5), remove_stopwords: bool = False,
                 keep_question_words: bool = False, sublinear_tf: bool = True,
                 index_field: str = "question") -> None:
        super().__init__(index_field)
        self.ngram_range = tuple(ngram_range)
        self.remove_stopwords = remove_stopwords
        self.keep_question_words = keep_question_words
        self.sublinear_tf = sublinear_tf

    def _make_vectorizer(self):
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            preprocessor=CharPreprocessor(self.remove_stopwords, self.keep_question_words),
            sublinear_tf=self.sublinear_tf,
        )

    def params(self) -> dict:
        return {**super().params(), "ngram_range": list(self.ngram_range),
                "remove_stopwords": self.remove_stopwords,
                "keep_question_words": self.keep_question_words, "sublinear_tf": self.sublinear_tf}


class BM25Retriever(BaseRetriever):
    """Okapi BM25 по леммам (rank_bm25)."""

    name = "bm25"

    def __init__(self, mode: str = "lemma", k1: float = 1.5, b: float = 0.75,
                 remove_stopwords: bool = True, keep_question_words: bool = False,
                 index_field: str = "question") -> None:
        super().__init__(index_field)
        self.mode = mode
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.keep_question_words = keep_question_words

    def _tokens(self, text: str) -> list[str]:
        return preprocess(text, mode=self.mode, remove_stopwords=self.remove_stopwords,
                          keep_question_words=self.keep_question_words)

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        from rank_bm25 import BM25Okapi

        corpus = [self._tokens(d) or ["<empty>"] for d in docs]  # BM25Okapi падает на пустых
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        return np.vstack([self.bm25.get_scores(self._tokens(q)) for q in queries])

    def _index_state(self):
        return self.bm25

    def params(self) -> dict:
        return {**super().params(), "mode": self.mode, "k1": self.k1, "b": self.b,
                "remove_stopwords": self.remove_stopwords,
                "keep_question_words": self.keep_question_words}
