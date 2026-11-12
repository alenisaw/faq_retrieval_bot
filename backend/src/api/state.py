"""Состояние приложения: база FAQ и обученные боты по методам.

Индекс каждого метода строится один раз, при первом запросе к нему
(лексические — доли секунды, e5 — скачивание модели при первом запуске),
дальше бот берётся из кэша. Блокировка на метод не даёт двум запросам
обучать один и тот же индекс параллельно.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from .. import config, db
from ..chatbot import ChatBot, ChatResponse, load_bot_config
from ..evaluation import load_faq
from ..retrievers import LEXICAL_METHODS, METHODS, build_retriever

METHOD_DESCRIPTIONS = {
    "rule_based": "baseline: ключевые слова -> тема, Jaccard по леммам внутри темы",
    "bow": "мешок слов (леммы) + косинус",
    "tfidf_word_lemma": "TF-IDF по словам, леммы, 1–2-граммы",
    "tfidf_word_stem": "TF-IDF по словам, стеммы, 1–2-граммы",
    "tfidf_char": "TF-IDF по символьным n-граммам 2–5, устойчив к опечаткам",
    "e5_small": "эмбеддинги multilingual-e5-small, поиск по смыслу",
    "minilm": "эмбеддинги paraphrase-multilingual-MiniLM-L12-v2",
    "hybrid_rrf": "Reciprocal Rank Fusion: tfidf_char + e5_small",
    "bm25": "Okapi BM25 по леммам",
}


class AppState:
    def __init__(self, faq_df: pd.DataFrame | None = None, db_path: str | Path | None = None,
                 bot_config_path: str | Path | None = None, log_chats: bool = True) -> None:
        self.faq = faq_df if faq_df is not None else load_faq()
        self.db_path = Path(db_path or config.DB_PATH)
        self.bot_config = load_bot_config(bot_config_path)
        self.default_method: str = self.bot_config["method"]
        self.log_chats = log_chats
        self._bots: dict[str, ChatBot] = {}
        self._locks: dict[str, threading.Lock] = {m: threading.Lock() for m in METHODS}
        self._faq_by_id = self.faq.set_index("faq_id", drop=False)

    # --- боты ---------------------------------------------------------------
    @property
    def loaded_methods(self) -> list[str]:
        return list(self._bots)

    def bot(self, method: str | None = None) -> ChatBot:
        method = method or self.default_method
        if method not in METHODS:
            raise KeyError(method)
        if method in self._bots:
            return self._bots[method]
        with self._locks[method]:
            if method not in self._bots:   # пока ждали блокировку, мог построить другой поток
                if method == self.default_method:
                    retriever = build_retriever(method, **self.bot_config.get("params", {}))
                    threshold = self.bot_config.get("threshold")
                else:
                    retriever, threshold = build_retriever(method), None
                self._bots[method] = ChatBot(retriever, self.faq, threshold=threshold)
        return self._bots[method]

    def answer(self, query: str, method: str | None = None, k: int = 3, log: bool = True) -> ChatResponse:
        bot = self.bot(method)
        resp = bot.answer(query, k=k)
        if log and self.log_chats and self.db_path.exists():
            try:
                with db.connect(self.db_path) as conn:
                    db.log_chat(conn, resp, bot.retriever.params(), source="api")
            except Exception as exc:  # журнал не должен ломать ответ пользователю
                print(f"[chat_log] не записано: {exc}")
        return resp

    # --- справочники ------------------------------------------------------------
    def methods(self) -> list[dict]:
        return [{"name": m, "family": "lexical" if m in LEXICAL_METHODS else "neural",
                 "is_default": m == self.default_method, "loaded": m in self._bots,
                 "description": METHOD_DESCRIPTIONS.get(m, "")} for m in METHODS]

    def faq_row(self, faq_id: int) -> dict | None:
        if faq_id not in self._faq_by_id.index:
            return None
        return self._faq_by_id.loc[faq_id].to_dict()

    def database_ready(self) -> bool:
        return self.db_path.exists()
