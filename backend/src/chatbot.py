"""Чат-бот: вопрос абитуриента -> топ-k разных ответов из FAQ со score.

    bot = ChatBot.from_saved()          # метод и порог из models/chatbot.json
    resp = bot.answer("общага есть?")
    print(format_response(resp))

Если score топ-1 ниже порога, бот не угадывает, а отказывает и даёт
контакты приёмной комиссии. Метод и порог выбираются по результатам
оценки (лучший MRR на тесте, порог на dev) и сохраняются save_bot_config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from . import config
from .evaluation import TOP_N, load_faq, validate_faq
from .retrievers import BaseRetriever, TopicFilteredRetriever, build_retriever

BOT_CONFIG_NAME = "chatbot.json"  # в config.MODELS_DIR
DEFAULT_METHOD = "tfidf_char"  # без нейросети: работает сразу, без скачивания модели
DATED_NOTE = "Внимание: ответ может быть устаревшим, проверьте на admission.astanait.edu.kz"
EXIT_WORDS = frozenset({"0", "выход", "exit", "quit", "q"})


@dataclass
class Hit:
    faq_id: object
    question: str
    answer: str
    score: float
    answer_cluster: object = None
    topic: str | None = None
    is_dated: bool = False


@dataclass
class ChatResponse:
    query: str
    answered: bool
    message: str
    hits: list[Hit] = field(default_factory=list)
    method: str = ""
    threshold: float | None = None
    predicted_topic: str | None = None
    latency_ms: float = 0.0

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None

    def to_dict(self) -> dict:
        """JSON-совместимый словарь (для API)."""
        d = asdict(self)
        for h in d["hits"]:
            h["faq_id"] = _plain(h["faq_id"])
            h["answer_cluster"] = _plain(h["answer_cluster"])
            h["is_dated"] = bool(h["is_dated"])
        return d


def _plain(value):
    return value.item() if hasattr(value, "item") else value


class ChatBot:
    def __init__(self, retriever: BaseRetriever, faq_df: pd.DataFrame,
                 threshold: float | None = None, k: int = 3, fitted: bool = False,
                 topic_classifier=None, min_topic_proba: float = 0.5,
                 on_answer: Callable[[ChatResponse], None] | None = None) -> None:
        """
        Args:
            threshold: порог score топ-1; None — отвечать всегда.
            fitted: retriever уже обучен на этом faq_df.
            topic_classifier: если задан, выдача поднимает FAQ предсказанной темы.
            on_answer: колбэк после каждого ответа (например, запись в chat_log).
        """
        self.faq = validate_faq(faq_df).set_index("faq_id", drop=False)
        if topic_classifier is not None:
            retriever = TopicFilteredRetriever(retriever, topic_classifier, min_topic_proba)
            fitted = False
        self.retriever = retriever if fitted else retriever.fit(faq_df)
        self.threshold = threshold
        self.k = k
        self.on_answer = on_answer

    # --- создание ---------------------------------------------------------
    @classmethod
    def from_saved(cls, faq_df: pd.DataFrame | None = None,
                   config_path: str | Path | None = None, **kwargs) -> "ChatBot":
        """Метод и порог из models/chatbot.json; если файла нет — DEFAULT_METHOD без порога."""
        cfg = load_bot_config(config_path)
        faq_df = load_faq() if faq_df is None else faq_df
        retriever = build_retriever(cfg["method"], **cfg.get("params", {}))
        return cls(retriever, faq_df, threshold=cfg.get("threshold"), **kwargs)

    # --- ответ --------------------------------------------------------------
    def _hit(self, faq_id, score: float) -> Hit:
        row = self.faq.loc[faq_id]
        return Hit(faq_id=faq_id, question=row["question"], answer=row["answer"], score=score,
                   answer_cluster=row["answer_cluster"], topic=row.get("topic"),
                   is_dated=bool(row.get("is_dated", False)))

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Топ-k FAQ с разными answer_cluster (без повторов одного ответа)."""
        k = k or self.k
        hits, seen = [], set()
        for faq_id, score in self.retriever.search(query, k=TOP_N):
            hit = self._hit(faq_id, score)
            if hit.answer_cluster in seen:
                continue
            seen.add(hit.answer_cluster)
            hits.append(hit)
            if len(hits) == k:
                break
        return hits

    def answer(self, query: str, k: int | None = None) -> ChatResponse:
        query = (query or "").strip()
        if not query:
            raise ValueError("Пустой вопрос")
        start = time.perf_counter()
        hits = self.search(query, k)
        answered = bool(hits) and (self.threshold is None or hits[0].score >= self.threshold)
        if answered:
            message = hits[0].answer
            if hits[0].is_dated:
                message += f"\n({DATED_NOTE})"
        else:
            message = config.FALLBACK_MESSAGE
        topic = None
        if isinstance(self.retriever, TopicFilteredRetriever):
            topic = self.retriever.predict_topics([query])[0]
        resp = ChatResponse(query=query, answered=answered, message=message, hits=hits,
                            method=self.retriever.name, threshold=self.threshold,
                            predicted_topic=topic,
                            latency_ms=1000 * (time.perf_counter() - start))
        if self.on_answer is not None:
            self.on_answer(resp)
        return resp


# --- конфиг бота -----------------------------------------------------------

def save_bot_config(method: str, threshold: float | None = None, params: dict | None = None,
                    metrics: dict | None = None, path: str | Path | None = None) -> Path:
    """Сохранить выбранный по результатам оценки метод и порог."""
    path = Path(path or config.MODELS_DIR / BOT_CONFIG_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": method, "threshold": threshold, "params": params or {},
               "metrics": metrics or {}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path


def load_bot_config(path: str | Path | None = None) -> dict:
    path = Path(path or config.MODELS_DIR / BOT_CONFIG_NAME)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"method": DEFAULT_METHOD, "threshold": None, "params": {}}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Повреждён файл {path}: {exc}") from exc
    if "method" not in cfg:
        raise ValueError(f"В {path} нет поля 'method'")
    return cfg


# --- консоль ----------------------------------------------------------------

def format_response(resp: ChatResponse, show_alternatives: bool = True) -> str:
    lines = []
    if resp.answered:
        top = resp.top
        lines.append(f"Ответ (score {top.score:.3f}): {resp.message}")
        lines.append(f"  Похожий вопрос из FAQ: {top.question}")
    else:
        score = f" (лучший score {resp.top.score:.3f} < порога {resp.threshold:.3f})" if resp.top else ""
        lines.append(f"{resp.message}{score}")
    # при ответе топ-1 уже показан — подсказки начинаются со второго
    suggestions = resp.hits[1:] if resp.answered else resp.hits
    if show_alternatives and suggestions:
        lines.append("  Возможно, вы имели в виду:" if resp.answered else "  Похожие вопросы из FAQ:")
        for i, h in enumerate(suggestions, start=1):
            lines.append(f"    {i}. {h.question} (score {h.score:.3f})")
    if resp.predicted_topic:
        lines.append(f"  Тема: {config.TOPICS.get(resp.predicted_topic, resp.predicted_topic)}")
    lines.append(f"  [{resp.method}, {resp.latency_ms:.0f} мс]")
    return "\n".join(lines)


def chat_loop(bot: ChatBot, input_fn: Callable[[str], str] = input,
              print_fn: Callable[[str], None] = print) -> int:
    """Диалог в консоли до слова выхода. Возвращает число отвеченных вопросов."""
    print_fn(f"Задайте вопрос о поступлении в AITU (выход: {', '.join(sorted(EXIT_WORDS))})")
    n = 0
    while True:
        try:
            query = input_fn("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in EXIT_WORDS:
            break
        try:
            resp = bot.answer(query)
        except ValueError as exc:
            print_fn(f"Ошибка: {exc}")
            continue
        n += resp.answered
        print_fn(format_response(resp))
    return n


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Чат-бот по FAQ AITU")
    parser.add_argument("--method", help="метод из реестра; по умолчанию из models/chatbot.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--faq", default=None, help="по умолчанию data/processed/faq_clean.parquet")
    parser.add_argument("--ask", help="задать один вопрос и выйти")
    args = parser.parse_args(argv)

    faq_df = load_faq(args.faq)
    if args.method:
        bot = ChatBot(build_retriever(args.method), faq_df, threshold=args.threshold)
    else:
        bot = ChatBot.from_saved(faq_df)
        if args.threshold is not None:
            bot.threshold = args.threshold
    if args.ask:
        print(format_response(bot.answer(args.ask)))
    else:
        chat_loop(bot)


if __name__ == "__main__":
    main()
