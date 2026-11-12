import json

import pytest

from src import config
from src.chatbot import ChatBot, chat_loop, format_response, load_bot_config, save_bot_config
from src.retrievers import build_retriever


@pytest.fixture()
def bot(faq_df):
    return ChatBot(build_retriever("tfidf_char"), faq_df, k=3)


def test_answer_returns_distinct_clusters(bot):
    resp = bot.answer("общага есть?")
    assert resp.answered
    assert resp.top.faq_id in {1, 2}
    clusters = [h.answer_cluster for h in resp.hits]
    assert len(clusters) == len(set(clusters)) == 3   # faq 1 и 2 — один ответ, второй раз не показан
    assert resp.message == resp.top.answer
    assert resp.latency_ms >= 0


def test_threshold_refusal_gives_contacts(faq_df):
    bot = ChatBot(build_retriever("tfidf_char"), faq_df, threshold=0.99)
    resp = bot.answer("рецепт плова")
    assert not resp.answered
    assert resp.message == config.FALLBACK_MESSAGE
    text = format_response(resp)
    assert "Похожие вопросы" in text and "1." in text


def test_hybrid_bot_refuses_by_component_confidence(faq_df):
    from src.retrievers import HybridRRFRetriever

    bot = ChatBot(HybridRRFRetriever(components=("tfidf_char", "bm25")), faq_df, threshold=1.0)
    ok, oos = bot.answer("Есть ли общежитие?"), bot.answer("рецепт плова")
    assert ok.answered and ok.confidence >= 1.0      # BM25 у хорошего запроса выше порога
    assert not oos.answered and oos.confidence < 1.0
    assert ok.top.score < 0.1                        # а сам RRF-score мал у обоих
    assert "уверенность" in format_response(oos)


def test_dated_answer_gets_note(faq_df):
    bot = ChatBot(build_retriever("tfidf_word_lemma"), faq_df)
    resp = bot.answer("Сколько стоит проживание в общежитии?")
    assert resp.top.faq_id == 3 and "устаревшим" in resp.message


def test_empty_query_raises(bot):
    with pytest.raises(ValueError):
        bot.answer("   ")


def test_to_dict_is_json_serializable(bot):
    payload = json.dumps(bot.answer("адрес университета").to_dict(), ensure_ascii=False)
    assert "faq_id" in payload


def test_on_answer_callback(faq_df):
    log = []
    bot = ChatBot(build_retriever("bm25"), faq_df, on_answer=log.append)
    bot.answer("военная кафедра")
    assert len(log) == 1 and log[0].query == "военная кафедра"


def test_chat_loop_scripted(bot):
    inputs = iter(["общага есть?", "", "выход"])
    out = []
    n = chat_loop(bot, input_fn=lambda _: next(inputs), print_fn=out.append)
    assert n == 1
    assert any("Ошибка" in line for line in out)      # пустой ввод не роняет цикл


def test_bot_config_roundtrip(tmp_path, faq_df):
    path = save_bot_config("bm25", threshold=1.5, metrics={"mrr@10": 0.7}, path=tmp_path / "bot.json")
    cfg = load_bot_config(path)
    assert cfg["method"] == "bm25" and cfg["threshold"] == 1.5
    bot = ChatBot.from_saved(faq_df, config_path=path)
    assert bot.retriever.name == "bm25" and bot.threshold == 1.5


def test_missing_config_falls_back_to_default(tmp_path):
    cfg = load_bot_config(tmp_path / "nope.json")
    assert cfg["method"] == "tfidf_char" and cfg["threshold"] is None


def test_broken_config_raises(tmp_path):
    path = tmp_path / "bot.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bot_config(path)
