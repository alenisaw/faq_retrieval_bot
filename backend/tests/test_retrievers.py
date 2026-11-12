import numpy as np
import pytest

from src.retrievers import (
    LEXICAL_METHODS,
    METHODS,
    HybridRRFRetriever,
    RuleBasedRetriever,
    TopicFilteredRetriever,
    build_retriever,
)
from src.retrievers.dense import E5_SMALL
from src.retrievers.hybrid import ranks_from_scores


@pytest.mark.parametrize("name", LEXICAL_METHODS)
def test_lexical_methods_find_dormitory(name, faq_df):
    r = build_retriever(name).fit(faq_df)
    hits = r.search("Есть ли общежитие для студентов?", k=3)
    assert len(hits) == 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] in {1, 2}
    assert isinstance(hits[0][0], int) and isinstance(hits[0][1], float)
    assert r.index_size_mb() > 0
    assert r.name == name


@pytest.mark.parametrize("name", LEXICAL_METHODS)
def test_mask_excludes_documents(name, faq_df):
    r = build_retriever(name).fit(faq_df)
    mask = np.zeros((1, len(faq_df)), dtype=bool)
    mask[0, :2] = True  # прячем оба FAQ про наличие общежития
    scores = r.score_matrix(["Есть ли у вас общежитие?"], mask)[0]
    assert np.isneginf(scores[:2]).all()
    found = [fid for fid, _ in r.rank(scores, k=len(faq_df))]
    assert 1 not in found and 2 not in found


def test_question_answer_index_field(faq_df):
    r = build_retriever("tfidf_word_lemma", index_field="question_answer").fit(faq_df)
    # слова "Мангилик Ел" есть только в ответе
    assert r.search("Мангилик Ел", k=1)[0][0] in {10, 11}
    with pytest.raises(ValueError):
        build_retriever("bow", index_field="answer")


def test_search_before_fit_raises():
    with pytest.raises(RuntimeError):
        build_retriever("bow").search("общежитие")


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        build_retriever("gpt")


def test_registry_covers_plan_methods():
    for name in ("rule_based", "bow", "tfidf_word_lemma", "tfidf_word_stem", "tfidf_char",
                 "e5_small", "minilm", "hybrid_rrf"):
        assert name in METHODS


def test_rule_based_topic_routing(faq_df):
    r = RuleBasedRetriever().fit(faq_df)
    assert r.predict_topic("сколько стоит общага") == "dormitory"  # "общага" весит больше "стоить"
    assert r.predict_topic("есть ли военная кафедра") == "military"
    assert r.predict_topic("абракадабра") is None
    top_id = r.search("какой проходной балл", k=1)[0][0]
    assert faq_df.set_index("faq_id").loc[top_id, "topic"] == "scores_grants"


def test_ranks_from_scores():
    scores = np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]])
    assert ranks_from_scores(scores).tolist() == [[3, 1, 2], [1, 3, 2]]


def test_hybrid_rrf_of_two_lexical_methods(faq_df):
    r = HybridRRFRetriever(components=("tfidf_char", "bm25")).fit(faq_df)
    assert r.search("общага есть?", k=1)[0][0] in {1, 2}
    params = r.params()
    assert [c["name"] for c in params["components"]] == ["tfidf_char", "bm25"]
    with pytest.raises(ValueError):
        HybridRRFRetriever(components=("bm25",))


def test_confidence_default_is_top_score(faq_df):
    r = build_retriever("tfidf_char").fit(faq_df)
    scores, conf = r.score_with_confidence(["общага есть?", "рецепт плова"])
    assert np.allclose(conf, scores.max(axis=1))


def test_hybrid_confidence_comes_from_component_not_rrf(faq_df):
    r = HybridRRFRetriever(components=("tfidf_char", "bm25")).fit(faq_df)
    queries = ["Есть ли общежитие?", "рецепт плова"]
    fused, conf = r.score_with_confidence(queries)
    bm25 = r.components[-1].score_matrix(queries)
    assert np.allclose(conf, bm25.max(axis=1))
    # RRF-score топ-1 почти не зависит от того, по теме запрос или нет —
    # поэтому порог по нему бесполезен, а по уверенности компоненты — нет
    assert abs(fused[0].max() - fused[1].max()) < 0.01
    assert conf[0] > conf[1]
    with pytest.raises(IndexError):
        HybridRRFRetriever(components=("tfidf_char", "bm25"), confidence_component=5)


class _StubClassifier:
    """Всегда предсказывает 'military' с вероятностью 0.9."""
    classes_ = np.array(["dormitory", "military"])

    def predict_proba(self, texts):
        return np.tile([0.1, 0.9], (len(texts), 1))


def test_topic_filter_lifts_predicted_topic(faq_df):
    r = TopicFilteredRetriever(build_retriever("tfidf_char"), _StubClassifier()).fit(faq_df)
    hits = r.search("общежитие", k=len(faq_df))
    topics = faq_df.set_index("faq_id")["topic"]
    assert topics[hits[0][0]] == "military"           # тема классификатора всегда сверху
    assert {fid for fid, _ in hits} >= {1, 2}         # но другие FAQ не удалены
    strict = TopicFilteredRetriever(build_retriever("tfidf_char"), _StubClassifier(), min_proba=0.95)
    assert strict.fit(faq_df).search("общежитие", k=1)[0][0] in {1, 2}  # низкая уверенность -> без фильтра
    # уверенность — у базового метода, сдвиг по теме её не раздувает
    _, conf = r.score_with_confidence(["общежитие"])
    assert np.isclose(conf[0], r.base.scores("общежитие").max())


def test_tfidf_top_terms_format(faq_df):
    r = build_retriever("tfidf_word_lemma").fit(faq_df)
    terms = r.top_terms(n=3)
    assert list(terms.columns) == ["faq_id", "term", "tfidf_score"]
    assert terms.groupby("faq_id").size().max() <= 3


def _model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except ImportError:
        return False


@pytest.mark.neural
@pytest.mark.skipif(not _model_cached(E5_SMALL), reason="модель e5-small не скачана")
def test_e5_paraphrase_search(faq_df, tmp_path):
    r = build_retriever("e5_small", cache_dir=tmp_path).fit(faq_df)
    assert r.search("Можно ли жить в кампусе, есть общага?", k=1)[0][0] in {1, 2}
    assert r.doc_prefix == "query: "
    assert list(tmp_path.glob("emb_*.npy"))           # эмбеддинги документов закэшированы
    r2 = build_retriever("e5_small", cache_dir=tmp_path).fit(faq_df)
    assert r2.encode_s == 0.0                          # второй fit берёт кэш
    assert r.model_size_mb() > 100
