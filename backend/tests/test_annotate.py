import pandas as pd
import pytest

from src import config
from src.annotate import (
    UNASSIGNED,
    add_query,
    candidate_pool,
    check,
    freeze,
    import_drafts,
    interactive,
    read_eval_files,
    review,
    search_faq,
)
from src.evaluation import annotator_agreement, load_eval_set
from src.retrievers import build_retriever


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(config, "TEST_QUERIES_PATH", tmp_path / "eval" / "test_queries.csv")
    monkeypatch.setattr(config, "QRELS_PATH", tmp_path / "eval" / "qrels.csv")
    return tmp_path


@pytest.fixture()
def pool(faq_df):
    return [build_retriever("tfidf_char").fit(faq_df), build_retriever("bm25").fit(faq_df)]


def _empty():
    return read_eval_files()


def test_add_query_validations(paths, faq_df):
    q, r = _empty()
    q, r, qid = add_query(q, r, faq_df, "общага есть?", "ayan", [10])
    assert qid == 1 and r.iloc[0]["topic"] == "dormitory"   # тема по умолчанию — тема кластера
    assert q.iloc[0]["split"] == UNASSIGNED
    with pytest.raises(ValueError, match="уже есть"):
        add_query(q, r, faq_df, "Общага есть", "ayan", [10])
    with pytest.raises(ValueError, match="утечка"):
        add_query(q, r, faq_df, "Есть ли у вас общежитие?", "ayan", [10])
    with pytest.raises(ValueError, match="Нет таких"):
        add_query(q, r, faq_df, "что-то новое", "ayan", [999])
    with pytest.raises(ValueError, match="хотя бы один"):
        add_query(q, r, faq_df, "что-то новое", "ayan", [])
    with pytest.raises(ValueError, match="не по теме"):
        add_query(q, r, faq_df, "погода", "ayan", [10], is_oos=True)
    q, r, qid = add_query(q, r, faq_df, "какая погода", "ayan", is_oos=True)
    assert qid == 2 and qid not in set(r["query_id"])


def test_import_is_all_or_nothing_and_skips_unreviewed(paths, faq_df, tmp_path):
    good = pd.DataFrame([
        {"text": "сколько стоит общага", "author": "ayan", "is_oos": 0, "answer_cluster": "11",
         "topic": "dormitory", "origin": "llm_draft", "reviewed": 1},
        {"text": "проходной на cs и se", "author": "ayan", "is_oos": 0, "answer_cluster": "30|20",
         "topic": "", "origin": "llm_draft", "reviewed": 1},
        {"text": "рецепт плова", "author": "ayan", "is_oos": 1, "answer_cluster": "",
         "topic": "", "origin": "llm_draft", "reviewed": 1},
        {"text": "не проверено", "author": "ayan", "is_oos": 0, "answer_cluster": "10",
         "topic": "dormitory", "origin": "llm_draft", "reviewed": 0},
    ])
    good.to_csv(tmp_path / "draft.csv", index=False)
    assert import_drafts(tmp_path / "draft.csv", faq_df) == 3
    q, r = read_eval_files()
    assert len(q) == 3 and set(q["origin"]) == {"llm_draft"}
    assert sorted(r.loc[r["query_id"] == 2, "answer_cluster"]) == [20, 30]

    bad = good.head(1).assign(text="новый запрос", answer_cluster="999")
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="строка 2"):
        import_drafts(tmp_path / "bad.csv", faq_df)
    assert len(read_eval_files()[0]) == 3                  # набор не изменился


def test_candidate_pool_has_distinct_clusters(faq_df, pool):
    view = candidate_pool(pool, faq_df, "сколько стоит общежитие")
    assert view["answer_cluster"].is_unique
    assert {10, 11} <= set(view["answer_cluster"])
    assert list(search_faq(faq_df, "мангилик")["answer_cluster"]) == [50]


def test_interactive_session(paths, faq_df, pool):
    script = iter(["общага есть?", "1", "",                # выбрать первого кандидата, тема по умолчанию
                   "погода в париже", "o",                 # не по теме
                   "адрес вуза", "f мангилик", "1", "contacts",  # поиск по подстроке
                   "что-то", "s",                          # пропуск
                   ""])
    out = []
    n = interactive("ayan", faq_df, input_fn=lambda _: next(script), print_fn=out.append,
                    retrievers=pool)
    assert n == 3
    q, r = read_eval_files()
    assert list(q["text"]) == ["общага есть?", "погода в париже", "адрес вуза"]
    assert list(q["is_oos"]) == [0, 1, 0]
    assert r.set_index("query_id").loc[3, "answer_cluster"] == 50


def test_review_and_agreement(paths, faq_df, pool):
    q, r = _empty()
    q, r, _ = add_query(q, r, faq_df, "общага есть?", "alen", [10])
    q, r, _ = add_query(q, r, faq_df, "погода", "alen", is_oos=True)
    from src.annotate import write_eval_files
    write_eval_files(q, r)
    script = iter(["f три дома", "1", "", "o"])
    path = review("ayan", "alen", faq_df, input_fn=lambda _: next(script), print_fn=lambda _: None,
                  retrievers=pool)
    other = pd.read_csv(path)
    metrics, diff = annotator_agreement(r, other, query_ids=sorted(set(other["query_id"])))
    assert metrics["n"] == 2 and metrics["cluster_agreement"] == 1.0 and diff.empty


def test_freeze_once_then_evaluation_ready(paths, faq_df):
    q, r = _empty()
    for i in range(10):
        q, r, _ = add_query(q, r, faq_df, f"общага вопрос {i}", "ayan", [10])
    for i in range(4):
        q, r, _ = add_query(q, r, faq_df, f"погода {i}", "ayan", is_oos=True)
    from src.annotate import write_eval_files
    write_eval_files(q, r)
    counts = freeze()
    assert counts["dev"] == 4 and counts["test"] == 10   # 3 из 10 по теме + 1 из 4 не по теме
    with pytest.raises(ValueError, match="заморожен"):
        freeze()
    queries, qrels = load_eval_set()                       # формат подходит оценке
    assert set(queries["split"]) == {"dev", "test"}
    stats = check(faq_df)
    assert stats["frozen"] and stats["n_oos"] == 4
    with pytest.raises(ValueError, match="заморожен"):
        add_query(queries, qrels, faq_df, "новый", "ayan", [10])
