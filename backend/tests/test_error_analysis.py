import pandas as pd

from src import error_analysis as ea


def _pq():
    return pd.DataFrame({
        "query_id": [1, 2, 3, 4, 5, 6, 7],
        "query": ["скок стоит учеба", "общежитие есть", "адрес", "военка", "погода", "контакты", "кампус"],
        "is_oos": [False, False, False, False, True, False, False],
        "gold_clusters": ["10", "20", "30", "40", "", "50", "60"],
        "gold_topic": ["tuition", "dormitory", "contacts", "military", None, "contacts", "about"],
        "top1_cluster": [11, 20, 31, 41, 99, 51, 61],
        "top1_topic": ["dormitory", "dormitory", "contacts", "military", "about", "contacts", "about"],
        "top1_question": ["q"] * 7,
        "top1_score": [0.3, 0.9, 0.5, 0.4, 0.6, 0.5, 0.2],
        "confidence": [0.3, 0.9, 0.5, 0.4, 0.6, 0.5, 0.2],
        "first_rel_rank": [5.0, 1.0, 2.0, 7.0, None, 4.0, None],
        "rr": [0.2, 1.0, 0.5, 1 / 7, 0.0, 0.25, 0.0],
        "hit@1": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })


def _faq():
    return pd.DataFrame({"answer_cluster": [10, 20, 30, 40, 50, 60], "question": list("abcdef"),
                         "answer": list("ABCDEF"), "topic": ["tuition"] * 6})


def test_categories_follow_rules():
    misses = ea.analyse(_pq(), _faq(), acceptable={6})
    cat = dict(zip(misses["query_id"], misses["category"]))
    assert cat == {1: "other_topic", 3: "near_miss", 4: "same_topic", 6: "duplicate", 7: "not_found"}


def test_slang_and_oos():
    slang = ea.slang_table(_pq())
    # сленг: «скок стоит учеба» и «военка»; остальные 4 запроса по теме — без сленга
    assert dict(zip(slang["запросы"], slang["n"])) == {"без сленга": 4, "есть сленг/сокращения": 2}
    assert len(ea.oos_false_answers(_pq(), threshold=0.5)) == 1
    assert len(ea.oos_false_answers(_pq(), threshold=0.7)) == 0
