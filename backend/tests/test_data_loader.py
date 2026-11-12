import json

import pandas as pd
import pytest

from src import config
from src.data_loader import (CleaningLog, build_answer_clusters, build_clean_faq, clean_rows, clean_text,
                             drop_duplicate_pairs, harmonize_cluster_topics, is_dated, optimize_dtypes,
                             prepare_data)
from src.topic_map import assign_topic, match_topic


def raw_frame(rows):
    return pd.DataFrame(rows, columns=["faq_id", "questions", "answers", "classes"])


# --- очистка текста -------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ('  "Математика и физика."  ', "Математика и физика."),
    ("Строка\\nс переносом", "Строка с переносом"),
    ("нужно\\\\необходимо", "нужно/необходимо"),
    (' "После получения образования', "После получения образования"),   # непарная кавычка
    (None, ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_row_rules():
    raw = raw_frame([
        (1, "Есть ли общежитие?", "Да, есть.", "AITU_dormitory"),
        (2, "questions", "answers", "classes"),
        (3, "1. Информационные технологии", " для которых нужны предметы", " а минимальный балл - 85."),
        (4, "Кем можно стать?", ' "После обучения', " выпускники могут стать:"),
        (5, "Где кабинет 1с?, ответ:  Кабинет на втором этаже.", ' "Расположение кабинета"', "Кабинет"),
        (6, "Вопрос 1: Какие скидки есть?", "Скидки до 30%.", "Financial Aid"),
        (7, "Интернет вещей: архитектура и протоколы", "AITU_iot", "Smart Technologies"),
        (8, "Основы журналистики: введение в профессию", "Introduction to journalism", "Digital Journalism"),
        (9, "Какая модель выпускника?", ' "Модель не представлена в данном контексте"', "x"),
        (10, "Устав", "Деятельность ведётся по уставу.", "документы"),
    ])
    log = CleaningLog()
    out = clean_rows(raw, log)
    assert out["faq_id"].tolist() == [1, 5, 6, 10]
    rules = {e["faq_id"]: e["rule"] for e in log.entries if e["action"] == "dropped"}
    assert rules == {2: "header_row", 3: "broken_csv_row", 4: "broken_csv_row", 7: "answer_is_slug",
                     8: "translation_not_answer", 9: "empty_answer"}
    row5 = out.set_index("faq_id").loc[5]
    assert row5["question"] == "Где кабинет 1с?" and row5["answer"] == "Кабинет на втором этаже."
    assert out.set_index("faq_id").loc[6, "question"] == "Какие скидки есть?"


def test_duplicate_pairs_keep_first():
    df = pd.DataFrame({"faq_id": [5, 1, 9], "question": ["Есть общежитие?", "есть  общежитие ?", "Другое?"],
                       "answer": ["Да.", "да", "Нет."], "orig_class": ["a", "b", "c"]})
    log = CleaningLog()
    out = drop_duplicate_pairs(df, log)
    assert sorted(out["faq_id"]) == [1, 9]
    assert log.entries[0]["faq_id"] == 5 and "faq_id=1" in log.entries[0]["detail"]


def test_clusters_short_answers_need_same_class():
    df = pd.DataFrame({
        "faq_id": [10, 11, 12, 13, 14, 15],
        "question": ["Балл CS?", "Проходной CS?", "Балл SE?", "Как поступить?", "Как поступить?", "Перефраз"],
        "answer": ["85 баллов.", "85 баллов.", "85 баллов.", "Длинный ответ " * 10, "Совсем другой ответ " * 5,
                   "Ответ перефраза"],
        "orig_class": ["cs_score", "cs_score", "se_score", "a", "b", "c"],
    })
    clusters = build_answer_clusters(df, manual_links=[(13, 15)])
    by_id = dict(zip(df["faq_id"], clusters))
    assert by_id[10] == by_id[11] == 10          # тот же короткий ответ и класс
    assert by_id[12] == 12                       # тот же ответ, другая специальность
    assert by_id[13] == by_id[14] == by_id[15] == 13  # одинаковый вопрос + ручная связь


def test_harmonize_majority_and_tie():
    df = pd.DataFrame({"faq_id": [1, 2, 3, 4, 5], "answer_cluster": [1, 1, 1, 4, 4],
                       "topic": ["tuition", "admission", "admission", "admission", "scores_grants"]})
    log = CleaningLog()
    topics = harmonize_cluster_topics(df, log).tolist()
    assert topics == ["admission"] * 3 + ["admission", "admission"]   # ничья -> тема минимального faq_id
    assert {e["faq_id"] for e in log.entries} == {1, 5}


def test_is_dated():
    assert is_dated("Экзамен пройдёт 08.12.2022")
    assert is_dated("на момент 2023 года")
    assert is_dated("в городе Нур-Султан")
    assert not is_dated("Университет основан в 2019 году")   # исторический факт, не устаревает
    assert not is_dated("Стоимость 2,500,000 тенге")


@pytest.mark.parametrize("question, topic", [
    ("Есть ли общежитие?", "dormitory"),
    ("Какие документы нужны для поступления на военную кафедру?", "military"),
    ("Какой проходной балл на Computer Science?", "scores_grants"),
    ("Какие предметы стоит выбрать для сдачи ЕНТ?", "scores_grants"),   # "стоит" — не про цену
    ("Сколько стоят курсы английского?", "tuition"),
    ("Когда открывается приемная комиссия?", "admission"),          # "комиссия" содержит "миссия"
    ("Какая миссия университета?", "about"),
    ("Кем можно стать окончив Digital Journalism?", "programs"),
    ("Какая оплата за AITU Excellence Test?", "admission"),
    ("Как связаться с приемной комиссией?", "contacts"),
])
def test_topic_rules(question, topic):
    assert assign_topic(question, "")[0] == topic


def test_topic_falls_back_to_class_then_default():
    assert assign_topic("Что это?", "AITU_dormitory") == ("dormitory", "class:dormitory")
    assert assign_topic("Что это?", "")[1] == "default"
    assert match_topic("") is None
    assert all(t in config.TOPICS for t, *_ in __import__("src.topic_map").topic_map.TOPIC_RULES)


def test_optimize_dtypes():
    df = pd.DataFrame({"faq_id": [1, 2], "question": ["a", "b"], "answer": ["x", "y"], "orig_class": ["", ""],
                       "topic": ["about", "military"], "answer_cluster": [1, 1], "is_dated": [0, 1]})
    out = optimize_dtypes(df)
    assert str(out["faq_id"].dtype) == "int32" and str(out["topic"].dtype) == "category"
    assert out["is_dated"].dtype == bool
    assert list(out["topic"].cat.categories) == list(config.TOPICS)


# --- реальные данные ---------------------------------------------------------------

@pytest.fixture(scope="module")
def real_clean(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clean")
    df, summary = prepare_data(config.RAW_CSV_PATH, tmp / "faq_clean.parquet", tmp / "log.json",
                               write_reviews=False, verbose=False)
    return df, summary, tmp


def test_real_data_contract(real_clean):
    df, summary, tmp = real_clean
    assert list(df.columns) == ["faq_id", "question", "answer", "orig_class", "topic", "answer_cluster", "is_dated"]
    assert summary["rows_raw"] == 608
    assert 500 <= len(df) < 608
    assert df["faq_id"].is_unique
    assert set(df["topic"].astype(str)) == set(config.TOPICS)       # все 12 тем непустые
    assert set(df["answer_cluster"]) <= set(df["faq_id"])           # номер кластера = faq_id его строки
    assert df.groupby("answer_cluster")["topic"].nunique().max() == 1
    reread = pd.read_parquet(tmp / "faq_clean.parquet")
    assert len(reread) == len(df) and str(reread["topic"].dtype) == "category"
    log = json.loads((tmp / "log.json").read_text(encoding="utf-8"))
    assert log["rows_clean"] == len(df) and log["entries"]


def test_real_data_known_rows(real_clean):
    df, _, _ = real_clean
    ids = set(df["faq_id"])
    assert not ids & {207, 65, 122, 309, 490, 166}                    # заголовок, обрывки, пустой, slug
    by_id = df.set_index("faq_id")
    assert by_id.loc[220, "question"] == "Где кабинет 1с?"
    assert by_id.loc[356, "question"].startswith("Какие виды скидок")
    assert by_id.loc[16, "topic"] == "dormitory"
    cs = set(df.loc[df["answer_cluster"] == by_id.loc[78, "answer_cluster"], "faq_id"])
    assert {78, 79, 80, 81, 92, 105} <= cs                            # перефразы про балл CS
    assert by_id.loc[92, "answer_cluster"] != by_id.loc[93, "answer_cluster"]   # CS и SE не склеены


def test_build_clean_faq_review_has_sources():
    raw = raw_frame([(1, "Есть ли общежитие?", "Да.", "x"), (2, "Что это?", "Ответ.", "AITU_military")])
    clean, log, review = build_clean_faq(raw)
    assert review.set_index("faq_id").loc[2, "topic_source"] == "class:military"
