"""Игрушечная база FAQ для тестов AI-модуля (не зависит от реальных данных)."""

import pandas as pd
import pytest

FAQ_ROWS = [
    # faq_id, question, answer, topic, answer_cluster
    (1, "Есть ли у вас общежитие?", "Да, есть три дома студентов.", "dormitory", 10),
    (2, "Предоставляется ли общежитие студентам?", "Да, есть три дома студентов.", "dormitory", 10),
    (3, "Сколько стоит проживание в общежитии?", "Проживание стоит 50 000 тенге в месяц.", "dormitory", 11),
    (4, "Какова стоимость обучения?", "Обучение стоит 2 000 000 тенге в год.", "tuition", 20),
    (5, "Сколько стоит учеба в университете?", "Обучение стоит 2 000 000 тенге в год.", "tuition", 20),
    (6, "Какой проходной балл на Computer Science?", "85 баллов.", "scores_grants", 30),
    (7, "Сколько баллов нужно для Software Engineering?", "85 баллов.", "scores_grants", 30),
    (8, "Есть ли в университете военная кафедра?", "Да, военная кафедра есть.", "military", 40),
    (9, "Есть ли в университете военная кафедра?", "Да, военная кафедра есть.", "military", 40),
    (10, "Где находится Astana IT University?", "Проспект Мангилик Ел, 55/11.", "contacts", 50),
    (11, "Какой адрес университета AITU?", "Проспект Мангилик Ел, 55/11.", "contacts", 50),
    (12, "Как проходит творческий экзамен?", "Сочинение и творческий проект.", "creative_exam", 60),
]


@pytest.fixture(scope="session")
def faq_df() -> pd.DataFrame:
    df = pd.DataFrame(FAQ_ROWS, columns=["faq_id", "question", "answer", "topic", "answer_cluster"])
    df["is_dated"] = df["faq_id"] == 3
    return df


@pytest.fixture()
def eval_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = pd.DataFrame([
        {"query_id": 1, "text": "общага есть?", "split": "test", "is_oos": False},
        {"query_id": 2, "text": "сколько стоит обучение", "split": "test", "is_oos": False},
        {"query_id": 3, "text": "проходной балл computer science", "split": "dev", "is_oos": False},
        {"query_id": 4, "text": "где находится университет", "split": "dev", "is_oos": False},
        {"query_id": 5, "text": "какая погода в париже", "split": "dev", "is_oos": True},
        {"query_id": 6, "text": "рецепт плова", "split": "test", "is_oos": True},
    ])
    qrels = pd.DataFrame([
        {"query_id": 1, "answer_cluster": 10, "topic": "dormitory"},
        {"query_id": 2, "answer_cluster": 20, "topic": "tuition"},
        {"query_id": 3, "answer_cluster": 30, "topic": "scores_grants"},
        {"query_id": 4, "answer_cluster": 50, "topic": "contacts"},
    ])
    return queries, qrels
