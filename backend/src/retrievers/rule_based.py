"""Baseline: ключевые слова -> тема, внутри темы — Jaccard по леммам.

1. Леммы запроса голосуют за темы по словарю TOPIC_KEYWORDS (вес слова
   1 или 2: "общежитие" однозначнее, чем "срок").
2. Score документа = Jaccard(леммы запроса, леммы вопроса FAQ),
   плюс 1.0, если тема документа совпала с темой запроса. Поэтому FAQ
   нужной темы всегда идут выше, а внутри темы порядок задаёт Jaccard.
3. Если ни одно ключевое слово не сработало или две темы набрали поровну —
   Jaccard по всей базе.

Словарь составлен вручную по просмотру всех 608 вопросов FAQ, без
тестовых запросов. Ключи — slug тем из config.TOPICS.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from ..preprocessing import preprocess
from .base import BaseRetriever

# слово -> вес; слова пишутся в начальной форме, при загрузке лемматизируются
TOPIC_KEYWORDS: dict[str, dict[str, float]] = {
    "about": {
        "миссия": 2, "видение": 2, "учредитель": 2, "аккредитация": 2, "устав": 2,
        "лицензия": 2, "история": 1, "основать": 1, "репутация": 1, "рейтинг": 1,
        "ректор": 1, "экосистема": 2, "технопарк": 2, "партнер": 1, "наука": 1,
        "научный": 1, "исследование": 1, "исследовательский": 1, "инновация": 1,
        "лаборатория": 1, "журнал": 1, "вендор": 1,
    },
    "admission": {
        "поступить": 1, "поступление": 1, "поступать": 1, "абитуриент": 1, "прием": 1,
        "приемный": 1, "документ": 1, "заявление": 1, "заявка": 1, "подача": 1,
        "подать": 1, "регистрация": 1, "aet": 2, "ает": 2, "excellence": 2, "вступительный": 2,
        "экзамен": 1, "зачисление": 1, "результат": 1, "колледж": 1, "foundation": 1,
        "пробник": 2, "дедлайн": 1, "ielts": 1, "toefl": 1,
    },
    "scores_grants": {
        "балл": 2, "проходной": 2, "пороговый": 2, "минимальный": 1, "грант": 2,
        "ент": 1, "предмет": 1, "профильный": 1, "сдать": 1, "сдавать": 1,
        "олимпиада": 2, "iqyn": 2, "айкын": 2, "icode": 2, "sat": 2, "mesc": 2, "конкурс": 1,
        "бюджет": 1, "бесплатный": 1, "бесплатно": 1,
    },
    # Названия специальностей (Software Engineering, Cybersecurity...) сюда
    # намеренно не входят: это сущности, а не намерение. "Проходной балл на SE"
    # и "что изучают на SE" содержат одно название, а темы у них разные —
    # тему задают слова "балл", "предмет", "изучать", "кем стать".
    "programs": {
        "специальность": 1, "бакалавриат": 1, "образовательный": 1, "программа": 1,
        "стать": 1, "профессия": 1, "код": 1, "гоп": 2, "учить": 1, "изучать": 1,
        "проходить": 1, "план": 1, "навык": 1,
    },
    "tuition": {
        "стоимость": 2, "стоить": 2, "цена": 2, "оплата": 2, "оплачивать": 2,
        "платить": 2, "платный": 2, "скидка": 2, "стипендия": 2, "льгота": 2,
        "деньги": 1, "тенге": 2, "рассрочка": 2, "финансовый": 1, "дорого": 1,
    },
    "dormitory": {
        "общежитие": 3, "общага": 3, "проживание": 2, "жилье": 2, "комната": 1,
        "заселение": 2, "поселиться": 2,
    },
    "creative_exam": {
        "творческий": 3, "сочинение": 2, "эссе": 2, "жанр": 2,
    },
    "military": {
        "военный": 3, "кафедра": 1, "армия": 2, "офицер": 2, "сержант": 2, "запас": 1,
        "медкомиссия": 2, "освидетельствование": 2, "призыв": 2,
    },
    "masters": {
        "магистратура": 3, "магистрант": 3, "магистерский": 3, "докторантура": 3,
        "phd": 3, "master": 2,
    },
    "study_process": {
        "триместр": 2, "семестр": 2, "сессия": 2, "moodle": 2, "кредит": 1,
        "оценка": 1, "оценивание": 1, "gpa": 2, "расписание": 2, "онлайн": 1,
        "дистанционный": 2, "удаленный": 1, "занятие": 1, "пара": 1, "учебный": 1,
        "календарь": 1, "аттестация": 2, "длительность": 1, "длиться": 1, "язык": 1,
        "справка": 1, "академический": 1, "учеба": 1,
    },
    "contacts": {
        "контакт": 2, "телефон": 2, "номер": 1, "email": 2, "почта": 2, "адрес": 2,
        "находиться": 1, "расположить": 1, "график": 2, "открыть": 1, "связаться": 2,
        "позвонить": 2, "кабинет": 1,
    },
    "student_life": {
        "клуб": 2, "мероприятие": 1, "спорт": 2, "секция": 2, "коворкинг": 2,
        "стартап": 1, "инкубатор": 2, "mvp": 2, "стажировка": 2, "практика": 1,
        "трудоустройство": 2, "карьера": 1, "работодатель": 1, "мобильность": 2,
        "обмен": 2, "иностранный": 1, "международный": 1, "культурный": 1,
        "волонтер": 2, "хакатон": 2, "сообщество": 1,
    },
}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class RuleBasedRetriever(BaseRetriever):
    name = "rule_based"

    def __init__(self, keywords: dict[str, dict[str, float]] | None = None,
                 topic_bonus: float = 1.0, index_field: str = "question") -> None:
        super().__init__(index_field)
        self.topic_bonus = topic_bonus
        self.keyword_topics = self._build_keyword_index(keywords or TOPIC_KEYWORDS)

    @staticmethod
    def _build_keyword_index(keywords: dict[str, dict[str, float]]) -> dict[str, list[tuple[str, float]]]:
        index: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for topic, words in keywords.items():
            for word, weight in words.items():
                for lemma in preprocess(word, mode="lemma", remove_stopwords=False):
                    index[lemma].append((topic, float(weight)))
        return dict(index)

    @staticmethod
    def _lemmas(text: str) -> set[str]:
        return set(preprocess(text, mode="lemma", remove_stopwords=True))

    def predict_topic(self, query: str) -> str | None:
        """Тема с максимальной суммой весов.

        None — если ни одно слово не сработало или две темы набрали поровну:
        при неоднозначности лучше искать по всей базе, чем угадать тему и
        утопить правильный ответ под FAQ чужой темы.
        """
        votes: dict[str, float] = defaultdict(float)
        for lemma in self._lemmas(query):
            for topic, weight in self.keyword_topics.get(lemma, ()):
                votes[topic] += weight
        if not votes:
            return None
        ranked = sorted(votes.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] == ranked[1]:
            return None
        return max(votes, key=votes.__getitem__)

    def _fit(self, docs: list[str], faq_df: pd.DataFrame) -> None:
        self.doc_lemmas = [self._lemmas(d) for d in docs]

    def _raw_scores(self, queries: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(queries), len(self.doc_lemmas)))
        for qi, query in enumerate(queries):
            q = self._lemmas(query)
            out[qi] = [jaccard(q, d) for d in self.doc_lemmas]
            topic = self.predict_topic(query)
            if topic is not None and self.topics is not None:
                out[qi] += self.topic_bonus * (self.topics == topic)
        return out

    def _index_state(self):
        return (self.doc_lemmas, self.keyword_topics)

    def params(self) -> dict:
        return {**super().params(), "topic_bonus": self.topic_bonus,
                "n_keywords": len(self.keyword_topics)}
