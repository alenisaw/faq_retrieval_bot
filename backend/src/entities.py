"""Извлечение сущностей из FAQ -> таблица entities в faq.db.

Типы: URL, EMAIL, PHONE, MONEY, DATE — регулярными выражениями (формат у них
жёсткий, регулярки точнее модели); ORG, LOC — NER spaCy ru_core_news_sm.
Если spaCy или модели нет, ORG/LOC ищутся по небольшому словарю.

Зачем: SQL-запрос 7 находит ответы с датами и суммами, которые могли
устареть (стоимость обучения, сроки приёма), а контакты (телефоны, почта,
сайты) видно одним запросом.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import pandas as pd

ENTITY_TYPES = ("ORG", "LOC", "URL", "PHONE", "EMAIL", "MONEY", "DATE")

_MONTHS = (r"января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
           r"|январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь")

_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "URL": re.compile(r"(?:https?://|www\.)[^\s,;)»\"']+|\b[\w-]+(?:\.[\w-]+)*\.(?:kz|com|org|edu|ru)"
                      r"(?:/[^\s,;)»\"']*)?", re.IGNORECASE),
    "PHONE": re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3,4}\)?(?:[\s-]?\d){6,7}(?!\d)"),
    "MONEY": re.compile(
        r"\d[\d\s.,]*\d\s*(?:тыс\.?|млн\.?|миллион\w*)?\s*(?:\([^)]{0,80}\)\s*)?(?:тенге|тг\b|₸|kzt)"
        r"|\d+\s*(?:тыс\.?|млн\.?|миллион\w*)\s*(?:тенге|тг\b|₸)", re.IGNORECASE),
    "DATE": re.compile(
        rf"(?<![\w.])\d{{1,2}}\.\d{{1,2}}\.(?:19|20)?\d{{2}}(?![\d.])"
        rf"|\b\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?\s+(?:{_MONTHS})(?:\s+\d{{4}}(?:\s*года?)?)?"
        rf"|\b(?:{_MONTHS})\s+\d{{4}}"
        rf"|\b(?:19|20)\d{{2}}(?:\s*[-–/]\s*(?:19|20)\d{{2}})?\s*(?:год\w*|г\.|уч\w* год\w*)",
        re.IGNORECASE),
}

# Запасной словарь ORG/LOC (если нет spaCy). Строки ищутся без учёта регистра.
_GAZETTEER: dict[str, tuple[str, ...]] = {
    "ORG": ("Astana IT University", "AITU", "АИТУ", "Astana Hub", "Astana IT College", "МФЦА",
            "Казахтелеком", "Ustudy", "PwC Kazakhstan", "NURIS", "Министерство цифрового развития"),
    "LOC": ("Астана", "Нур-Султан", "Казахстан", "Алматы", "Шымкент", "Центральной Азии",
            "Мангилик Ел", "Латвия", "Турция", "Германия", "Польша"),
}


@lru_cache(maxsize=1)
def _ner_model():
    """spaCy с включённым NER (в preprocessing он отключён ради скорости)."""
    try:
        import spacy

        return spacy.load("ru_core_news_sm", disable=["parser", "lemmatizer", "attribute_ruler"])
    except (ImportError, OSError):
        return None


def ner_available() -> bool:
    return _ner_model() is not None


def _clean(text: str) -> str:
    if text.count("(") > text.count(")"):          # "Astana IT University (AITU / аиту" -> без хвоста
        text = text.split("(")[0]
    return " ".join(text.strip(" .,;:–-\"'«»()").split())


def regex_entities(text: str) -> list[tuple[str, str]]:
    """[(текст, тип)] для URL, EMAIL, PHONE, MONEY, DATE."""
    found, taken = [], []
    for etype in ("EMAIL", "URL", "PHONE", "MONEY", "DATE"):
        for m in _PATTERNS[etype].finditer(text):
            span = m.span()
            # e-mail содержит домен, похожий на URL; телефон похож на дату и сумму
            if any(span[0] < e and s < span[1] for s, e in taken):
                continue
            value = _clean(m.group())
            if etype == "URL" and "@" in value:
                continue
            if value:
                found.append((value, etype))
                taken.append(span)
    return found


def gazetteer_entities(text: str) -> list[tuple[str, str]]:
    low = text.lower()
    return [(name, etype) for etype, names in _GAZETTEER.items()
            for name in names if name.lower() in low]


def ner_entities(texts: list[str], batch_size: int = 32) -> list[list[tuple[str, str]]]:
    """ORG/LOC для каждого текста (spaCy, иначе словарь)."""
    nlp = _ner_model()
    if nlp is None:
        return [gazetteer_entities(t) for t in texts]
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        ents = [(_clean(e.text), e.label_) for e in doc.ents if e.label_ in ("ORG", "LOC")]
        out.append([(t, label) for t, label in ents if len(t) > 1])
    return out


def extract_entities(faq_df: pd.DataFrame, use_ner: bool = True) -> pd.DataFrame:
    """Сущности из вопросов и ответов -> (faq_id, text, type), без повторов внутри FAQ."""
    texts = (faq_df["question"].fillna("") + "\n" + faq_df["answer"].fillna("")).tolist()
    ner = ner_entities(texts) if use_ner else [gazetteer_entities(t) for t in texts]
    rows, seen = [], set()
    for faq_id, text, named in zip(faq_df["faq_id"], texts, ner):
        for value, etype in regex_entities(text) + named:
            key = (int(faq_id), value.lower(), etype)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"faq_id": int(faq_id), "text": value, "type": etype})
    return pd.DataFrame(rows, columns=["faq_id", "text", "type"])


def entity_summary(entities: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Сколько сущностей каждого типа и самые частые значения — для отчёта."""
    rows = []
    for etype in ENTITY_TYPES:
        sub = entities[entities["type"] == etype]
        common = sub["text"].value_counts().head(top)
        rows.append({"type": etype, "n": len(sub), "n_faqs": sub["faq_id"].nunique(),
                     "top": ", ".join(f"{t} ({n})" for t, n in common.items())})
    return pd.DataFrame(rows)


def iter_types(entities: pd.DataFrame, types: Iterable[str]) -> pd.DataFrame:
    return entities[entities["type"].isin(list(types))]
