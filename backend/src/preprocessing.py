"""Предобработка русского текста для поиска по FAQ.

Пайплайн: normalize -> tokenize -> стоп-слова -> stem | lemma.

Одна функция `preprocess(text, mode)` с режимами:
    raw        только нормализация и токенизация
    stem       NLTK SnowballStemmer("russian")
    lemma      pymorphy3 (основной лемматизатор)
    lemma_spacy  spaCy ru_core_news_sm (для сравнения, если установлен)

Латиница (Software Engineering, AET, IELTS) в режимах stem/lemma*
стеммится английским Snowball: pymorphy3 и русский Snowball её не трогают,
а без этого "engineering" и "engineer" остались бы разными токенами.

Чтобы любой метод можно было гонять с любым режимом, sklearn-векторизаторы
получают анализатор через `make_analyzer(...)`.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import pandas as pd

MODES = ("raw", "stem", "lemma", "lemma_spacy")

# --- Нормализация ------------------------------------------------------

# Варианты названия университета -> один токен. Порядок важен: сначала
# длинные формы, иначе "astana it university" распадётся на части.
_AITU_PATTERNS = [
    r"astana\s+it\s+university",
    r"астана\s+(?:ит|айти|it)\s+университет\w*",
    r"\baitu\b",
    r"\bаиту\b",
    r"\bайту\b",
]
_AITU_RE = re.compile("|".join(_AITU_PATTERNS), flags=re.IGNORECASE)
_AITU_REPEAT_RE = re.compile(r"\baitu\b(?:\W+\baitu\b)+")

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Всё, что не буква/цифра, -> пробел. \w в Python понимает и казахские буквы.
_NON_WORD_RE = re.compile(r"[^\w\s]|_")
_SPACES_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"^[a-z]+$")
_TOKEN_RE = re.compile(r"^\w+$")


def normalize(text: str) -> str:
    """Нижний регистр, ё->е, без URL/email/пунктуации, синонимы AITU -> aitu."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("ё", "е")
    text = text.replace("\\n", " ").replace("\n", " ")
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _AITU_RE.sub(" aitu ", text)
    text = _NON_WORD_RE.sub(" ", text)  # заодно режет дефисы: "it-компании" -> "it компании"
    text = _AITU_REPEAT_RE.sub("aitu", text)  # "aitu / aitu" -> "aitu"
    return _SPACES_RE.sub(" ", text).strip()


# --- Токенизация ---------------------------------------------------------

_nltk_ready: bool | None = None


def _ensure_nltk() -> bool:
    """Скачивает punkt_tab и stopwords при первом вызове. False — офлайн."""
    global _nltk_ready
    if _nltk_ready is not None:
        return _nltk_ready
    import nltk

    ok = True
    for resource, path in (("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                ok &= bool(nltk.download(resource, quiet=True))
            except Exception:  # нет сети, нет прав на запись и т.п.
                ok = False
    _nltk_ready = ok
    return ok


def tokenize(text: str) -> list[str]:
    """NLTK word_tokenize(language="russian"); без NLTK-данных — split по пробелам.

    Текст должен быть уже нормализован: пунктуации в нём нет, поэтому
    запасной вариант даёт тот же результат.
    """
    if not text:
        return []
    tokens: list[str]
    if _ensure_nltk():
        from nltk.tokenize import word_tokenize

        try:
            tokens = word_tokenize(text, language="russian")
        except LookupError:
            tokens = text.split()
    else:
        tokens = text.split()
    return [t for t in tokens if _TOKEN_RE.match(t)]


# --- Стоп-слова ----------------------------------------------------------

# Вопросительные слова несут интент ("где" -> адрес, "сколько" -> цена/баллы).
# Часть из них есть в стоп-листе NLTK, часть нет — держим полный список,
# чтобы режим keep_question_words работал одинаково для всех.
QUESTION_WORDS = frozenset({
    "где", "куда", "откуда", "когда", "сколько", "как", "какой", "какая",
    "какое", "какие", "каков", "какова", "каково", "каковы", "кто", "что",
    "чем", "чего", "зачем", "почему", "можно",
})

_FALLBACK_STOPWORDS = frozenset({
    "и", "в", "во", "не", "что", "на", "я", "с", "со", "как", "а", "то", "все",
    "так", "но", "да", "к", "у", "же", "вы", "за", "бы", "по", "только", "от",
    "о", "из", "ли", "если", "уже", "или", "ни", "быть", "до", "для", "мы",
    "их", "чем", "без", "под", "кто", "этот", "при", "об", "про", "есть",
    "где", "когда", "какой", "какая", "можно", "куда", "зачем", "чтобы",
})


@lru_cache(maxsize=1)
def russian_stopwords() -> frozenset[str]:
    """Стоп-лист NLTK для русского (151 слово) с ё->е; офлайн — короткий запасной."""
    if _ensure_nltk():
        try:
            from nltk.corpus import stopwords

            return frozenset(w.replace("ё", "е") for w in stopwords.words("russian"))
        except LookupError:
            pass
    return _FALLBACK_STOPWORDS


def get_stopwords(keep_question_words: bool = False) -> frozenset[str]:
    """Стоп-лист NLTK, в котором вопросительные слова убраны или добавлены целиком.

    В NLTK есть "какой", но нет "какие" и "сколько". Без выравнивания
    сравнение "с вопросительными словами / без них" было бы нечистым.
    """
    sw = russian_stopwords()
    return sw - QUESTION_WORDS if keep_question_words else sw | QUESTION_WORDS


# --- Стемминг и лемматизация --------------------------------------------

@lru_cache(maxsize=1)
def _ru_stemmer():
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("russian")


@lru_cache(maxsize=1)
def _en_stemmer():
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("english")


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=1)
def _spacy_nlp():
    try:
        import spacy

        return spacy.load("ru_core_news_sm", disable=["parser", "ner"])
    except (ImportError, OSError) as exc:  # нет spaCy или модели
        raise RuntimeError(
            "Режим lemma_spacy требует spaCy и модель ru_core_news_sm: "
            "pip install spacy && python -m spacy download ru_core_news_sm"
        ) from exc


@lru_cache(maxsize=100_000)
def _stem(token: str) -> str:
    if _LATIN_RE.match(token):
        return _en_stemmer().stem(token)
    return _ru_stemmer().stem(token)


@lru_cache(maxsize=100_000)
def _lemma(token: str) -> str:
    if _LATIN_RE.match(token):
        return _en_stemmer().stem(token)
    return _morph().parse(token)[0].normal_form.replace("ё", "е")


def spacy_available() -> bool:
    try:
        _spacy_nlp()
        return True
    except RuntimeError:
        return False


def _lemmatize_spacy(tokens: list[str]) -> list[str]:
    from spacy.tokens import Doc

    nlp = _spacy_nlp()
    doc = Doc(nlp.vocab, words=tokens)  # своя токенизация, spaCy только размечает
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    out = []
    for tok in doc:
        if _LATIN_RE.match(tok.text):
            out.append(_en_stemmer().stem(tok.text))
        else:
            out.append((tok.lemma_ or tok.text).lower().replace("ё", "е"))
    return out


# --- Главная функция ------------------------------------------------------

def preprocess(
    text: str,
    mode: str = "lemma",
    remove_stopwords: bool = True,
    keep_question_words: bool = False,
) -> list[str]:
    """Текст -> список токенов.

    Args:
        mode: raw | stem | lemma | lemma_spacy.
        remove_stopwords: убирать стоп-слова NLTK (по исходной словоформе).
        keep_question_words: не удалять вопросительные слова (где, сколько...).
    """
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим {mode!r}, доступны: {', '.join(MODES)}")

    tokens = tokenize(normalize(text))
    sw = get_stopwords(keep_question_words) if remove_stopwords else frozenset()

    if mode == "lemma_spacy":
        # spaCy разбирает предложение целиком (ему нужен контекст),
        # стоп-слова убираем уже после, по исходной словоформе
        if not tokens:
            return []
        return [lem for tok, lem in zip(tokens, _lemmatize_spacy(tokens)) if tok not in sw]

    tokens = [t for t in tokens if t not in sw]
    if mode == "stem":
        return [_stem(t) for t in tokens]
    if mode == "lemma":
        return [_lemma(t) for t in tokens]
    return tokens


def preprocess_text(text: str, **kwargs) -> str:
    """То же, что preprocess, но строкой через пробел (для логов и sklearn)."""
    return " ".join(preprocess(text, **kwargs))


@dataclass(frozen=True)
class TextAnalyzer:
    """Анализатор для CountVectorizer/TfidfVectorizer(analyzer=...).

    Класс, а не замыкание: так векторизатор сохраняется через joblib/pickle.
    Словные n-граммы строит сам, потому что при analyzer=callable
    sklearn параметр ngram_range игнорирует.
    """

    mode: str = "lemma"
    remove_stopwords: bool = True
    keep_question_words: bool = False
    ngram_range: tuple[int, int] = (1, 1)

    def __call__(self, text: str) -> list[str]:
        toks = preprocess(text, mode=self.mode, remove_stopwords=self.remove_stopwords,
                          keep_question_words=self.keep_question_words)
        lo, hi = self.ngram_range
        if hi <= 1:
            return toks
        grams: list[str] = []
        for n in range(lo, hi + 1):
            grams.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return grams


@dataclass(frozen=True)
class CharPreprocessor:
    """preprocessor для char_wb TF-IDF: нормализация и, по желанию, без стоп-слов."""

    remove_stopwords: bool = False
    keep_question_words: bool = False

    def __call__(self, text: str) -> str:
        if not self.remove_stopwords:
            return normalize(text)
        return " ".join(preprocess(text, mode="raw", remove_stopwords=True,
                                   keep_question_words=self.keep_question_words))


def make_analyzer(mode: str = "lemma", remove_stopwords: bool = True,
                  keep_question_words: bool = False,
                  ngram_range: tuple[int, int] = (1, 1)) -> TextAnalyzer:
    return TextAnalyzer(mode, remove_stopwords, keep_question_words, tuple(ngram_range))


# --- Сравнение режимов (для отчёта) ---------------------------------------

def compare_modes(
    texts: Iterable[str],
    modes: Iterable[str] | None = None,
    remove_stopwords: bool = True,
) -> pd.DataFrame:
    """Размер словаря, число токенов и время обработки корпуса по режимам.

    Кэши стеммера и лемматизатора сбрасываются перед каждым замером,
    иначе второй прогон был бы нечестно быстрым.
    """
    texts = list(texts)
    if modes is None:
        modes = [m for m in MODES if m != "lemma_spacy" or spacy_available()]

    # прогрев: загрузка словарей pymorphy3 и NLTK не должна попадать в замер
    _ensure_nltk(), _morph(), _ru_stemmer(), _en_stemmer()

    rows = []
    for mode in modes:
        _stem.cache_clear()
        _lemma.cache_clear()
        start = time.perf_counter()
        docs = [preprocess(t, mode=mode, remove_stopwords=remove_stopwords) for t in texts]
        elapsed = time.perf_counter() - start
        counts = Counter(tok for doc in docs for tok in doc)
        n_tokens = sum(counts.values())
        rows.append({
            "mode": mode,
            "vocab_size": len(counts),
            "n_tokens": n_tokens,
            "avg_tokens_per_doc": n_tokens / max(len(docs), 1),
            "hapax_share": sum(1 for c in counts.values() if c == 1) / max(len(counts), 1),
            "time_s": elapsed,
            "ms_per_doc": 1000 * elapsed / max(len(docs), 1),
        })
    return pd.DataFrame(rows)
