import pickle

import pytest

from src.preprocessing import (
    QUESTION_WORDS,
    TextAnalyzer,
    compare_modes,
    get_stopwords,
    normalize,
    preprocess,
    tokenize,
)


def test_normalize_basic():
    assert normalize("Ещё ВОПРОС?!") == "еще вопрос"
    assert normalize(None) == ""
    assert normalize("IT-компании") == "it компании"


def test_normalize_removes_urls_and_emails():
    text = "Сайт https://admission.astanait.edu.kz и почта info@astanait.edu.kz"
    assert normalize(text) == "сайт и почта"


@pytest.mark.parametrize("variant", [
    "Astana IT University", "AITU", "аиту", "АИТУ", "Астана ИТ Университет", "Astana IT University (AITU / аиту)",
])
def test_aitu_synonyms_collapse_to_one_token(variant):
    assert normalize(variant) == "aitu"


def test_tokenize_keeps_digits_and_drops_punctuation():
    assert tokenize(normalize("Код 6B06102, балл 85.")) == ["код", "6b06102", "балл", "85"]
    assert tokenize("") == []


def test_question_words_toggle_is_symmetric():
    removed = get_stopwords(keep_question_words=False)
    kept = get_stopwords(keep_question_words=True)
    assert QUESTION_WORDS <= removed          # убираются все, включая "сколько" и "какие"
    assert not (QUESTION_WORDS & kept)        # сохраняются все
    assert "и" in removed and "и" in kept     # обычные стоп-слова убираются в обоих режимах


def test_preprocess_modes():
    q = "Где находятся общежития?"
    assert preprocess(q, mode="raw") == ["находятся", "общежития"]
    assert preprocess(q, mode="lemma") == ["находиться", "общежитие"]
    assert preprocess(q, mode="stem") == ["наход", "общежит"]
    assert preprocess(q, mode="lemma", keep_question_words=True)[0] == "где"


def test_lemma_homonymy_without_context():
    # Известное ограничение pymorphy3: без контекста "стоят" -> "стоять",
    # а не "стоить". Пример для раздела "ограничения" в отчёте.
    assert preprocess("стоят", mode="lemma") == ["стоять"]
    assert preprocess("стоит", mode="lemma") == ["стоить"]


def test_latin_tokens_get_english_stemming():
    assert preprocess("Software Engineering", mode="lemma") == ["softwar", "engin"]
    assert preprocess("Software Engineering", mode="stem") == ["softwar", "engin"]
    assert preprocess("Software Engineering", mode="raw") == ["software", "engineering"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        preprocess("текст", mode="magic")


def test_text_analyzer_ngrams_and_pickle():
    analyzer = TextAnalyzer("lemma", ngram_range=(1, 2))
    grams = analyzer("общежитие для студентов")
    assert grams == ["общежитие", "студент", "общежитие студент"]
    restored = pickle.loads(pickle.dumps(analyzer))
    assert restored("общежитие для студентов") == grams


def test_compare_modes_shrinks_vocabulary():
    texts = ["общежитие общежития общежитием", "студент студенты студентов"]
    table = compare_modes(texts, modes=["raw", "stem", "lemma"]).set_index("mode")
    assert table.loc["raw", "vocab_size"] == 6
    assert table.loc["lemma", "vocab_size"] == 2
    assert {"time_s", "ms_per_doc", "hapax_share"} <= set(table.columns)
