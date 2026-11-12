import pandas as pd

from src.entities import entity_summary, extract_entities, gazetteer_entities, regex_entities


def types_of(text):
    return {(t, e) for t, e in regex_entities(text)}


def test_contacts():
    found = types_of("Тел.: +7 (7172) 645-710, +7 7172 70 91 05, почта info@astanait.edu.kz, "
                     "сайт https://admission.astanait.edu.kz/aet.")
    assert ("+7 (7172) 645-710", "PHONE") in found
    assert ("+7 7172 70 91 05", "PHONE") in found
    assert ("info@astanait.edu.kz", "EMAIL") in found
    assert ("https://admission.astanait.edu.kz/aet", "URL") in found
    assert not any(e == "URL" and "@" in t for t, e in found)    # почта не дублируется как сайт


def test_money_and_dates():
    found = types_of("Стоимость 2,500,000 (два миллиона пятьсот тысяч) тенге, общежитие 30000 тенге. "
                     "Приём до 25 августа 2023 года, лекция 08.12.2022, в 2022-2023 учебном году.")
    assert ("2,500,000 (два миллиона пятьсот тысяч) тенге", "MONEY") in found
    assert ("30000 тенге", "MONEY") in found
    assert ("25 августа 2023 года", "DATE") in found
    assert ("08.12.2022", "DATE") in found
    assert any(e == "DATE" and t.startswith("2022-2023") for t, e in found)


def test_room_number_is_not_a_date():
    assert not [t for t, e in regex_entities("Кабинет С.1.1.239") if e == "DATE"]


def test_gazetteer():
    found = set(gazetteer_entities("Astana IT University находится в Астане, Казахстан"))
    assert ("Astana IT University", "ORG") in found and ("Казахстан", "LOC") in found


def test_extract_entities_dedup(faq_df):
    df = pd.DataFrame({"faq_id": [1, 2], "question": ["Контакты?", "Цена?"],
                       "answer": ["info@astanait.edu.kz или info@astanait.edu.kz", "1900 тенге"]})
    ents = extract_entities(df, use_ner=False)
    assert len(ents[(ents.faq_id == 1) & (ents.type == "EMAIL")]) == 1
    assert ents[ents.faq_id == 2]["type"].tolist() == ["MONEY"]
    summary = entity_summary(ents)
    assert summary.set_index("type").loc["EMAIL", "n"] == 1
