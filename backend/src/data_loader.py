"""Загрузка, аудит и очистка исходного FAQ -> faq_clean.parquet.

    python -m src.data_loader            # то же, что пункт 1 меню main.py

Исходник: data/raw/QA_addmissionAitu.csv (608 строк, колонки questions,
answers, classes). Данные грязные; каждое правило ниже пишет в лог, какие
строки оно задело, а итоговая таблица аудита идёт в отчёт.

Правила (в порядке применения):
    header_row             повтор строки заголовка внутри файла -> удалить
    broken_csv_row         строка, разрезанная запятой: хвост ответа уехал в класс ->
                           склеить; если вопроса нет или ответ оборван -> удалить
    answer_in_question     "Где кабинет 1с?, ответ: ..." -> разнести по колонкам
    question_prefix        "Вопрос 1: ..." -> убрать префикс
    whitespace_quotes      strip, внешние кавычки, литеральные \\n -> пробел
    answer_is_slug         вместо ответа идентификатор вида AITU_iot -> удалить
    translation_not_answer "вопрос" — пункт учебного плана, "ответ" — его перевод -> удалить
    empty_answer           "...не представлена в данном контексте" -> удалить
    duplicate_pair         одинаковая пара вопрос–ответ после нормализации -> оставить первую

Не удаляем, а помечаем:
    answer_cluster  FAQ с одним ответом и перефразы одного вопроса (см. build_answer_clusters)
    topic           одна из 12 тем (src/topic_map.py)
    is_dated        ответ привязан к конкретному году / старому названию столицы
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .preprocessing import normalize
from .topic_map import assign_topic

CLEAN_COLUMNS = ["faq_id", "question", "answer", "orig_class", "topic", "answer_cluster", "is_dated"]

RULE_DESCRIPTIONS = {
    "header_row": "повтор строки заголовка внутри файла",
    "broken_csv_row": "строка разрезана запятой, хвост ответа попал в класс",
    "answer_in_question": "ответ записан внутри вопроса",
    "question_prefix": "служебный префикс «Вопрос N:»",
    "whitespace_quotes": "лишние пробелы, внешние кавычки, литеральные \\n",
    "answer_is_slug": "вместо ответа — идентификатор (AITU_iot)",
    "translation_not_answer": "не вопрос: пункт учебного плана и его перевод на английский",
    "empty_answer": "пустой ответ («не представлена в данном контексте»)",
    "duplicate_pair": "точный дубль пары вопрос–ответ",
    "topic_harmonized": "тема строки заменена на тему её кластера",
}

# --- Регулярные выражения -------------------------------------------------

_SLUG_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")
_PREFIX_RE = re.compile(r"^\s*вопрос\s*\d+\s*[:.)]\s*", re.IGNORECASE)
_ANSWER_IN_Q_RE = re.compile(r"^(?P<q>.*?\?)\s*,?\s*ответ\s*:\s*(?P<a>.+)$", re.IGNORECASE | re.DOTALL)
_EMPTY_ANSWER_RE = re.compile(r"не представлен\w*\s+в\s+данном\s+контексте|^нет информации", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LETTER_RE = re.compile(r"[a-zа-яё]", re.IGNORECASE)
# Устаревание: явные годы 2020–2025, даты, старое название столицы, "на момент ..."
_DATED_RE = re.compile(
    r"\b202[0-5]\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b|нур-султан|на момент|в этом году|в текущем году",
    re.IGNORECASE)

# Короткие ответы ("85 баллов.", "Математика и физика.") повторяются у разных
# специальностей. Склеивать их только по тексту ответа нельзя: Computer Science
# и Software Engineering оказались бы одним кластером. Для коротких ответов
# ключ кластера — пара (ответ, исходный класс).
SHORT_ANSWER_CHARS = 60

# Перефразы одного вопроса с разными формулировками ответа. Подобраны вручную
# по просмотру всех строк (группы по faq_id = индекс строки исходного CSV).
MANUAL_CLUSTER_LINKS: list[tuple[int, ...]] = [
    (78, 79, 80, 81, 92, 105),      # проходной балл Computer Science
    (82, 93, 104),                  # проходной балл Software Engineering
    (83, 94, 101),                  # проходной балл Information Security
    (84, 87, 244),                  # предметы ЕНТ Software Engineering
    (85, 88, 107, 111),             # предметы ЕНТ Information Security
    (258, 652),                     # предметы ЕНТ Computer Science
    (259, 653), (260, 654), (262, 656), (263, 657), (264, 658), (265, 659),
    (266, 660), (267, 661),         # предметы ЕНТ остальных специальностей
    (665, 676), (666, 677), (667, 678), (668, 679), (669, 680), (670, 681),
    (671, 682), (672, 683), (673, 684),  # "что изучают на специальности X" — пары перефразов
    (443, 472),                     # что такое Foundation
    (428, 438),                     # преимущества колледжа
    (674, 685),                     # условия для иностранных студентов
]


# --- Лог очистки ------------------------------------------------------------

@dataclass
class CleaningLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, faq_id, rule: str, action: str, before=None, after=None, detail: str = "") -> None:
        self.entries.append({"faq_id": int(faq_id), "rule": rule, "action": action,
                             "before": before, "after": after, "detail": detail})

    def counts(self) -> Counter:
        return Counter(e["rule"] for e in self.entries)

    def dropped_ids(self) -> set[int]:
        return {e["faq_id"] for e in self.entries if e["action"] == "dropped"}


# --- Загрузка ---------------------------------------------------------------------

def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Исходный CSV как есть: faq_id (индекс строки в файле), questions, answers, classes."""
    path = Path(path or config.RAW_CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Нет исходного файла {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.rename(columns={first: "faq_id"})
    else:
        df.insert(0, "faq_id", range(len(df)))
    missing = {"questions", "answers", "classes"} - set(df.columns)
    if missing:
        raise ValueError(f"В {path.name} нет колонок {sorted(missing)}")
    df["faq_id"] = pd.to_numeric(df["faq_id"], errors="raise").astype("int64")
    return df[["faq_id", "questions", "answers", "classes"]]


def save_raw_parquet(raw: pd.DataFrame, path: str | Path | None = None) -> Path:
    path = Path(path or config.RAW_PARQUET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path, index=False)
    return path


# --- Очистка текста ----------------------------------------------------------

def clean_text(text: object) -> str:
    """strip, литеральные "\\n" и переводы строк -> пробел, "\\\\" -> "/", внешние кавычки."""
    if not isinstance(text, str):
        return ""
    s = text.replace("\\\\", "/").replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.count('"') == 1:                     # непарная кавычка от разрезанной строки
        s = s.replace('"', "").strip()
    return s


def clean_class(text: object) -> str:
    s = clean_text(text)
    return s.strip(" ,;")


def _cyrillic_share(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    return len(_CYRILLIC_RE.findall(text)) / len(letters) if letters else 0.0


def _is_broken_class(raw_class: str) -> bool:
    """Класс-обрывок: начинается с пробела (после запятой-разделителя), не в
    кавычках и выглядит как кусок предложения."""
    stripped = raw_class.strip()
    return (raw_class.startswith(" ") and not stripped.startswith('"')
            and bool(stripped) and (stripped[0].islower() or stripped[-1] in ".:"))


def is_dated(text: str) -> bool:
    return bool(_DATED_RE.search(text or ""))


def clean_rows(raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Построчные правила (всё, кроме дедупликации)."""
    out = []
    for row in raw.itertuples(index=False):
        fid, q_raw, a_raw, c_raw = row.faq_id, row.questions, row.answers, row.classes

        if q_raw.strip().lower() == "questions" and a_raw.strip().lower() == "answers":
            log.add(fid, "header_row", "dropped", before=f"{q_raw} | {a_raw} | {c_raw}")
            continue

        q, a, c = q_raw, a_raw, c_raw
        if _is_broken_class(c_raw):
            glued = f"{a_raw.rstrip()}, {c_raw.strip()}"
            a_glued = clean_text(glued)
            if "?" not in q or a_glued.endswith(":"):
                log.add(fid, "broken_csv_row", "dropped", before={"q": q_raw, "a": a_raw, "c": c_raw},
                        after=a_glued, detail="после склейки нет вопроса или ответ оборван")
                continue
            log.add(fid, "broken_csv_row", "fixed", before=a_raw, after=a_glued)
            a, c = a_glued, ""

        m = _ANSWER_IN_Q_RE.match(q.strip())
        if m:
            new_q, new_a = m.group("q").strip(), m.group("a").strip()
            log.add(fid, "answer_in_question", "fixed", before={"q": q, "a": a},
                    after={"q": new_q, "a": new_a})
            q, a = new_q, new_a

        if _PREFIX_RE.match(q):
            new_q = _PREFIX_RE.sub("", q, count=1)
            log.add(fid, "question_prefix", "fixed", before=q, after=new_q)
            q = new_q

        q_clean, a_clean, c_clean = clean_text(q), clean_text(a), clean_class(c)
        changed = [name for name, before, after in (("question", q, q_clean), ("answer", a, a_clean),
                                                    ("class", c, c_clean)) if before != after]
        if changed:
            log.add(fid, "whitespace_quotes", "fixed", detail=", ".join(changed))
        q, a, c = q_clean, a_clean, c_clean

        if _SLUG_RE.match(a):
            log.add(fid, "answer_is_slug", "dropped", before={"q": q[:120], "a": a})
            continue
        if "?" not in q and _cyrillic_share(q) > 0.5 and _cyrillic_share(a) < 0.3:
            log.add(fid, "translation_not_answer", "dropped", before={"q": q[:120], "a": a[:120]})
            continue
        if not a or _EMPTY_ANSWER_RE.search(a):
            log.add(fid, "empty_answer", "dropped", before={"q": q, "a": a})
            continue

        out.append({"faq_id": fid, "question": q, "answer": a, "orig_class": c})
    return pd.DataFrame(out, columns=["faq_id", "question", "answer", "orig_class"])


def drop_duplicate_pairs(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    key = df["question"].map(normalize) + "\x00" + df["answer"].map(normalize)
    first_id = df.groupby(key, sort=False)["faq_id"].transform("min")
    dup = df["faq_id"] != first_id
    for fid, keep in zip(df.loc[dup, "faq_id"], first_id[dup]):
        log.add(fid, "duplicate_pair", "dropped", detail=f"дубль faq_id={int(keep)}")
    return df[~dup].reset_index(drop=True)


# --- Кластеры ответов ----------------------------------------------------------

class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_answer_clusters(df: pd.DataFrame,
                          manual_links: list[tuple[int, ...]] | None = None) -> pd.Series:
    """answer_cluster для каждой строки.

    Строки попадают в один кластер, если у них:
      * одинаковый нормализованный ответ (для коротких ответов — ещё и класс),
      * одинаковый нормализованный вопрос (тот же вопрос с другим ответом),
      * они перечислены вместе в MANUAL_CLUSTER_LINKS.
    Номер кластера = минимальный faq_id в нём. Он не меняется, пока эта
    строка остаётся в данных, — на номера ссылается eval/qrels.csv.
    """
    ids = df["faq_id"].tolist()
    uf = _UnionFind(ids)
    groups: dict[tuple, int] = {}
    for fid, q, a, c in zip(ids, df["question"], df["answer"], df["orig_class"]):
        a_norm = normalize(a)
        a_key = ("a", a_norm) if len(a_norm) >= SHORT_ANSWER_CHARS else ("a", a_norm, normalize(c))
        for key in (a_key, ("q", normalize(q))):
            if key in groups:
                uf.union(fid, groups[key])
            else:
                groups[key] = fid
    present = set(ids)
    for group in (MANUAL_CLUSTER_LINKS if manual_links is None else manual_links):
        members = [g for g in group if g in present]
        for other in members[1:]:
            uf.union(members[0], other)
    return pd.Series([uf.find(i) for i in ids], index=df.index, name="answer_cluster")


def harmonize_cluster_topics(df: pd.DataFrame, log: CleaningLog) -> pd.Series:
    """Одна тема на кластер: самая частая тема его строк (при ничьей — тема
    строки с минимальным faq_id). Иначе один и тот же ответ числился бы в двух
    темах, а у запроса в qrels.csv тема одна."""
    def cluster_topic(topics: pd.Series) -> str:
        counts = topics.value_counts()
        tied = set(counts[counts == counts.iloc[0]].index)
        return next(t for t in topics if t in tied)   # topics идут по возрастанию faq_id

    majority = df.sort_values("faq_id").groupby("answer_cluster")["topic"].agg(cluster_topic)
    new_topic = df["answer_cluster"].map(majority)
    for fid, old, new in zip(df["faq_id"], df["topic"], new_topic):
        if old != new:
            log.add(fid, "topic_harmonized", "fixed", before=old, after=new)
    return new_topic


# --- Типы и память --------------------------------------------------------------

def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 2**20)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """int64 -> int32, тема -> category (12 значений), флаг -> bool."""
    out = df.copy()
    out["faq_id"] = out["faq_id"].astype("int32")
    out["answer_cluster"] = out["answer_cluster"].astype("int32")
    out["topic"] = pd.Categorical(out["topic"], categories=list(config.TOPICS))
    out["is_dated"] = out["is_dated"].astype(bool)
    return out


# --- Весь этап ---------------------------------------------------------------------

def build_clean_faq(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog, pd.DataFrame]:
    """raw -> (faq_clean, лог, таблица ручной проверки тем)."""
    log = CleaningLog()
    df = clean_rows(raw, log)
    df = drop_duplicate_pairs(df, log)

    topics = [assign_topic(q, c, fid) for fid, q, c in zip(df["faq_id"], df["question"], df["orig_class"])]
    df["topic"] = [t for t, _ in topics]
    df["answer_cluster"] = build_answer_clusters(df)
    df["topic"] = harmonize_cluster_topics(df, log)
    df["is_dated"] = (df["question"] + " " + df["answer"]).map(is_dated)

    sources = dict(zip(df["faq_id"], (s for _, s in topics)))
    harmonized = {e["faq_id"] for e in log.entries if e["rule"] == "topic_harmonized"}
    review = df[["faq_id", "topic", "question", "orig_class"]].copy()
    review.insert(2, "topic_source", [("cluster" if f in harmonized else sources[f]) for f in df["faq_id"]])
    return df[CLEAN_COLUMNS], log, review


def audit_table(raw: pd.DataFrame, clean: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Сколько строк задело каждое правило — таблица для отчёта."""
    rows = []
    for rule, desc in RULE_DESCRIPTIONS.items():
        entries = [e for e in log.entries if e["rule"] == rule]
        rows.append({"rule": rule, "description": desc,
                     "dropped": sum(e["action"] == "dropped" for e in entries),
                     "fixed": sum(e["action"] == "fixed" for e in entries)})
    cluster_sizes = clean["answer_cluster"].value_counts()
    rows += [
        {"rule": "answer_cluster", "description": "строк в кластерах из 2+ FAQ (не удаляются)",
         "dropped": 0, "fixed": int(cluster_sizes[cluster_sizes > 1].sum())},
        {"rule": "is_dated", "description": "ответ с датой/годом/«Нур-Султан» (помечен флагом)",
         "dropped": 0, "fixed": int(clean["is_dated"].sum())},
        {"rule": "no_question_mark", "description": "вопрос без «?» (заголовок вида «Устав»), оставлен",
         "dropped": 0, "fixed": int((~clean["question"].str.contains("?", regex=False)).sum())},
    ]
    return pd.DataFrame(rows)


def cluster_review(clean: pd.DataFrame) -> pd.DataFrame:
    sizes = clean.groupby("answer_cluster")["faq_id"].transform("size")
    return (clean[sizes > 1].sort_values(["answer_cluster", "faq_id"])
            [["answer_cluster", "faq_id", "topic", "question", "answer"]])


def prepare_data(raw_path: str | Path | None = None, out_path: str | Path | None = None,
                 log_path: str | Path | None = None, write_reviews: bool = True,
                 verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Пункт 1 меню: исходник -> raw parquet, faq_clean.parquet, cleaning_log.json, аудит."""
    raw = load_raw(raw_path)
    out_path = Path(out_path or config.FAQ_CLEAN_PATH)
    log_path = Path(log_path or config.CLEANING_LOG_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path is None:
        save_raw_parquet(raw)

    clean, log, review = build_clean_faq(raw)
    mem_before = memory_mb(clean)
    clean = optimize_dtypes(clean)
    mem_after = memory_mb(clean)
    clean.to_parquet(out_path, index=False)

    audit = audit_table(raw, clean, log)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": Path(raw_path or config.RAW_CSV_PATH).name,
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "rows_dropped": int(len(log.dropped_ids())),
        "answer_clusters": int(clean["answer_cluster"].nunique()),
        "clusters_with_2plus": int((clean["answer_cluster"].value_counts() > 1).sum()),
        "orig_classes_raw": int(raw["classes"].nunique()),
        "topics": {str(k): int(v) for k, v in clean["topic"].value_counts().items()},
        "memory_mb": {"before_optimize": round(mem_before, 4), "after_optimize": round(mem_after, 4)},
        "dtypes": {c: str(t) for c, t in clean.dtypes.items()},
        "audit": audit.to_dict(orient="records"),
        "entries": log.entries,
    }
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if write_reviews:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        audit.to_csv(config.METRICS_DIR / "cleaning_audit.csv", index=False)
        review.to_csv(config.TOPIC_REVIEW_PATH, index=False)
        cluster_review(clean).to_csv(config.CLUSTER_REVIEW_PATH, index=False)

    if verbose:
        print_summary(summary, audit)
    return clean, summary


def print_summary(summary: dict, audit: pd.DataFrame) -> None:
    print(f"Исходных строк: {summary['rows_raw']}, после очистки: {summary['rows_clean']} "
          f"(удалено {summary['rows_dropped']})")
    print(f"Кластеров ответов: {summary['answer_clusters']} "
          f"(из них с 2+ вопросами: {summary['clusters_with_2plus']})")
    print(f"Исходных классов: {summary['orig_classes_raw']} -> тем: {len(summary['topics'])}")
    m = summary["memory_mb"]
    print(f"Память DataFrame: {m['before_optimize']:.3f} МБ -> {m['after_optimize']:.3f} МБ "
          f"после оптимизации типов")
    with pd.option_context("display.width", 160, "display.max_colwidth", 70):
        print("\nАудит:")
        print(audit.to_string(index=False))
        print("\nТемы:")
        for slug, n in summary["topics"].items():
            print(f"  {config.TOPICS.get(slug, slug):<35} {n}")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Очистка FAQ AITU -> faq_clean.parquet")
    parser.add_argument("--raw", default=None, help="по умолчанию data/raw/QA_addmissionAitu.csv")
    parser.add_argument("--out", default=None, help="по умолчанию data/processed/faq_clean.parquet")
    args = parser.parse_args(argv)
    prepare_data(args.raw, args.out)


if __name__ == "__main__":
    main()
