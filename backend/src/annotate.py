"""Разметка тестового набора (протокол A): запрос абитуриента -> gold answer_cluster.

    python -m src.annotate --author ayan                 # интерактивная разметка
    python -m src.annotate --import eval/drafts/x.csv    # добавить проверенный черновик
    python -m src.annotate --check                       # проверка и статистика набора
    python -m src.annotate --freeze                      # ОДИН раз: dev 30% / test 70%
    python -m src.annotate --agreement eval/qrels_review.csv   # Cohen's kappa

Кандидаты для разметки собираются пулом из нескольких методов (char TF-IDF,
BM25, e5 — если модель скачана) и показываются вперемешку, без указания,
какой метод что нашёл. Так разметка меньше зависит от одного метода. Ответ,
которого нет среди кандидатов, ищется по подстроке командой "f <слово>".
Правила разметки — eval/GUIDELINES.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from . import config
from .evaluation import annotator_agreement, assign_split, load_faq, validate_eval_set
from .preprocessing import normalize
from .retrievers import BaseRetriever, build_retriever

QUERY_COLS = ["query_id", "text", "author", "split", "is_oos", "origin"]
QREL_COLS = ["query_id", "answer_cluster", "topic"]
UNASSIGNED = "unassigned"          # split до заморозки набора
ORIGINS = ("manual", "llm_draft")  # llm_draft — черновик LLM, проверенный человеком


# --- файлы -------------------------------------------------------------------

def read_eval_files(queries_path: str | Path | None = None,
                    qrels_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Текущий набор; если файлов ещё нет — пустые таблицы нужного формата."""
    qp = Path(queries_path or config.TEST_QUERIES_PATH)
    rp = Path(qrels_path or config.QRELS_PATH)
    queries = pd.read_csv(qp) if qp.exists() else pd.DataFrame(columns=QUERY_COLS)
    qrels = pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=QREL_COLS)
    if "origin" not in queries:
        queries["origin"] = "manual"
    return queries, qrels


def write_eval_files(queries: pd.DataFrame, qrels: pd.DataFrame,
                     queries_path: str | Path | None = None,
                     qrels_path: str | Path | None = None) -> None:
    qp = Path(queries_path or config.TEST_QUERIES_PATH)
    rp = Path(qrels_path or config.QRELS_PATH)
    qp.parent.mkdir(parents=True, exist_ok=True)
    queries[QUERY_COLS].to_csv(qp, index=False)
    qrels[QREL_COLS].to_csv(rp, index=False)


def _is_frozen(queries: pd.DataFrame) -> bool:
    return bool(len(queries)) and queries["split"].isin(["dev", "test"]).any()


# --- добавление запросов --------------------------------------------------------

def cluster_topic(faq_df: pd.DataFrame, cluster: int) -> str | None:
    """Тема кластера (самая частая среди его FAQ)."""
    topics = faq_df.loc[faq_df["answer_cluster"] == cluster, "topic"]
    return str(topics.mode().iloc[0]) if len(topics) else None


def add_query(queries: pd.DataFrame, qrels: pd.DataFrame, faq_df: pd.DataFrame, text: str,
              author: str, clusters: Sequence[int] = (), topic: str | None = None,
              is_oos: bool = False, origin: str = "manual") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Добавить один запрос с проверками. Возвращает (queries, qrels, query_id)."""
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("Пустой запрос")
    if not author:
        raise ValueError("Не указан автор")
    if origin not in ORIGINS:
        raise ValueError(f"origin должен быть одним из {ORIGINS}")
    if _is_frozen(queries):
        raise ValueError("Набор уже заморожен (--freeze): новые запросы не добавляются")
    norm = normalize(text)
    if norm in set(queries["text"].map(normalize)):
        raise ValueError(f"Такой запрос уже есть: {text!r}")
    if norm in set(faq_df["question"].map(normalize)):
        raise ValueError("Запрос дословно совпадает с вопросом FAQ — это утечка, перефразируйте")

    clusters = sorted({int(c) for c in clusters})
    if is_oos:
        if clusters:
            raise ValueError("У запроса не по теме не должно быть gold-кластеров")
        topic = None
    else:
        if not clusters:
            raise ValueError("Нужен хотя бы один gold answer_cluster (или пометка 'не по теме')")
        unknown = set(clusters) - set(faq_df["answer_cluster"].astype(int))
        if unknown:
            raise ValueError(f"Нет таких answer_cluster: {sorted(unknown)}")
        topic = topic or cluster_topic(faq_df, clusters[0])
        if topic not in config.TOPICS:
            raise ValueError(f"Тема {topic!r} не из config.TOPICS")

    qid = int(queries["query_id"].max()) + 1 if len(queries) else 1
    row = {"query_id": qid, "text": text, "author": author, "split": UNASSIGNED,
           "is_oos": int(is_oos), "origin": origin}
    queries = pd.concat([queries, pd.DataFrame([row])], ignore_index=True)
    if clusters:
        new = pd.DataFrame({"query_id": qid, "answer_cluster": clusters, "topic": topic})
        qrels = pd.concat([qrels, new], ignore_index=True)
    return queries, qrels, qid


def _parse_clusters(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(float(x)) for x in str(value).replace(",", "|").split("|") if x.strip()]


def import_drafts(path: str | Path, faq_df: pd.DataFrame,
                  queries_path: str | Path | None = None,
                  qrels_path: str | Path | None = None) -> int:
    """Добавить проверенный черновик: колонки text, author, is_oos, answer_cluster
    ("12" или "12|15"), topic, origin. Всё или ничего: при любой ошибке набор
    не меняется. Строки с reviewed=0 (если колонка есть) пропускаются."""
    drafts = pd.read_csv(path)
    if "reviewed" in drafts:
        skipped = int((drafts["reviewed"].astype(int) == 0).sum())
        drafts = drafts[drafts["reviewed"].astype(int) == 1]
        if skipped:
            print(f"Пропущено непроверенных строк (reviewed=0): {skipped}")
    queries, qrels = read_eval_files(queries_path, qrels_path)
    errors = []
    for i, r in drafts.iterrows():
        try:
            queries, qrels, _ = add_query(
                queries, qrels, faq_df, r["text"], r["author"], _parse_clusters(r.get("answer_cluster")),
                r.get("topic") if isinstance(r.get("topic"), str) else None,
                is_oos=str(r.get("is_oos", 0)).strip().lower() in {"1", "true", "да"},
                origin=r.get("origin", "manual") if isinstance(r.get("origin"), str) else "manual")
        except ValueError as exc:
            errors.append(f"строка {i + 2}: {exc}")
    if errors:
        raise ValueError("Черновик не добавлен:\n  " + "\n  ".join(errors))
    write_eval_files(queries, qrels, queries_path, qrels_path)
    return len(drafts)


# --- кандидаты ---------------------------------------------------------------------

def build_pool_retrievers(faq_df: pd.DataFrame, use_neural: bool = True) -> list[BaseRetriever]:
    names = ["tfidf_char", "bm25"] + (["e5_small"] if use_neural else [])
    out = []
    for name in names:
        try:
            out.append(build_retriever(name).fit(faq_df))
        except (OSError, RuntimeError) as exc:  # нет сети для скачивания модели
            print(f"  {name} недоступен ({exc}), кандидаты без него")
    return out


def candidate_pool(retrievers: Sequence[BaseRetriever], faq_df: pd.DataFrame, query: str,
                   per_method: int = 5) -> pd.DataFrame:
    """Разные кластеры из топов нескольких методов, по очереди (round-robin)."""
    faq = faq_df.set_index("faq_id")
    lists = []
    for r in retrievers:
        seen, items = set(), []
        for fid, _ in r.search(query, k=50):
            c = int(faq.loc[fid, "answer_cluster"])
            if c not in seen:
                seen.add(c)
                items.append(fid)
            if len(items) == per_method:
                break
        lists.append(items)
    pooled, clusters = [], set()
    for rank in range(per_method):
        for items in lists:
            if rank < len(items):
                fid = items[rank]
                c = int(faq.loc[fid, "answer_cluster"])
                if c not in clusters:
                    clusters.add(c)
                    pooled.append(fid)
    return _cluster_view(faq_df, pooled)


def search_faq(faq_df: pd.DataFrame, needle: str, limit: int = 10) -> pd.DataFrame:
    """Поиск по подстроке в вопросе и ответе (без учёта регистра)."""
    n = normalize(needle)
    hay = (faq_df["question"] + " " + faq_df["answer"]).map(normalize)
    ids = faq_df.loc[hay.str.contains(n, regex=False), "faq_id"].tolist()
    view = _cluster_view(faq_df, ids)
    return view.head(limit)


def _cluster_view(faq_df: pd.DataFrame, faq_ids: Sequence) -> pd.DataFrame:
    faq = faq_df.set_index("faq_id")
    rows, seen = [], set()
    for fid in faq_ids:
        c = int(faq.loc[fid, "answer_cluster"])
        if c in seen:
            continue
        seen.add(c)
        rows.append({"answer_cluster": c, "topic": str(faq.loc[fid, "topic"]),
                     "question": faq.loc[fid, "question"], "answer": faq.loc[fid, "answer"]})
    return pd.DataFrame(rows, columns=["answer_cluster", "topic", "question", "answer"])


def _show_candidates(view: pd.DataFrame, print_fn: Callable[[str], None]) -> None:
    if view.empty:
        print_fn("  (ничего не найдено)")
    for i, r in enumerate(view.itertuples(index=False), start=1):
        answer = r.answer if len(r.answer) <= 110 else r.answer[:107] + "..."
        print_fn(f"  {i:>2}. [кластер {r.answer_cluster}, {r.topic}] {r.question}\n      -> {answer}")


# --- интерактив ----------------------------------------------------------------------

HELP = ("Номер(а) через запятую — выбрать ответ | f <слово> — поиск по FAQ | "
        "o — не по теме | s — пропустить | q — выход")


class _Quit(Exception):
    pass


def _select(text: str, retrievers, faq_df: pd.DataFrame, input_fn, print_fn):
    """Показать кандидатов и спросить выбор.

    Returns: (clusters, topic) — выбран ответ; ([], None) — не по теме;
    None — пропустить. Выход из разметки — исключение _Quit.
    """
    view = candidate_pool(retrievers, faq_df, text)
    _show_candidates(view, print_fn)
    print_fn(HELP)
    while True:
        try:
            cmd = input_fn("Выбор: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise _Quit
        low = cmd.lower()
        if low == "q":
            raise _Quit
        if low == "s":
            return None
        if low == "o":
            return [], None
        if low.startswith("f "):
            view = search_faq(faq_df, cmd[2:])
            _show_candidates(view, print_fn)
            continue
        try:
            picks = [int(x) for x in cmd.replace(" ", "").split(",") if x]
        except ValueError:
            picks = []
        if not picks or any(p < 1 or p > len(view) for p in picks):
            print_fn(f"  Ошибка: введите номера от 1 до {len(view)} или команду")
            continue
        clusters = [int(view.iloc[p - 1]["answer_cluster"]) for p in picks]
        default_topic = str(view.iloc[picks[0] - 1]["topic"])
        try:
            topic = input_fn(f"Тема [{default_topic}]: ").strip() or default_topic
        except (EOFError, KeyboardInterrupt):
            raise _Quit
        if topic not in config.TOPICS:
            print_fn(f"  Ошибка: тема должна быть одной из: {', '.join(config.TOPICS)}")
            continue
        return clusters, topic


def interactive(author: str, faq_df: pd.DataFrame, use_neural: bool = True,
                input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                queries_path: str | Path | None = None, qrels_path: str | Path | None = None,
                retrievers: Sequence[BaseRetriever] | None = None) -> int:
    """Диалог разметки новых запросов; каждый сохраняется сразу. Возвращает число добавленных."""
    retrievers = retrievers if retrievers is not None else build_pool_retrievers(faq_df, use_neural)
    added = 0
    print_fn(f"Разметка от имени {author!r}. Пустой ввод — выход.")
    while True:
        try:
            text = input_fn("\nЗапрос абитуриента: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        try:
            choice = _select(text, retrievers, faq_df, input_fn, print_fn)
        except _Quit:
            break
        if choice is None:
            continue
        clusters, topic = choice
        try:
            queries, qrels = read_eval_files(queries_path, qrels_path)
            queries, qrels, qid = add_query(queries, qrels, faq_df, text, author, clusters, topic,
                                            is_oos=not clusters)
            write_eval_files(queries, qrels, queries_path, qrels_path)
        except ValueError as exc:
            print_fn(f"  Ошибка: {exc}")
            continue
        added += 1
        print_fn(f"  сохранено: query_id {qid}")
    print_fn(f"Добавлено запросов: {added}")
    return added


def review_path(reviewer: str, author: str) -> Path:
    return config.EVAL_DIR / "reviews" / f"qrels_review_{reviewer}_on_{author}.csv"


def review(reviewer: str, author: str, faq_df: pd.DataFrame, use_neural: bool = True,
           input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
           out_path: str | Path | None = None, retrievers: Sequence[BaseRetriever] | None = None,
           queries_path: str | Path | None = None) -> Path:
    """Независимая разметка чужих запросов для Cohen's kappa.

    Разметка автора НЕ показывается. Результат — eval/reviews/qrels_review_<кто>_on_<чей>.csv
    (query_id, answer_cluster, topic; пустой кластер = "не по теме"). Можно прерваться
    и продолжить: уже размеченные запросы пропускаются.
    """
    queries, _ = read_eval_files(queries_path)
    todo = queries[queries["author"] == author]
    if todo.empty:
        raise ValueError(f"У автора {author!r} нет запросов")
    out = Path(out_path or review_path(reviewer, author))
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=QREL_COLS)
    todo = todo[~todo["query_id"].isin(set(done["query_id"]))]
    retrievers = retrievers if retrievers is not None else build_pool_retrievers(faq_df, use_neural)
    print_fn(f"{reviewer} проверяет запросы {author}: осталось {len(todo)}")
    for row in todo.itertuples(index=False):
        print_fn(f"\n[{row.query_id}] {row.text}")
        try:
            choice = _select(row.text, retrievers, faq_df, input_fn, print_fn)
        except _Quit:
            break
        if choice is None:
            continue
        clusters, topic = choice
        new = (pd.DataFrame({"query_id": row.query_id, "answer_cluster": clusters, "topic": topic})
               if clusters else pd.DataFrame([{"query_id": row.query_id, "answer_cluster": None,
                                               "topic": None}]))
        done = pd.concat([done, new], ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        done[QREL_COLS].to_csv(out, index=False)
    print_fn(f"Разметка сохранена: {out}")
    return out


# --- проверка, заморозка, согласие ------------------------------------------------

def check(faq_df: pd.DataFrame, queries_path: str | Path | None = None,
          qrels_path: str | Path | None = None) -> dict:
    """Проверка формата и статистика: по авторам, темам, доля LLM, состояние split."""
    queries, qrels = read_eval_files(queries_path, qrels_path)
    if queries.empty:
        print("Тестовый набор пуст.")
        return {"n_queries": 0}
    validate_eval_set(queries, qrels, faq_df)
    oos = queries["is_oos"].astype(int).astype(bool)
    stats = {
        "n_queries": len(queries),
        "n_oos": int(oos.sum()),
        "llm_share": float((queries["origin"] == "llm_draft").mean()),
        "frozen": _is_frozen(queries),
    }
    print(f"Запросов: {stats['n_queries']} (не по теме: {stats['n_oos']}), "
          f"доля черновиков LLM: {stats['llm_share']:.0%}, заморожен: {'да' if stats['frozen'] else 'нет'}")
    print("\nПо авторам:")
    print(queries.groupby(["author", "is_oos"]).size().unstack(fill_value=0).to_string())
    topics = qrels.drop_duplicates("query_id")["topic"].value_counts()
    print("\nПо темам:")
    print(topics.reindex(list(config.TOPICS), fill_value=0).to_string())
    if stats["llm_share"] > 0.5:
        print("\nВНИМАНИЕ: черновиков LLM больше половины — по плану не больше 50%.")
    missing = [t for t in config.TOPICS if topics.get(t, 0) == 0]
    if missing:
        print(f"\nТемы без запросов: {', '.join(missing)}")
    if stats["frozen"]:
        print("\nSplit:", queries["split"].value_counts().to_dict())
    return stats


def freeze(dev_frac: float = 0.3, force: bool = False, queries_path: str | Path | None = None,
           qrels_path: str | Path | None = None) -> pd.Series:
    """Разметить dev/test со стратификацией по темам. Делается один раз."""
    queries, qrels = read_eval_files(queries_path, qrels_path)
    if queries.empty:
        raise ValueError("Тестовый набор пуст")
    if _is_frozen(queries) and not force:
        raise ValueError("Набор уже заморожен. Повторная заморозка меняет dev/test — "
                         "только осознанно, с --force")
    queries = assign_split(queries, qrels, dev_frac=dev_frac)
    write_eval_files(queries, qrels, queries_path, qrels_path)
    return queries["split"].value_counts()


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Разметка тестового набора")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--import", dest="import_path", metavar="CSV", help="добавить черновик")
    group.add_argument("--check", action="store_true", help="проверить набор и показать статистику")
    group.add_argument("--freeze", action="store_true", help="разметить dev/test (один раз)")
    group.add_argument("--agreement", metavar="CSV", help="kappa с разметкой второго человека")
    group.add_argument("--review", metavar="АВТОР", help="независимо разметить запросы этого автора")
    parser.add_argument("--author", help="ваше имя (автор новых запросов или проверяющий)")
    parser.add_argument("--no-neural", action="store_true", help="кандидаты без e5")
    parser.add_argument("--force", action="store_true", help="для --freeze: перезаморозить")
    args = parser.parse_args(argv)

    if args.freeze:
        print("Split:", freeze(force=args.force).to_dict())
        return
    faq_df = load_faq()
    if args.import_path:
        print(f"Добавлено запросов: {import_drafts(args.import_path, faq_df)}")
    elif args.check:
        check(faq_df)
    elif args.agreement:
        _, qrels = read_eval_files()
        other = pd.read_csv(args.agreement)
        metrics, diff = annotator_agreement(qrels, other, query_ids=sorted(set(other["query_id"])))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
        if not diff.empty:
            print("\nРасхождения (обсудить и исправить qrels.csv):")
            print(diff.to_string(index=False))
    elif args.review:
        if not args.author:
            parser.error("укажите --author (кто проверяет)")
        path = review(args.author, args.review, faq_df, use_neural=not args.no_neural)
        print(f"Дальше: python -m src.annotate --agreement {path}")
    else:
        if not args.author:
            parser.error("для интерактивной разметки укажите --author")
        interactive(args.author, faq_df, use_neural=not args.no_neural)


if __name__ == "__main__":
    main()
