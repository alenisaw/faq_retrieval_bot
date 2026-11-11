"""Оценка методов поиска: метрики качества, ресурсы, протоколы A и B.

Протокол A (основной): ручные запросы из eval/test_queries.csv + qrels.csv.
Протокол B (доп.): leave-one-out по перефразам внутри answer_cluster.

Метрики считаются на уровне answer_cluster. Выдача сначала сворачивается
по кластерам (пользователь видит разные ответы, а не 4 перефраза одного),
после чего у запроса обычно ровно один релевантный элемент:
    P@1 = Hit@1, Hit@k (= Recall@k при одном релевантном кластере), MRR@10.
P@3/P@5 по стандартному определению при одном релевантном кластере не
больше 1/k — их приводим для полноты, сравниваем методы по MRR и Hit@k.

Ресурсы (требование 9): время построения индекса, задержка на запрос
(средняя и p95), пиковая память Python (tracemalloc), прирост RSS процесса,
размер индекса и весов модели в МБ.

Запуск из консоли:
    python -m src.evaluation --protocol loo --methods lexical
    python -m src.evaluation --protocol test --methods all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config
from .preprocessing import normalize
from .retrievers import LEXICAL_METHODS, METHODS, BaseRetriever, build_retriever

KS = (1, 3, 5)
MRR_K = 10
TOP_N = 50  # сколько FAQ запрашивать, чтобы после свёртки осталось >= 10 кластеров


# --- Загрузка и проверка данных -------------------------------------------

def validate_faq(faq_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.FAQ_REQUIRED_COLUMNS if c not in faq_df.columns]
    if missing:
        raise ValueError(f"В FAQ нет колонок {missing}. Ожидается схема faq_clean.parquet: "
                         f"{', '.join(config.FAQ_REQUIRED_COLUMNS + config.FAQ_OPTIONAL_COLUMNS)}")
    if faq_df["faq_id"].duplicated().any():
        raise ValueError("faq_id должен быть уникальным")
    return faq_df.reset_index(drop=True)


def load_faq(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or config.FAQ_CLEAN_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. Сначала выполните загрузку и очистку данных "
                                f"(пункт 1 меню main.py).")
    return validate_faq(pd.read_parquet(path))


def load_eval_set(queries_path: str | Path | None = None,
                  qrels_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """test_queries.csv (query_id, text, author, split, is_oos) и qrels.csv."""
    queries_path = queries_path or config.TEST_QUERIES_PATH
    qrels_path = qrels_path or config.QRELS_PATH
    for p in (queries_path, qrels_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Нет файла {p}: тестовый набор ещё не собран")
    queries = pd.read_csv(queries_path)
    qrels = pd.read_csv(qrels_path)
    return validate_eval_set(queries, qrels)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "да"})


def validate_eval_set(queries: pd.DataFrame, qrels: pd.DataFrame,
                      faq_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ("query_id", "text"):
        if col not in queries.columns:
            raise ValueError(f"В test_queries.csv нет колонки {col!r}")
    for col in ("query_id", "answer_cluster"):
        if col not in qrels.columns:
            raise ValueError(f"В qrels.csv нет колонки {col!r}")
    queries = queries.copy()
    queries["is_oos"] = _as_bool(queries["is_oos"]) if "is_oos" in queries else False
    if "split" not in queries:
        queries["split"] = "test"
    if queries["query_id"].duplicated().any():
        raise ValueError("query_id в test_queries.csv повторяются")

    in_domain = set(queries.loc[~queries["is_oos"], "query_id"])
    no_gold = in_domain - set(qrels["query_id"])
    if no_gold:
        raise ValueError(f"Нет разметки в qrels для запросов: {sorted(no_gold)[:10]}")
    oos_with_gold = set(queries.loc[queries["is_oos"], "query_id"]) & set(qrels["query_id"])
    if oos_with_gold:
        raise ValueError(f"Запросы не по теме не должны иметь qrels: {sorted(oos_with_gold)[:10]}")
    if faq_df is not None:
        unknown = set(qrels["answer_cluster"]) - set(faq_df["answer_cluster"])
        if unknown:
            raise ValueError(f"В qrels есть кластеры, которых нет в FAQ: {sorted(unknown)[:10]}")
    return queries, qrels


def assign_split(queries: pd.DataFrame, qrels: pd.DataFrame, dev_frac: float = 0.3,
                 seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Разметить split = dev/test со стратификацией по теме (OOS — отдельная страта).

    Вызывается один раз при заморозке тестового набора; результат сохраняется
    в test_queries.csv и больше не меняется. Страты из одного запроса идут в test.
    """
    queries = queries.copy()
    is_oos = _as_bool(queries["is_oos"]) if "is_oos" in queries else pd.Series(False, index=queries.index)
    topic_by_query = (qrels.drop_duplicates("query_id").set_index("query_id")["topic"]
                      if "topic" in qrels else pd.Series(dtype=object))
    strata = np.where(is_oos, "__oos__",
                      queries["query_id"].map(topic_by_query).fillna("__unknown__"))
    rng = np.random.default_rng(seed)
    split = np.full(len(queries), "test", dtype=object)
    for stratum in sorted(set(strata)):
        idx = np.flatnonzero(strata == stratum)
        if len(idx) < 2:
            continue
        n_dev = int(round(dev_frac * len(idx)))
        split[rng.permutation(idx)[:n_dev]] = "dev"
    queries["split"] = split
    return queries


# --- Метрики ----------------------------------------------------------------

def collapse_to_clusters(ranked_faq_ids: Iterable, faq_to_cluster: dict) -> list:
    """FAQ-выдача -> список разных кластеров в порядке первого появления."""
    seen, out = set(), []
    for fid in ranked_faq_ids:
        c = faq_to_cluster[fid]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def hit_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return float(any(c in relevant for c in ranked[:k]))


def precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return sum(c in relevant for c in ranked[:k]) / k


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return sum(c in relevant for c in ranked[:k]) / len(relevant) if relevant else 0.0


def first_relevant_rank(ranked: Sequence, relevant: set) -> int | None:
    for i, c in enumerate(ranked, start=1):
        if c in relevant:
            return i
    return None


def reciprocal_rank(ranked: Sequence, relevant: set, k: int = MRR_K) -> float:
    r = first_relevant_rank(ranked[:k], relevant)
    return 1.0 / r if r else 0.0


# --- Ресурсы ----------------------------------------------------------------

def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def fit_with_stats(retriever: BaseRetriever, faq_df: pd.DataFrame,
                   measure_memory: bool = True) -> dict:
    """fit + замеры. Время меряется без tracemalloc (он замедляет Python в 2–3 раза),
    память — отдельным повторным fit под tracemalloc.

    tracemalloc видит только аллокации Python/NumPy, но не веса PyTorch,
    поэтому дополнительно пишем прирост RSS процесса за первый fit.
    """
    gc.collect()
    rss_before = _rss_mb()
    start = time.perf_counter()
    retriever.fit(faq_df)
    index_time = time.perf_counter() - start
    rss_after = _rss_mb()

    peak_mb = None
    if measure_memory:
        gc.collect()
        tracemalloc.start()
        try:
            retriever.fit(faq_df)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peak_mb = peak / 2**20

    return {
        "index_time_s": index_time,
        "model_load_s": float(getattr(retriever, "model_load_s", 0.0)),
        "peak_mem_mb": peak_mb,
        "rss_delta_mb": (rss_after - rss_before) if rss_before is not None else None,
        "index_size_mb": retriever.index_size_mb(),
        "model_size_mb": retriever.model_size_mb(),
    }


# --- Прогон запросов ----------------------------------------------------------

@dataclass
class RunResult:
    """Результат одного прогона метода. Формат совпадает с таблицами faq.db:
    run_record() -> runs, results -> results."""

    method: str
    protocol: str
    params: dict
    metrics: dict
    per_query: pd.DataFrame
    results: pd.DataFrame
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def run_record(self) -> dict:
        m = self.metrics
        return {"method": self.method, "protocol": self.protocol, "created_at": self.created_at,
                "params_json": json.dumps(self.params, ensure_ascii=False, default=str),
                "index_time_s": m.get("index_time_s"), "peak_mem_mb": m.get("peak_mem_mb"),
                "model_size_mb": m.get("model_size_mb")}


def _search_each(retriever: BaseRetriever, queries: Sequence[str],
                 mask: np.ndarray | None, top_n: int) -> tuple[list[list[tuple]], list[float]]:
    """Каждый запрос отдельно, как в чат-боте, — чтобы замер задержки был честным."""
    ranked, latencies = [], []
    for i, q in enumerate(queries):
        row_mask = None if mask is None else mask[i:i + 1]
        start = time.perf_counter()
        scores = retriever.score_matrix([q], row_mask)[0]
        hits = retriever.rank(scores, top_n)
        latencies.append(1000 * (time.perf_counter() - start))
        ranked.append(hits)
    return ranked, latencies


def _per_query_table(query_ids, texts, gold_sets, gold_topics, is_oos, ranked, latencies,
                     faq_df: pd.DataFrame, ks=KS, mrr_k=MRR_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    faq_to_cluster = dict(zip(faq_df["faq_id"], faq_df["answer_cluster"]))
    faq_to_topic = dict(zip(faq_df["faq_id"], faq_df["topic"])) if "topic" in faq_df else {}
    faq_to_question = dict(zip(faq_df["faq_id"], faq_df["question"]))

    rows, result_rows = [], []
    for qid, text, gold, gtopic, oos, hits, lat in zip(query_ids, texts, gold_sets, gold_topics,
                                                       is_oos, ranked, latencies):
        clusters = collapse_to_clusters([f for f, _ in hits], faq_to_cluster)
        top1 = hits[0] if hits else (None, float("nan"))
        row = {
            "query_id": qid, "query": text, "is_oos": bool(oos),
            "gold_clusters": "|".join(map(str, sorted(gold))) if gold else "",
            "gold_topic": gtopic,
            "top1_faq_id": top1[0], "top1_score": top1[1],
            "top1_cluster": faq_to_cluster.get(top1[0]),
            "top1_topic": faq_to_topic.get(top1[0]),
            "top1_question": faq_to_question.get(top1[0]),
            "latency_ms": lat,
        }
        if not oos:
            rank = first_relevant_rank(clusters, gold)
            row["first_rel_rank"] = rank
            row["rr"] = reciprocal_rank(clusters, gold, mrr_k)
            for k in ks:
                row[f"hit@{k}"] = hit_at_k(clusters, gold, k)
                row[f"p@{k}"] = precision_at_k(clusters, gold, k)
                row[f"recall@{k}"] = recall_at_k(clusters, gold, k)
        rows.append(row)
        result_rows.extend({"query_id": qid, "rank": r, "faq_id": f, "score": s, "latency_ms": lat}
                           for r, (f, s) in enumerate(hits[:mrr_k], start=1))
    return pd.DataFrame(rows), pd.DataFrame(result_rows)


def summarize(per_query: pd.DataFrame, ks=KS, mrr_k=MRR_K) -> dict:
    """Средние метрики по запросам по теме + задержка по всем запросам."""
    pq = per_query[~per_query["is_oos"]]
    out = {"n_queries": int(len(pq))}
    if len(pq):
        out["p@1"] = pq["hit@1"].mean()
        for k in ks:
            if k > 1:
                out[f"hit@{k}"] = pq[f"hit@{k}"].mean()
                out[f"p@{k}"] = pq[f"p@{k}"].mean()
        out[f"recall@{max(ks)}"] = pq[f"recall@{max(ks)}"].mean()
        out[f"mrr@{mrr_k}"] = pq["rr"].mean()
    lat = per_query["latency_ms"]
    out["latency_mean_ms"] = lat.mean()
    out["latency_p95_ms"] = float(np.percentile(lat, 95)) if len(lat) else float("nan")
    return out


def evaluate(retriever: BaseRetriever, faq_df: pd.DataFrame, queries: pd.DataFrame,
             qrels: pd.DataFrame, split: str | None = "test", fit: bool = True,
             measure_memory: bool = True, top_n: int = TOP_N) -> RunResult:
    """Протокол A: ручные запросы. split=None — все запросы (dev + test)."""
    faq_df = validate_faq(faq_df)
    queries, qrels = validate_eval_set(queries, qrels, faq_df)
    if split is not None:
        queries = queries[queries["split"] == split]
    if queries.empty:
        raise ValueError(f"Нет запросов со split={split!r}")

    stats = fit_with_stats(retriever, faq_df, measure_memory) if fit else {}
    gold = qrels.groupby("query_id")["answer_cluster"].apply(set).to_dict()
    gtopic = (qrels.drop_duplicates("query_id").set_index("query_id")["topic"].to_dict()
              if "topic" in qrels else {})
    qids = queries["query_id"].tolist()

    ranked, lat = _search_each(retriever, queries["text"].tolist(), None, top_n)
    per_query, results = _per_query_table(
        qids, queries["text"].tolist(), [gold.get(q, set()) for q in qids],
        [gtopic.get(q) for q in qids], queries["is_oos"].tolist(), ranked, lat, faq_df)
    metrics = {**summarize(per_query), **stats}
    return RunResult(retriever.name, split or "all", retriever.params(), metrics, per_query, results)


def leave_one_out(retriever: BaseRetriever, faq_df: pd.DataFrame, fit: bool = True,
                  measure_memory: bool = True, top_n: int = TOP_N) -> RunResult:
    """Протокол B: каждый вопрос кластера ищет остальные вопросы того же кластера.

    Из индекса прячется не только сам вопрос, но и все FAQ с тем же
    нормализованным текстом вопроса (в данных есть точные дубли) — иначе
    запрос нашёл бы свою копию и метрика была бы завышена. Вопрос участвует,
    только если в кластере остался хотя бы один другой вопрос.
    Индекс строится один раз (маскирование вместо переобучения): для TF-IDF
    это значит, что IDF посчитан с учётом скрытого вопроса — сдвиг 1/N, им
    пренебрегаем.
    """
    faq_df = validate_faq(faq_df)
    stats = fit_with_stats(retriever, faq_df, measure_memory) if fit else {}

    norm_q = faq_df["question"].map(normalize).to_numpy()
    clusters = faq_df["answer_cluster"].to_numpy()
    rows = []
    for i in range(len(faq_df)):
        same_text = norm_q == norm_q[i]
        remaining = (clusters == clusters[i]) & ~same_text
        if remaining.any():
            rows.append((i, same_text))
    if not rows:
        raise ValueError("Нет кластеров с двумя и более разными вопросами — LOO невозможен")

    idx = [i for i, _ in rows]
    mask = np.vstack([m for _, m in rows])
    sub = faq_df.iloc[idx]
    ranked, lat = _search_each(retriever, sub["question"].tolist(), mask, top_n)
    topics = sub["topic"].tolist() if "topic" in sub else [None] * len(sub)
    per_query, results = _per_query_table(
        sub["faq_id"].tolist(), sub["question"].tolist(), [{c} for c in sub["answer_cluster"]],
        topics, [False] * len(sub), ranked, lat, faq_df)
    metrics = {**summarize(per_query), **stats}
    return RunResult(retriever.name, "loo", retriever.params(), metrics, per_query, results)


# --- Сравнение методов ---------------------------------------------------------

def resolve_methods(methods: str | Sequence[str]) -> list[str]:
    if isinstance(methods, str):
        if methods == "all":
            return list(METHODS)
        if methods == "lexical":
            return list(LEXICAL_METHODS)
        return [m.strip() for m in methods.split(",") if m.strip()]
    return list(methods)


def run_benchmark(methods: str | Sequence[str | BaseRetriever], faq_df: pd.DataFrame,
                  protocol: str = "loo", queries: pd.DataFrame | None = None,
                  qrels: pd.DataFrame | None = None, split: str | None = "test",
                  measure_memory: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, RunResult]]:
    """Прогнать несколько методов одним протоколом -> (таблица метрик, прогоны)."""
    if protocol not in ("loo", "test"):
        raise ValueError("protocol: 'loo' или 'test'")
    if protocol == "test" and (queries is None or qrels is None):
        queries, qrels = load_eval_set()

    items = resolve_methods(methods) if isinstance(methods, str) else list(methods)
    runs: dict[str, RunResult] = {}
    for item in items:
        retriever = build_retriever(item) if isinstance(item, str) else item
        if verbose:
            print(f"  [{protocol}] {retriever.name} ...", flush=True)
        if protocol == "loo":
            run = leave_one_out(retriever, faq_df, measure_memory=measure_memory)
        else:
            run = evaluate(retriever, faq_df, queries, qrels, split=split,
                           measure_memory=measure_memory)
        runs[retriever.name] = run

    table = pd.DataFrame([{"method": name, **run.metrics} for name, run in runs.items()])
    sort_col = f"mrr@{MRR_K}"
    if sort_col in table:
        table = table.sort_values(sort_col, ascending=False, kind="stable").reset_index(drop=True)
    return table, runs


def best_method(table: pd.DataFrame, by: str = f"mrr@{MRR_K}") -> str:
    """Метод для чат-бота: максимум MRR, при равенстве — быстрее по p95."""
    ordered = table.sort_values([by, "latency_p95_ms"], ascending=[False, True], kind="stable")
    return str(ordered.iloc[0]["method"])


def save_benchmark(table: pd.DataFrame, runs: dict[str, RunResult], tag: str,
                   out_dir: str | Path | None = None) -> Path:
    """metrics_<tag>.csv + per_query_<tag>_<method>.csv (для error analysis)."""
    out_dir = Path(out_dir or config.METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"metrics_{tag}.csv", index=False)
    for name, run in runs.items():
        run.per_query.to_csv(out_dir / f"per_query_{tag}_{name}.csv", index=False)
    return out_dir / f"metrics_{tag}.csv"


# --- Порог отказа -----------------------------------------------------------------

def threshold_report(per_query: pd.DataFrame, threshold: float) -> dict:
    """Качество решения "отвечать / отказать" при данном пороге score топ-1.

    refuse_precision/recall — про запросы не по теме (их надо отклонять);
    coverage — доля запросов по теме, на которые бот ответил;
    answered_correct — доля запросов по теме, где бот ответил и топ-1 верный.
    """
    answered = per_query["top1_score"] >= threshold
    oos = per_query["is_oos"].astype(bool)
    refused = ~answered
    tp = int((refused & oos).sum())
    in_domain = ~oos
    correct = per_query.get("hit@1", pd.Series(0.0, index=per_query.index)).fillna(0).astype(bool)
    return {
        "threshold": float(threshold),
        "refuse_precision": float(tp / refused.sum()) if refused.sum() else float("nan"),
        "refuse_recall": float(tp / oos.sum()) if oos.sum() else float("nan"),
        "coverage": float(answered[in_domain].mean()) if in_domain.any() else float("nan"),
        "answered_correct": float((answered & correct)[in_domain].mean()) if in_domain.any() else float("nan"),
        "balanced_accuracy": _balanced_accuracy(answered, oos),
    }


def _balanced_accuracy(answered: pd.Series, oos: pd.Series) -> float:
    parts = []
    if (~oos).any():
        parts.append(answered[~oos].mean())   # по теме -> должен ответить
    if oos.any():
        parts.append((~answered[oos]).mean())  # не по теме -> должен отказать
    return float(np.mean(parts)) if parts else float("nan")


def tune_threshold(per_query_dev: pd.DataFrame) -> dict:
    """Порог, максимизирующий balanced accuracy решения на dev-запросах.

    Шкалы score у методов разные (косинус, BM25, RRF), поэтому порог
    подбирается отдельно для каждого метода. При равенстве берётся
    меньший порог: лучше ответить, чем отказать.
    """
    if not per_query_dev["is_oos"].any():
        raise ValueError("Для подбора порога нужны dev-запросы не по теме (is_oos=1)")
    scores = np.unique(per_query_dev["top1_score"].dropna().to_numpy())
    candidates = np.concatenate([[scores.min() - 1e-9], scores])
    best = max((threshold_report(per_query_dev, t) for t in candidates),
               key=lambda r: (r["balanced_accuracy"], -r["threshold"]))
    return best


# --- Confusion matrix по темам -----------------------------------------------

def topic_confusion(per_query: pd.DataFrame, labels: Sequence[str] | None = None,
                    pred_col: str = "top1_topic") -> tuple[np.ndarray, list[str]]:
    """Тема топ-1 ответа против gold-темы запроса (запросы по теме)."""
    from sklearn.metrics import confusion_matrix

    pq = per_query[~per_query["is_oos"]].dropna(subset=["gold_topic", pred_col])
    if labels is None:
        present = set(pq["gold_topic"]) | set(pq[pred_col])
        labels = [t for t in config.TOPICS if t in present] + sorted(present - set(config.TOPICS))
    return confusion_matrix(pq["gold_topic"], pq[pred_col], labels=list(labels)), list(labels)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str,
                          path: str | Path | None = None, normalize_rows: bool = True):
    """Тепловая карта confusion matrix; строки — истинная тема, столбцы — предсказанная."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalize_rows:
        sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, sums, out=np.zeros_like(data), where=sums > 0)
    names = [config.TOPICS.get(l, l) for l in labels]
    side = max(6.0, 2.5 + 0.7 * len(labels))  # место под длинные русские подписи
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize_rows else None)
    ax.set_xticks(range(len(labels)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Предсказанная тема")
    ax.set_ylabel("Истинная тема")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="доля строки" if normalize_rows else "число запросов")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# --- CLI ----------------------------------------------------------------------

def _print_table(table: pd.DataFrame) -> None:
    cols = [c for c in ["method", "n_queries", "p@1", "hit@3", "hit@5", f"mrr@{MRR_K}",
                        "index_time_s", "latency_mean_ms", "latency_p95_ms", "peak_mem_mb",
                        "index_size_mb", "model_size_mb"] if c in table]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(table[cols].to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # консоль Windows по умолчанию cp1251
    parser = argparse.ArgumentParser(description="Сравнение методов поиска по FAQ")
    parser.add_argument("--protocol", choices=["loo", "test", "dev"], default="loo")
    parser.add_argument("--methods", default="lexical",
                        help="all | lexical | список через запятую: " + ", ".join(METHODS))
    parser.add_argument("--faq", default=None, help="по умолчанию data/processed/faq_clean.parquet")
    parser.add_argument("--no-memory", action="store_true", help="не мерить память (быстрее)")
    parser.add_argument("--save", action="store_true", help="сохранить CSV в reports/metrics")
    args = parser.parse_args(argv)

    faq_df = load_faq(args.faq)
    protocol = "loo" if args.protocol == "loo" else "test"
    split = args.protocol if args.protocol in ("test", "dev") else None
    print(f"FAQ: {len(faq_df)} записей, кластеров: {faq_df['answer_cluster'].nunique()}")
    table, runs = run_benchmark(args.methods, faq_df, protocol=protocol, split=split,
                                measure_memory=not args.no_memory)
    _print_table(table)
    print(f"\nЛучший по MRR@{MRR_K}: {best_method(table)}")
    if args.save:
        print("Сохранено:", save_benchmark(table, runs, tag=args.protocol))


if __name__ == "__main__":
    main()
