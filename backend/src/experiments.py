"""Все эксперименты AI-модуля одним запуском -> таблицы и графики для отчёта.

    python -m src.experiments --protocol loo            # без ручной разметки
    python -m src.experiments --protocol test           # основной: ручные запросы
    python -m src.experiments --protocol test --skip-neural

Что считается (меняем один параметр за раз):
    1. Предобработка: размер словаря и время raw / stem / lemma (/ spaCy).
    2. Сравнение методов: baseline + BoW, TF-IDF, эмбеддинги, гибрид.
    3. Абляции: режим предобработки для TF-IDF, вопросительные слова,
       стоп-слова, что индексируем (вопрос или вопрос + ответ).
    Только для protocol=test (нужны dev-запросы и gold-темы):
    4. Веса гибрида и порог отказа — подбор на dev, отчёт на test.
    5. Классификатор тем: Dummy / TF-IDF+LR / e5+LR, confusion matrix.
    6. Фильтр по теме: MRR лучшего метода с фильтром и без.
    7. Выбор метода для бота и сохранение models/chatbot.json.

Результаты: reports/metrics/*.csv, reports/figures/*.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .chatbot import save_bot_config
from .classifier import evaluate_classifiers, make_classifiers, save_classifier, topic_eval_data
from .evaluation import (
    MRR_K,
    best_method,
    evaluate,
    load_eval_set,
    load_faq,
    plot_confusion_matrix,
    run_benchmark,
    save_benchmark,
    threshold_report,
    topic_confusion,
    tune_threshold,
)
from .preprocessing import compare_modes
from .retrievers import (
    LEXICAL_METHODS,
    METHODS,
    BaseRetriever,
    HybridRRFRetriever,
    TfidfCharRetriever,
    TfidfWordRetriever,
    TopicFilteredRetriever,
    build_retriever,
)

__all__ = ["run_all", "run_ablations", "hybrid_weight_search", "ablation_retrievers"]

MRR = f"mrr@{MRR_K}"


def ablation_retrievers(include_neural: bool = True) -> dict[str, Callable[[], BaseRetriever]]:
    """Варианты, отличающиеся от базовой конфигурации одним параметром."""
    variants: dict[str, Callable[[], BaseRetriever]] = {
        # режим предобработки (база — lemma)
        "tfidf_word | raw": lambda: TfidfWordRetriever(mode="raw"),
        "tfidf_word | stem": lambda: TfidfWordRetriever(mode="stem"),
        "tfidf_word | lemma": lambda: TfidfWordRetriever(mode="lemma"),
        # стоп-слова
        "tfidf_word | lemma, вопр. слова оставлены": lambda: TfidfWordRetriever(keep_question_words=True),
        "tfidf_word | lemma, без удаления стоп-слов": lambda: TfidfWordRetriever(remove_stopwords=False),
        "tfidf_char | вопр. слова убраны": lambda: TfidfCharRetriever(remove_stopwords=True),
        # что индексируем
        "tfidf_word | lemma, вопрос+ответ": lambda: TfidfWordRetriever(index_field="question_answer"),
        "tfidf_char | вопрос+ответ": lambda: TfidfCharRetriever(index_field="question_answer"),
    }
    if include_neural:
        variants["e5_small | вопрос+ответ"] = lambda: build_retriever("e5_small", index_field="question_answer")
    return variants


def _named(name: str, factory: Callable[[], BaseRetriever]) -> BaseRetriever:
    r = factory()
    r.name = name
    return r


def run_ablations(faq_df, protocol, queries=None, qrels=None, include_neural=True) -> pd.DataFrame:
    retrievers = [_named(n, f) for n, f in ablation_retrievers(include_neural).items()]
    table, _ = run_benchmark(retrievers, faq_df, protocol=protocol, queries=queries, qrels=qrels,
                             split="test", measure_memory=False)
    return table


def hybrid_weight_search(faq_df, queries, qrels, lexical: str = "tfidf_char",
                         grid=(0.25, 0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Вес лексической компоненты RRF относительно e5 — только на dev."""
    rows = []
    for w in grid:
        r = HybridRRFRetriever(components=(lexical, "e5_small"), weights=(w, 1.0))
        run = evaluate(r, faq_df, queries, qrels, split="dev", measure_memory=False)
        rows.append({"lexical": lexical, "weight_lexical": w, "weight_e5": 1.0, **run.metrics})
    return pd.DataFrame(rows).sort_values(MRR, ascending=False, kind="stable").reset_index(drop=True)


def _save(df: pd.DataFrame, name: str) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.METRICS_DIR / f"{name}.csv", index=False)


def _show(title: str, df: pd.DataFrame, cols: list[str] | None = None) -> None:
    print(f"\n=== {title} ===")
    cols = [c for c in (cols or df.columns) if c in df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))


QUALITY_COLS = ["method", "n_queries", "p@1", "hit@3", "hit@5", MRR, "index_time_s",
                "latency_mean_ms", "latency_p95_ms", "peak_mem_mb", "rss_delta_mb",
                "index_size_mb", "model_size_mb"]


def run_all(protocol: str = "loo", include_neural: bool = True,
            faq_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    faq_df = load_faq(faq_path)
    out: dict[str, pd.DataFrame] = {}
    print(f"FAQ: {len(faq_df)} записей, кластеров {faq_df['answer_cluster'].nunique()}, протокол {protocol}")

    # 1. предобработка
    prep = compare_modes(faq_df["question"])
    _save(prep, "preprocessing_modes")
    _show("Предобработка (вопросы FAQ)", prep)
    out["preprocessing"] = prep

    queries = qrels = None
    bench_protocol = "loo"
    if protocol == "test":
        queries, qrels = load_eval_set()
        bench_protocol = "test"

    # 2. сравнение методов
    methods = list(METHODS) if include_neural else list(LEXICAL_METHODS)
    table, runs = run_benchmark(methods, faq_df, protocol=bench_protocol, queries=queries,
                                qrels=qrels, split="test")
    save_benchmark(table, runs, tag=protocol)
    _show(f"Методы ({protocol})", table, QUALITY_COLS)
    out["methods"] = table

    # 3. абляции
    abl = run_ablations(faq_df, bench_protocol, queries, qrels, include_neural)
    _save(abl, f"ablations_{protocol}")
    _show(f"Абляции ({protocol})", abl, ["method", "p@1", "hit@3", MRR, "latency_mean_ms"])
    out["ablations"] = abl

    if protocol != "test":
        best = best_method(table)
        cm, labels = topic_confusion(runs[best].per_query)
        plot_confusion_matrix(cm, labels, f"Тема топ-1 ответа, {best} (LOO)",
                              config.FIGURES_DIR / f"cm_topics_{best}_loo.png")
        print(f"\nЛучший по {MRR} (LOO): {best}. Порог, классификатор и фильтр "
              f"считаются только с ручными запросами (--protocol test).")
        return out

    # 4. гибрид и порог — подбор на dev
    if include_neural:
        # лучший лексический метод, кроме baseline (в плане: "лучший TF-IDF + e5")
        candidates = [m for m in LEXICAL_METHODS if m != "rule_based"]
        lexical_best = table[table["method"].isin(candidates)].iloc[0]["method"]
        weights = hybrid_weight_search(faq_df, queries, qrels, lexical=lexical_best)
        _save(weights, "hybrid_weights_dev")
        _show("Веса гибрида (dev)", weights, ["lexical", "weight_lexical", MRR, "p@1"])
        out["hybrid_weights"] = weights
        w = float(weights.iloc[0]["weight_lexical"])
        tuned = HybridRRFRetriever(components=(lexical_best, "e5_small"), weights=(w, 1.0))
        tuned.name = "hybrid_rrf_tuned"
        run = evaluate(tuned, faq_df, queries, qrels, split="test", measure_memory=False)
        runs[tuned.name] = run
        table = pd.concat([table, pd.DataFrame([{"method": tuned.name, **run.metrics}])],
                          ignore_index=True).sort_values(MRR, ascending=False, kind="stable")
        save_benchmark(table, runs, tag=protocol)

    thr_rows = []
    try:
        for name in table["method"]:
            dev_run = evaluate(_rebuild(name, runs), faq_df, queries, qrels, split="dev",
                               measure_memory=False)
            tuned_thr = tune_threshold(dev_run.per_query)
            test_rep = threshold_report(runs[name].per_query, tuned_thr["threshold"])
            thr_rows.append({"method": name, "threshold_dev": tuned_thr["threshold"],
                             **{f"test_{k}": v for k, v in test_rep.items() if k != "threshold"}})
    except ValueError as exc:  # нет dev-запросов или среди них нет запросов не по теме
        print(f"\nПорог отказа не подобран: {exc}")
    thresholds = pd.DataFrame(thr_rows, columns=None if thr_rows else ["method", "threshold_dev"])
    if thr_rows:
        _save(thresholds, "thresholds_test")
        _show("Порог отказа (подбор на dev, метрики на test)", thresholds)
    out["thresholds"] = thresholds

    best = best_method(table)
    cm, labels = topic_confusion(runs[best].per_query)
    plot_confusion_matrix(cm, labels, f"Тема топ-1 ответа: {best}",
                          config.FIGURES_DIR / f"cm_topics_{best}.png")

    # 5. классификатор тем
    texts, topics = topic_eval_data(queries, qrels, split="test")
    clf_table, details = evaluate_classifiers(faq_df, texts, topics,
                                              models=make_classifiers(include_neural), do_tune=True)
    _save(clf_table, "topic_classifiers_test")
    _show("Классификатор тем (train: FAQ, test: ручные запросы)", clf_table)
    for name, d in details.items():
        d["report"].to_csv(config.METRICS_DIR / f"topic_report_{name}.csv")
        plot_confusion_matrix(d["cm"], d["labels"], f"Классификатор тем: {name}",
                              config.FIGURES_DIR / f"cm_classifier_{name}.png")
    best_clf = clf_table.iloc[0]["model"]
    save_classifier(details[best_clf]["model"])
    out["classifiers"] = clf_table

    # 6. фильтр по теме
    filtered = TopicFilteredRetriever(_rebuild(best, runs), details[best_clf]["model"])
    frun = evaluate(filtered, faq_df, queries, qrels, split="test", measure_memory=False)
    topic_filter = pd.DataFrame([{"method": best, "topic_filter": False, **runs[best].metrics},
                                 {"method": best, "topic_filter": True, **frun.metrics}])
    _save(topic_filter, "topic_filter_test")
    _show("Фильтр по теме", topic_filter, ["method", "topic_filter", "p@1", "hit@3", MRR])
    out["topic_filter"] = topic_filter

    # 7. метод для бота
    best_row = table.set_index("method").loc[best]
    thr = None
    if best in set(thresholds["method"]):
        thr = float(thresholds.set_index("method").loc[best, "threshold_dev"])
    method, params = _registry_spec(best, runs)
    path = save_bot_config(method, threshold=thr, params=params,
                           metrics={k: float(best_row[k]) for k in ("p@1", MRR) if k in best_row})
    thr_text = f"порог {thr:.3f}" if thr is not None else "без порога"
    print(f"\nДля бота выбран {best} (MRR={best_row[MRR]:.3f}), {thr_text} -> {path}")
    return out


def _registry_spec(name: str, runs) -> tuple[str, dict]:
    """Имя прогона -> (метод из реестра, параметры), чтобы бот мог его пересобрать."""
    if name in METHODS:
        return name, {}
    p = runs[name].params  # подобранный гибрид
    return "hybrid_rrf", {"components": [c["name"] for c in p["components"]],
                          "weights": p["weights"], "k_rrf": p["k_rrf"]}


def _rebuild(name: str, runs) -> BaseRetriever:
    """Свежий экземпляр метода по имени прогона (для dev-прогонов и фильтра)."""
    method, params = _registry_spec(name, runs)
    r = build_retriever(method, **params)
    r.name = name
    return r


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Эксперименты AI-модуля")
    parser.add_argument("--protocol", choices=["loo", "test"], default="loo")
    parser.add_argument("--skip-neural", action="store_true", help="без e5/MiniLM/гибрида")
    parser.add_argument("--faq", default=None, help="по умолчанию data/processed/faq_clean.parquet")
    args = parser.parse_args(argv)
    run_all(args.protocol, include_neural=not args.skip_neural, faq_path=args.faq)


if __name__ == "__main__":
    main()
