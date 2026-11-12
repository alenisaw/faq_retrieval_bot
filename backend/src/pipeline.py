"""Сборка данных одним вызовом: очистка -> faq_clean.parquet -> faq.db.

    python -m src.pipeline                 # пункт 1 меню без диалога
    python -m src.pipeline --evaluate      # + leave-one-out лексических методов в БД

Используется в main.py (пункт 1 и --setup) и в Docker при первом запуске.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, db
from .data_loader import prepare_data
from .entities import entity_summary, extract_entities


def build_keywords(faq_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Топ-n термов каждого FAQ по TF-IDF (леммы) -> таблица keywords."""
    from .retrievers import TfidfWordRetriever

    return TfidfWordRetriever(mode="lemma").fit(faq_df).top_terms(n)


def build_database(faq_df: pd.DataFrame, db_path: str | Path | None = None,
                   verbose: bool = True) -> dict:
    """Схема + корпус + сущности + ключевые слова (+ тест-набор, если он уже есть)."""
    entities = extract_entities(faq_df)
    keywords = build_keywords(faq_df)
    with db.connect(db_path) as conn:
        db.init_db(conn)
        counts = db.load_corpus(conn, faq_df, keywords=keywords, entities=entities)
        eval_counts = load_eval_set_if_ready(conn, verbose=verbose)
    if verbose:
        print(f"\nБаза {Path(db_path or config.DB_PATH).name}: " +
              ", ".join(f"{k}={v}" for k, v in counts.items() if not k.startswith(("orphan", "chat")) ))
        lost = {k: v for k, v in counts.items() if k.startswith(("orphan", "chat")) and v}
        if lost:
            print(f"  После пересборки корпуса удалены/отвязаны ссылки: {lost}")
        print("\nСущности:")
        print(entity_summary(entities).to_string(index=False))
    return {"tables": counts, "eval_set": eval_counts, "n_entities": len(entities),
            "n_keywords": len(keywords)}


def load_eval_set_if_ready(conn, verbose: bool = True) -> dict | None:
    """eval/test_queries.csv + qrels.csv -> таблицы queries/qrels, если файлы есть."""
    if not (config.TEST_QUERIES_PATH.exists() and config.QRELS_PATH.exists()):
        if verbose:
            print("Тест-набор (eval/test_queries.csv, eval/qrels.csv) ещё не собран — пропускаю.")
        return None
    from .evaluation import load_eval_set

    queries, qrels = load_eval_set()
    counts = db.load_eval_set(conn, queries, qrels)
    if verbose:
        print(f"Тест-набор загружен: {counts}")
    return counts


def save_runs(runs: dict, db_path: str | Path | None = None) -> list[int]:
    """Прогоны run_benchmark -> runs/results в faq.db."""
    with db.connect(db_path) as conn:
        db.init_db(conn)
        return [db.save_run(conn, run) for run in runs.values()]


def prepare_all(db_path: str | Path | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Пункт 1 меню: исходник -> faq_clean.parquet + cleaning_log.json + faq.db."""
    faq_df, summary = prepare_data(verbose=verbose)
    summary["database"] = build_database(faq_df, db_path, verbose=verbose)
    return faq_df, summary


def evaluate_lexical_loo(faq_df: pd.DataFrame, db_path: str | Path | None = None,
                         verbose: bool = True) -> pd.DataFrame:
    """Протокол B для методов без нейросетей: секунды, без скачивания моделей."""
    from .evaluation import run_benchmark, save_benchmark
    from .retrievers import LEXICAL_METHODS

    table, runs = run_benchmark(list(LEXICAL_METHODS), faq_df, protocol="loo", verbose=verbose)
    save_benchmark(table, runs, tag="loo")
    save_runs(runs, db_path)
    return table


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Сборка faq_clean.parquet и faq.db")
    parser.add_argument("--evaluate", action="store_true",
                        help="после сборки прогнать leave-one-out лексических методов и записать в БД")
    args = parser.parse_args(argv)
    faq_df, _ = prepare_all()
    if args.evaluate:
        table = evaluate_lexical_loo(faq_df)
        cols = [c for c in ("method", "p@1", "hit@3", "mrr@10", "latency_mean_ms") if c in table]
        print(table[cols].to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
