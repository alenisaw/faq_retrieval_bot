"""AITU FAQ Retrieval Chatbot — консольное меню.

    python main.py              # интерактивное меню
    python main.py --setup      # пункт 1 + оценка лексических методов (без диалога)
    python main.py --ask "общага есть?"
    python main.py --serve      # веб-API на http://localhost:8000 (документация /docs)

Пункты меню:
    1 – Загрузить и очистить данные (аудит -> cleaning_log.json, faq.db)
    2 – EDA: графики в reports/figures
    3 – Построить индексы всех методов (время, память, размер)
    4 – Оценка: таблица метрик + confusion matrix (результаты -> faq.db)
    5 – Задать вопрос (выбор метода, топ-3 со score)
    6 – SQL-отчёты
    7 – Запустить веб-API
    0 – Выход
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from src import config, db

MENU = """
================ AITU FAQ Retrieval Chatbot ================
 1 – Загрузить и очистить данные (аудит -> cleaning_log.json)
 2 – EDA: графики в reports/figures
 3 – Построить индексы всех методов
 4 – Оценка: таблица метрик + confusion matrix
 5 – Задать вопрос (выбор метода, топ-3 со score)
 6 – SQL-отчёты
 7 – Запустить веб-API (FastAPI)
 0 – Выход
============================================================"""

MENU_LOG = config.REPORTS_DIR / "menu_log.jsonl"   # что и когда запускалось из меню


# --- Ввод ---------------------------------------------------------------------

def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default or "")


def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    """Выбор по номеру или по имени; повторяет вопрос, пока ответ не подойдёт."""
    for i, opt in enumerate(options, start=1):
        mark = "  (по умолчанию)" if opt == default else ""
        print(f"  {i}. {opt}{mark}")
    while True:
        raw = ask(prompt, default)
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Введите номер от 1 до {len(options)} или имя из списка.")


def ask_yes(prompt: str, default: bool = False) -> bool:
    raw = ask(f"{prompt} (д/н)", "д" if default else "н").lower()
    return raw in {"д", "да", "y", "yes", "1"}


def log_action(action: str, **details) -> None:
    """Строка в reports/menu_log.jsonl: время, пункт меню, итог."""
    MENU_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, **details}
    with MENU_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def print_table(df: pd.DataFrame, max_rows: int = 40) -> None:
    if df.empty:
        print("  (пусто)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_colwidth", 60, "display.float_format", "{:.3f}".format):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... ещё {len(df) - max_rows} строк")


def load_faq_or_none() -> pd.DataFrame | None:
    from src.evaluation import load_faq

    try:
        return load_faq()
    except FileNotFoundError:
        print("Нет data/processed/faq_clean.parquet — сначала пункт 1.")
        return None


# --- Пункты меню ------------------------------------------------------------------

def action_prepare() -> None:
    from src.pipeline import prepare_all

    faq_df, summary = prepare_all()
    print(f"\nСохранено: {config.FAQ_CLEAN_PATH.name}, {config.CLEANING_LOG_PATH.name}, "
          f"{config.DB_PATH.name}; для ручной проверки тем — {config.TOPIC_REVIEW_PATH.name}")
    log_action("prepare", rows_raw=summary["rows_raw"], rows_clean=summary["rows_clean"],
               clusters=summary["answer_clusters"])


def action_eda() -> None:
    try:
        from src import eda
    except ImportError:
        print("Модуль src/eda.py ещё не добавлен (зона Алена). Пункт заработает после его PR.")
        return
    runner = getattr(eda, "run_all", None) or getattr(eda, "main", None)
    if runner is None:
        print("В src/eda.py нет функции run_all() или main().")
        return
    runner()
    print(f"Графики: {config.FIGURES_DIR}")
    log_action("eda")


def _choose_methods() -> list[str]:
    from src.retrievers import LEXICAL_METHODS, METHODS

    choice = ask_choice("Какие методы", ["лексические (быстро, без скачивания)", "все (e5/MiniLM ~470 МБ каждая)"],
                        default="лексические (быстро, без скачивания)")
    return list(LEXICAL_METHODS) if choice.startswith("лексические") else list(METHODS)


def action_build_indexes() -> None:
    from src.evaluation import fit_with_stats
    from src.retrievers import build_retriever

    faq_df = load_faq_or_none()
    if faq_df is None:
        return
    rows = []
    for name in _choose_methods():
        print(f"  строю {name} ...", flush=True)
        try:
            stats = fit_with_stats(build_retriever(name), faq_df)
        except (OSError, RuntimeError) as exc:   # нет сети для скачивания модели и т.п.
            print(f"  {name}: не удалось ({exc})")
            continue
        rows.append({"method": name, **stats})
    table = pd.DataFrame(rows)
    print()
    print_table(table)
    if not table.empty:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.METRICS_DIR / "index_build.csv"
        table.to_csv(path, index=False)
        print(f"Сохранено: {path}")
        log_action("build_indexes", methods=table["method"].tolist())


def action_evaluate() -> None:
    from src.evaluation import (best_method, load_eval_set, plot_confusion_matrix, run_benchmark,
                                save_benchmark, topic_confusion)
    from src.pipeline import save_runs

    faq_df = load_faq_or_none()
    if faq_df is None:
        return
    has_test = config.TEST_QUERIES_PATH.exists() and config.QRELS_PATH.exists()
    protocols = ["loo — leave-one-out по перефразам"]
    if has_test:
        protocols.insert(0, "test — ручной тест-набор (основной)")
        protocols.append("experiments — все эксперименты AI-модуля (протокол test)")
    else:
        print("Ручного тест-набора ещё нет (eval/test_queries.csv) — доступен только протокол B.")
    protocol = ask_choice("Протокол", protocols, default=protocols[0]).split(" ")[0]

    if protocol == "experiments":
        from src.experiments import run_all

        run_all("test", include_neural=ask_yes("С нейросетевыми методами?", True))
        log_action("experiments")
        return

    methods = _choose_methods()
    queries = qrels = None
    if protocol == "test":
        queries, qrels = load_eval_set()
        with db.connect() as conn:
            db.init_db(conn)
            db.load_eval_set(conn, queries, qrels)
    table, runs = run_benchmark(methods, faq_df, protocol=protocol, queries=queries, qrels=qrels)
    cols = [c for c in ("method", "n_queries", "p@1", "hit@3", "hit@5", "mrr@10", "index_time_s",
                        "latency_mean_ms", "latency_p95_ms", "peak_mem_mb") if c in table]
    print()
    print_table(table[cols])
    save_benchmark(table, runs, tag=protocol)
    run_ids = save_runs(runs)
    best = best_method(table)
    cm, labels = topic_confusion(runs[best].per_query)
    fig_path = config.FIGURES_DIR / f"cm_topics_{best}_{protocol}.png"
    plot_confusion_matrix(cm, labels, f"Тема топ-1 ответа: {best} ({protocol})", fig_path)
    print(f"\nЛучший по MRR@10: {best}. Confusion matrix: {fig_path}")
    print(f"Метрики: {config.METRICS_DIR}, прогоны в faq.db: run_id {run_ids[0]}–{run_ids[-1]}")
    log_action("evaluate", protocol=protocol, best=best, runs=run_ids)


def action_ask(question: str | None = None, method: str | None = None) -> None:
    from src.chatbot import ChatBot, chat_loop, format_response, load_bot_config
    from src.retrievers import METHODS, build_retriever

    faq_df = load_faq_or_none()
    if faq_df is None:
        return
    cfg = load_bot_config()
    if method is None and question is None:
        method = ask_choice("Метод поиска", list(METHODS), default=cfg["method"])
    method = method or cfg["method"]
    if method == cfg["method"]:
        retriever, threshold = build_retriever(method, **cfg.get("params", {})), cfg.get("threshold")
    else:
        retriever, threshold = build_retriever(method), None
    logger = db.make_chat_logger(method_params=retriever.params(), source="cli") if config.DB_PATH.exists() else None
    print(f"Строю индекс {method} ...", flush=True)
    bot = ChatBot(retriever, faq_df, threshold=threshold, on_answer=logger)
    if question:
        print(format_response(bot.answer(question)))
        return
    n = chat_loop(bot)
    log_action("ask", method=method, answered=n)


def action_reports() -> None:
    if not config.DB_PATH.exists():
        print("Нет faq.db — сначала пункт 1.")
        return
    reports = list(db.load_reports().values())
    while True:
        print()
        for i, r in enumerate(reports, start=1):
            print(f"  {i:>2}. {r.title}")
        print("   0. Назад")
        raw = ask("Номер отчёта", "0")
        if raw == "0":
            return
        if not raw.isdigit() or not 1 <= int(raw) <= len(reports):
            print("  Нет такого номера.")
            continue
        r = reports[int(raw) - 1]
        with db.connect() as conn:
            df = db.run_report(conn, r.name)
        print(f"\n### {r.title}")
        if r.description:
            print(f"    {r.description}")
        print_table(df)
        if not df.empty and ask_yes("Сохранить в CSV?"):
            out = config.REPORTS_DIR / "sql" / f"{r.name}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"  Сохранено: {out}")
        log_action("sql_report", name=r.name, rows=len(df))


def action_serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"API: http://{host}:{port}  (документация /docs, остановка — Ctrl+C)")
    uvicorn.run("src.api.app:app", host=host, port=port)


ACTIONS = {
    "1": action_prepare,
    "2": action_eda,
    "3": action_build_indexes,
    "4": action_evaluate,
    "5": action_ask,
    "6": action_reports,
    "7": action_serve,
}


def menu() -> None:
    while True:
        print(MENU)
        try:
            choice = input("Выберите пункт: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "0":
            break
        elif choice in ACTIONS:
            try:
                ACTIONS[choice]()
            except KeyboardInterrupt:
                print("\nПрервано, возврат в меню.")
            except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
                print(f"\nОшибка: {exc}")
                log_action("error", item=choice, error=str(exc))
        else:
            print("Нет такого пункта, введите число от 0 до 7.")
    print("До свидания!")


def setup() -> None:
    """Одна команда для свежего клона: данные, база, оценка без скачивания моделей."""
    from src.pipeline import evaluate_lexical_loo, prepare_all

    faq_df, _ = prepare_all()
    print("\nОценка лексических методов (протокол B, leave-one-out):")
    table = evaluate_lexical_loo(faq_df)
    print_table(table[[c for c in ("method", "p@1", "hit@3", "mrr@10", "latency_mean_ms") if c in table]])
    print("\nГотово. Дальше: python main.py (меню) или python main.py --serve (веб-API).")
    log_action("setup")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # консоль Windows по умолчанию cp1251
    parser = argparse.ArgumentParser(description="AITU FAQ Retrieval Chatbot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--setup", action="store_true", help="подготовить данные и базу, оценить лексические методы")
    group.add_argument("--ask", metavar="ВОПРОС", help="задать один вопрос и выйти")
    group.add_argument("--serve", action="store_true", help="запустить веб-API")
    parser.add_argument("--method", help="метод для --ask (по умолчанию из models/chatbot.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.setup:
        setup()
    elif args.ask:
        action_ask(args.ask, args.method)
    elif args.serve:
        action_serve(args.host, args.port)
    else:
        menu()


if __name__ == "__main__":
    main()
