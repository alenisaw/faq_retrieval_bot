"""Разбор ошибок поиска на тестовом наборе (протокол A).

    python -m src.error_analysis                 # метод из models/chatbot.json или tfidf_char
    python -m src.error_analysis --method bm25

Читает reports/metrics/per_query_test_<метод>.csv (их пишет src.experiments) и
раскладывает промахи top-1 по категориям. Категория назначается по первому
сработавшему правилу:

1. duplicate   — бот выдал равноценный ответ из другого кластера (дубль в базе,
                 эталон его не перечислил). Отмечается вручную в
                 reports/errors/review_<метод>.csv (query_id, acceptable, note):
                 автоматически по близости ответов это не ловится — шаблонные
                 ответы про разные специальности похожи, хотя отвечают на разное;
2. near_miss   — эталон на 2–3 месте;
3. other_topic — первый ответ из другой темы;
4. same_topic  — тема та же, вопрос не тот;
5. not_found   — эталона нет даже в топ-10.

Отдельно: доля промахов у запросов со сленгом/сокращениями, ложные ответы на
вопросы не по теме при текущем пороге и сравнение методов по темам.

Результаты: reports/errors/*.csv, reports/errors/error_analysis.md,
reports/figures/err_*.png.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .eda import INK, INK_SOFT, LINE, ULTRA, ULTRA_SOFT  # noqa: E402  (тот же стиль графиков)
from .evaluation import load_faq  # noqa: E402

CATEGORIES = {
    "duplicate": "Равноценный ответ из другого кластера (ручная проверка)",
    "near_miss": "Правильный ответ на 2–3 месте",
    "other_topic": "Ответ из другой темы",
    "same_topic": "Тема та же, вопрос не тот",
    "not_found": "Правильного ответа нет в топ-10",
}
# Разговорные слова, сокращения и транслит, которых нет в формальных вопросах FAQ
SLANG = re.compile(
    r"\b(?:скок|скока|доки|общаг\w*|универ(?!ситет)\w*|маг[уеи]?|аэт|ает|се|биг|"
    r"физр\w*|военк\w*|приемк\w*|зарег\w*|платк\w*|творч|англ|инфобез|норм|че|чё)\b",
    re.IGNORECASE,
)

METHOD_ORDER = ["rule_based", "bow", "tfidf_word_lemma", "tfidf_word_stem", "bm25", "tfidf_char",
                "tfidf_char_qa", "e5_small", "e5_small_qa", "minilm", "hybrid_rrf"]


# --- данные ---------------------------------------------------------------------------

def per_query_path(method: str, protocol: str = "test") -> Path:
    return config.METRICS_DIR / f"per_query_{protocol}_{method}.csv"


def load_per_query(method: str, protocol: str = "test") -> pd.DataFrame:
    path = per_query_path(method, protocol)
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}: сначала python -m src.experiments --protocol {protocol}")
    df = pd.read_csv(path)
    df["is_oos"] = df["is_oos"].astype(str).str.lower().isin({"1", "true"})
    return df


def parse_clusters(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(float(x)) for x in re.split(r"[|,;\s]+", str(value)) if x.strip()]


def review_path(method: str) -> Path:
    return config.ERRORS_DIR / f"review_{method}.csv"


def load_review(method: str) -> set[int]:
    """query_id, где ответ бота признан равноценным при ручной проверке."""
    path = review_path(method)
    if not path.exists():
        return set()
    rev = pd.read_csv(path)
    return set(rev.loc[rev["acceptable"].astype(int) == 1, "query_id"].astype(int))


# --- категории -----------------------------------------------------------------------

def categorize(row, acceptable: set[int] = frozenset()) -> str:
    if int(row.query_id) in acceptable:
        return "duplicate"
    rank = row.first_rel_rank
    if pd.notna(rank) and rank <= 3:
        return "near_miss"
    if pd.isna(rank) or rank > 10:
        return "not_found"
    if row.top1_topic != row.gold_topic:
        return "other_topic"
    return "same_topic"


def analyse(per_query: pd.DataFrame, faq_df: pd.DataFrame, acceptable: set[int] = frozenset()) -> pd.DataFrame:
    """Промахи top-1 среди запросов по теме, с категорией и флагом сленга."""
    in_scope = per_query[~per_query["is_oos"]].copy()
    in_scope["slang"] = in_scope["query"].str.contains(SLANG)
    misses = in_scope[in_scope["hit@1"] < 1].copy()
    misses["category"] = [categorize(r, acceptable) for r in misses.itertuples(index=False)]
    gold_q = (faq_df.drop_duplicates("answer_cluster").set_index("answer_cluster")["question"])
    misses["gold_question"] = misses["gold_clusters"].map(
        lambda v: (gold_q.get(parse_clusters(v)[0], "") if parse_clusters(v) else ""))
    return misses


def slang_table(per_query: pd.DataFrame) -> pd.DataFrame:
    in_scope = per_query[~per_query["is_oos"]].copy()
    in_scope["slang"] = in_scope["query"].str.contains(SLANG)
    out = in_scope.groupby("slang").agg(n=("hit@1", "size"), p_at_1=("hit@1", "mean"), mrr=("rr", "mean"))
    out.index = out.index.map({True: "есть сленг/сокращения", False: "без сленга"})
    return out.reset_index(names="запросы")


def oos_false_answers(per_query: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """Вопросы не по теме, на которые бот всё же ответил бы при этом пороге."""
    oos = per_query[per_query["is_oos"]].copy()
    if threshold is None:
        return oos.assign(answered=True)
    oos["answered"] = oos["confidence"] >= threshold
    return oos[oos["answered"]]


def topic_by_method(protocol: str = "test") -> pd.DataFrame:
    """P@1 по темам для всех методов, у которых есть per_query-файл."""
    cols = {}
    for m in METHOD_ORDER:
        path = per_query_path(m, protocol)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[~df["is_oos"].astype(str).str.lower().isin({"1", "true"})]
        cols[m] = df.groupby("gold_topic")["hit@1"].mean()
    table = pd.DataFrame(cols)
    return table.reindex([t for t in config.TOPICS if t in table.index])


# --- графики ------------------------------------------------------------------------

def plot_categories(misses: pd.DataFrame, method: str, total: int, out: Path) -> Path:
    counts = misses["category"].value_counts().reindex(list(CATEGORIES), fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = [CATEGORIES[c] for c in counts.index]
    colors = [ULTRA_SOFT if c == "duplicate" else ULTRA for c in counts.index]
    ax.barh(labels[::-1], counts.values[::-1], color=colors[::-1])
    for y, v in enumerate(counts.values[::-1]):
        ax.text(v + 0.3, y, str(v), va="center", color=INK, fontsize=9)
    ax.set_xlabel("число запросов")
    ax.set_title(f"Почему ошибается {method}: {len(misses)} промахов top-1 из {total}")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_topic_method(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * table.shape[1], 0.45 * table.shape[0] + 1.6))
    im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]), [config.TOPICS.get(t, t) for t in table.index])
    for (i, j), v in np.ndenumerate(table.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)
    ax.set_title("P@1 по темам и методам (тестовый набор)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.outline.set_edgecolor(LINE)
    fig.savefig(out)
    plt.close(fig)
    return out


# --- отчёт ---------------------------------------------------------------------------

def _examples(misses: pd.DataFrame, k: int = 4) -> list[str]:
    lines = []
    for cat, title in CATEGORIES.items():
        part = misses[misses["category"] == cat]
        if part.empty:
            continue
        lines.append(f"\n### {title} ({len(part)})\n")
        lines.append("| запрос | бот нашёл | правильный ответ | ранг |")
        lines.append("|---|---|---|---|")
        for r in part.head(k).itertuples(index=False):
            rank = "—" if pd.isna(r.first_rel_rank) else int(r.first_rel_rank)
            lines.append(f"| {r.query} | {r.top1_question} | {r.gold_question} | {rank} |")
    return lines


def write_markdown(method: str, per_query: pd.DataFrame, misses: pd.DataFrame, slang: pd.DataFrame,
                   oos_answered: pd.DataFrame, threshold: float | None, out: Path) -> Path:
    in_scope = per_query[~per_query["is_oos"]]
    counts = misses["category"].value_counts()
    dup = int(counts.get("duplicate", 0))
    p1 = in_scope["hit@1"].mean()
    p1_fixed = (in_scope["hit@1"].sum() + dup) / len(in_scope)
    n_oos = int(per_query["is_oos"].sum())
    lines = [
        f"# Разбор ошибок: {method}, тестовый набор",
        "",
        f"Запросов по теме: {len(in_scope)}, промахов top-1: {len(misses)} (P@1 = {p1:.3f}).",
        f"Из них {dup} — равноценные ответы из других кластеров (ручная проверка, "
        f"reports/errors/review_{method}.csv). Если засчитать их, P@1 = {p1_fixed:.3f}.",
        "",
        "| категория | запросов | доля промахов |",
        "|---|---|---|",
    ]
    for cat, title in CATEGORIES.items():
        n = int(counts.get(cat, 0))
        lines.append(f"| {title} | {n} | {n / max(len(misses), 1):.0%} |")
    lines += ["", "## Сленг и сокращения", "",
              "| запросы | n | P@1 | MRR |", "|---|---|---|---|"]
    for r in slang.itertuples(index=False):
        lines.append(f"| {r[0]} | {r.n} | {r.p_at_1:.3f} | {r.mrr:.3f} |")
    thr = "не задан" if threshold is None else f"{threshold:.3f}"
    lines += ["", "## Вопросы не по теме", "",
              f"Порог {thr}: бот ответил бы на {len(oos_answered)} из {n_oos} вопросов не по теме."]
    for r in oos_answered.head(6).itertuples(index=False):
        lines.append(f"- «{r.query}» → «{r.top1_question}» ({r.confidence:.2f})")
    lines += ["", "## Примеры"] + _examples(misses)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def bot_method() -> tuple[str, float | None]:
    path = config.MODELS_DIR / "chatbot.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cfg.get("method", "tfidf_char"), cfg.get("threshold")
    return "tfidf_char", None


def run(method: str | None = None, threshold: float | None = None, faq_df: pd.DataFrame | None = None,
        out_dir: Path | None = None, fig_dir: Path | None = None) -> dict:
    default_method, default_thr = bot_method()
    method = method or default_method
    threshold = threshold if threshold is not None else (default_thr if method == default_method else None)
    faq_df = faq_df if faq_df is not None else load_faq()
    out_dir = Path(out_dir or config.ERRORS_DIR)
    fig_dir = Path(fig_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_query = load_per_query(method)
    misses = analyse(per_query, faq_df, load_review(method))
    slang = slang_table(per_query)
    oos_answered = oos_false_answers(per_query, threshold)
    topics = topic_by_method()

    cols = ["query_id", "query", "category", "first_rel_rank", "gold_topic", "top1_topic",
            "top1_score", "top1_question", "gold_question", "slang"]
    misses[cols].to_csv(out_dir / f"errors_{method}.csv", index=False)
    summary = (misses["category"].value_counts().reindex(list(CATEGORIES), fill_value=0)
               .rename_axis("category").reset_index(name="n"))
    summary["title"] = summary["category"].map(CATEGORIES)
    summary.to_csv(out_dir / f"error_summary_{method}.csv", index=False)
    slang.to_csv(out_dir / f"slang_{method}.csv", index=False)
    topics.round(3).to_csv(out_dir / "p1_by_topic_method.csv")

    n_in = int((~per_query["is_oos"]).sum())
    plot_categories(misses, method, n_in, fig_dir / "err_categories.png")
    if not topics.empty:
        plot_topic_method(topics, fig_dir / "err_topic_method.png")
    write_markdown(method, per_query, misses, slang, oos_answered, threshold, out_dir / "error_analysis.md")
    return {"method": method, "n_in_scope": n_in, "n_misses": len(misses),
            "categories": summary.set_index("category")["n"].to_dict(),
            "oos_answered": len(oos_answered), "n_oos": int(per_query["is_oos"].sum())}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Разбор ошибок на тестовом наборе")
    parser.add_argument("--method", help="по умолчанию — метод бота из models/chatbot.json")
    parser.add_argument("--threshold", type=float, help="порог для вопросов не по теме")
    args = parser.parse_args(argv)
    res = run(args.method, args.threshold)
    print(f"{res['method']}: {res['n_misses']} промахов из {res['n_in_scope']}")
    for cat, n in res["categories"].items():
        print(f"  {CATEGORIES[cat]}: {n}")
    print(f"Ответил бы на вопросы не по теме: {res['oos_answered']} из {res['n_oos']}")
    print(f"Отчёт: {config.ERRORS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
