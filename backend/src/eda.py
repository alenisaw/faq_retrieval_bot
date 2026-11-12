"""EDA корпуса FAQ: графики в reports/figures и сводка в reports/eda_summary.json.

    python -m src.eda            # или пункт 2 консольного меню

Графики (имена фиксированы — по ним фронтенд подписывает картинки на /insights):
    eda_topics.png      сколько вопросов в каждой теме
    eda_lengths.png     длина вопросов и ответов в словах, выбросы по IQR
    eda_top_words.png   топ-20 слов до и после удаления стоп-слов
    eda_ngrams.png      частые биграммы и триграммы в вопросах (леммы)
    eda_wordcloud.png   облака слов по четырём крупным темам (нужен пакет wordcloud)
    eda_vocab.png       размер словаря и время обработки: raw / stem / lemma
    eda_clusters.png    сколько вопросов приходится на один ответ

Цвета те же, что на сайте: ультрамарин для данных, чернильный для текста.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: графики только в файлы
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .evaluation import load_faq  # noqa: E402
from .preprocessing import compare_modes, normalize, preprocess, tokenize  # noqa: E402

INK = "#172033"
INK_SOFT = "#4a5568"
ULTRA = "#2f45d8"
ULTRA_SOFT = "#9aa6ee"
LINE = "#d5dbe5"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SOFT,
    "ytick.color": INK,
    "grid.color": LINE,
})


# --- Подсчёты -------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(tokenize(normalize(text)))


def iqr_outliers(values: pd.Series) -> tuple[float, int]:
    """Верхняя граница по правилу 1.5·IQR и число значений выше неё."""
    q1, q3 = values.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    return float(upper), int((values > upper).sum())


def top_tokens(texts: list[str], k: int = 20, **kwargs) -> list[tuple[str, int]]:
    counts = Counter(tok for t in texts for tok in preprocess(t, **kwargs))
    return counts.most_common(k)


def top_ngrams(texts: list[str], n: int, k: int = 15) -> list[tuple[str, int]]:
    """Частые n-граммы по леммам без стоп-слов."""
    counts: Counter[str] = Counter()
    for t in texts:
        toks = preprocess(t, mode="lemma", remove_stopwords=True)
        counts.update(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return counts.most_common(k)


def topic_names(df: pd.DataFrame) -> pd.Series:
    return df["topic"].astype(str).map(lambda s: config.TOPICS.get(s, s))


# --- Графики --------------------------------------------------------------------

def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _barh(ax, labels: list[str], values: list[float], color: str = ULTRA) -> None:
    """Горизонтальные полосы, самая длинная сверху, значение у конца полосы."""
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=color, height=0.65)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:g}", va="center", fontsize=9, color=INK_SOFT)


def plot_topics(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    counts = topic_names(df).value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _barh(ax, counts.index.tolist(), counts.values.tolist())
    ax.set_title(f"Сколько вопросов в каждой теме (всего {len(df)})")
    ax.set_xlabel("число вопросов")
    return _save(fig, out_dir, "eda_topics.png"), counts.to_dict()


def plot_lengths(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    q_len = df["question"].map(word_count)
    a_len = df["answer"].map(word_count)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    stats = {}
    for ax, values, title, key in ((axes[0], q_len, "Вопросы", "question"), (axes[1], a_len, "Ответы", "answer")):
        upper, n_out = iqr_outliers(values)
        ax.hist(values, bins=30, color=ULTRA, edgecolor="white", linewidth=0.5)
        ax.axvline(values.median(), color=INK, linewidth=1.2, linestyle="--", label="медиана")
        ax.axvline(upper, color="#c2410c", linewidth=1, linestyle=":", label=f"граница выбросов (IQR): {n_out} шт.")
        ax.set_title(f"{title}: медиана {values.median():.0f} слов")
        ax.set_xlabel("слов в тексте")
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_ylabel("число текстов")
        stats[key] = {"median": float(values.median()), "mean": float(values.mean()), "max": int(values.max()),
                      "iqr_upper": upper, "outliers": n_out}
    fig.tight_layout()
    return _save(fig, out_dir, "eda_lengths.png"), stats


def plot_top_words(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    texts = (df["question"] + " " + df["answer"]).tolist()
    raw = top_tokens(texts, mode="raw", remove_stopwords=False)
    clean = top_tokens(texts, mode="lemma", remove_stopwords=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    _barh(axes[0], [w for w, _ in raw], [c for _, c in raw], ULTRA_SOFT)
    axes[0].set_title("До очистки: служебные слова")
    _barh(axes[1], [w for w, _ in clean], [c for _, c in clean])
    axes[1].set_title("Леммы без стоп-слов: предмет корпуса")
    for ax in axes:
        ax.set_xlabel("частота в вопросах и ответах")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_top_words.png"), {"raw": raw, "lemma_no_stopwords": clean}


def plot_ngrams(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    questions = df["question"].tolist()
    bi, tri = top_ngrams(questions, 2), top_ngrams(questions, 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    _barh(axes[0], [g for g, _ in bi], [c for _, c in bi])
    axes[0].set_title("Биграммы в вопросах")
    _barh(axes[1], [g for g, _ in tri], [c for _, c in tri])
    axes[1].set_title("Триграммы в вопросах")
    for ax in axes:
        ax.set_xlabel("частота")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_ngrams.png"), {"bigrams": bi, "trigrams": tri}


def plot_wordclouds(df: pd.DataFrame, out_dir: Path, n_topics: int = 4) -> Path | None:
    """Облака слов по крупным темам. Без пакета wordcloud — пропускаем, а не падаем."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[eda] пакет wordcloud не установлен — облака слов пропущены (pip install wordcloud)")
        return None
    # шрифт с кириллицей из поставки matplotlib, иначе вместо букв будут квадраты
    font = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    top = df["topic"].astype(str).value_counts().index[:n_topics]
    # слова, которые есть во всём корпусе, забивают облака и не отличают темы
    common = {"aitu", "университет", "it", "astana", "студент", "программа", "мочь", "это", "такой"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, slug in zip(axes.ravel(), top):
        part = df[df["topic"].astype(str) == slug]
        freqs = Counter(tok for t in part["question"] + " " + part["answer"]
                        for tok in preprocess(t, mode="lemma") if len(tok) > 2 and tok not in common)
        wc = WordCloud(width=600, height=360, background_color="white", font_path=font, max_words=60,
                       color_func=lambda *a, **k: ULTRA, random_state=config.RANDOM_SEED)
        ax.imshow(wc.generate_from_frequencies(freqs), interpolation="bilinear")
        ax.set_title(f"{config.TOPICS.get(slug, slug)} ({len(part)})")
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_wordcloud.png")


def plot_vocab(df: pd.DataFrame, out_dir: Path) -> tuple[Path, list[dict]]:
    texts = (df["question"] + " " + df["answer"]).tolist()
    table = compare_modes(texts, modes=["raw", "stem", "lemma"])
    labels = {"raw": "словоформы", "stem": "стеммы (Snowball)", "lemma": "леммы (pymorphy3)"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    names = [labels[m] for m in table["mode"]]
    _barh(axes[0], names, table["vocab_size"].tolist())
    axes[0].set_title("Размер словаря")
    axes[0].set_xlabel("уникальных токенов")
    _barh(axes[1], names, table["ms_per_doc"].round(3).tolist(), ULTRA_SOFT)
    axes[1].set_title("Время обработки")
    axes[1].set_xlabel("мс на один текст")
    fig.tight_layout()
    table.to_csv(config.METRICS_DIR / "eda_vocab.csv", index=False)
    return _save(fig, out_dir, "eda_vocab.png"), table.to_dict(orient="records")


def plot_clusters(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    sizes = df.groupby("answer_cluster").size()
    dist = sizes.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(dist.index.astype(str), dist.values, color=ULTRA, width=0.6)
    ax.set_yscale("log")
    ax.set_title(f"Вопросов на один ответ: {int((sizes > 1).sum())} ответов из {len(sizes)} повторяются")
    ax.set_xlabel("вопросов в кластере ответа")
    ax.set_ylabel("число кластеров (лог. шкала)")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    for x, v in zip(dist.index.astype(str), dist.values):
        ax.text(x, v, str(v), ha="center", va="bottom", fontsize=9, color=INK_SOFT)
    return _save(fig, out_dir, "eda_clusters.png"), {"n_clusters": int(len(sizes)),
                                                    "repeated": int((sizes > 1).sum()),
                                                    "max_size": int(sizes.max())}


# --- Запуск ---------------------------------------------------------------------

def run_all(df: pd.DataFrame | None = None, out_dir: str | Path | None = None) -> dict:
    """Строит все графики и сводку. Возвращает словарь {что: путь или числа}."""
    df = load_faq() if df is None else df
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_faqs": int(len(df))}
    files: list[str] = []
    for name, fn in (("topics", plot_topics), ("lengths", plot_lengths), ("top_words", plot_top_words),
                     ("ngrams", plot_ngrams), ("vocab", plot_vocab), ("clusters", plot_clusters)):
        path, stats = fn(df, out_dir)
        summary[name] = stats
        files.append(path.name)
        print(f"[eda] {path.name}")
    cloud = plot_wordclouds(df, out_dir)
    if cloud:
        files.append(cloud.name)
        print(f"[eda] {cloud.name}")

    summary["figures"] = files
    summary_path = config.REPORTS_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[eda] сводка: {summary_path}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="EDA корпуса FAQ: графики в reports/figures")
    parser.add_argument("--out", default=None, help="папка для графиков (по умолчанию reports/figures)")
    args = parser.parse_args(argv)
    run_all(out_dir=args.out)


if __name__ == "__main__":
    main()
