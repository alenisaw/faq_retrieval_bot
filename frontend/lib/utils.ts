import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Человеческие названия методов; ключи — из backend/src/retrievers. */
export const METHOD_LABELS: Record<string, string> = {
  rule_based: "Правила (baseline)",
  bow: "Мешок слов",
  tfidf_word_lemma: "TF-IDF, леммы",
  tfidf_word_stem: "TF-IDF, стеммы",
  tfidf_char: "TF-IDF, символы",
  bm25: "BM25",
  e5_small: "e5-small",
  minilm: "MiniLM",
  hybrid_rrf: "Гибрид (RRF)",
};

export const methodLabel = (name: string) => METHOD_LABELS[name] ?? name;

export function formatMs(ms: number): string {
  if (ms < 1) return "< 1 мс";
  if (ms < 1000) return `${Math.round(ms)} мс`;
  return `${(ms / 1000).toFixed(1)} с`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}
