"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

// Подписи к графикам, которые строит backend/src/eda.py (имя файла -> что на нём).
const CAPTIONS: Record<string, string> = {
  "eda_topics.png": "Сколько вопросов в каждой теме",
  "eda_lengths.png": "Длина вопросов и ответов в словах",
  "eda_top_words.png": "Самые частые слова до и после удаления стоп-слов",
  "eda_ngrams.png": "Частые биграммы и триграммы в вопросах",
  "eda_wordcloud.png": "Облако слов по крупным темам",
  "eda_vocab.png": "Размер словаря: исходные слова, стеммы, леммы",
  "eda_clusters.png": "Сколько вопросов приходится на один ответ",
  "eda_umap.html": "Карта вопросов в пространстве эмбеддингов (интерактивная)",
};

export function FiguresSection() {
  const [files, setFiles] = useState<string[] | null>(null);

  useEffect(() => {
    api
      .figures()
      .then(setFiles)
      .catch(() => setFiles([]));
  }, []);

  if (files === null) return null;
  if (!files.length) return <p className="text-ink-soft">Графиков пока нет: запустите пункт 2 меню (EDA).</p>;

  const images = files.filter((f) => /\.(png|svg)$/i.test(f));
  const pages = files.filter((f) => /\.html$/i.test(f));

  return (
    <div>
      <div className="grid gap-8 md:grid-cols-2">
        {images.map((f) => (
          <figure key={f} className="rounded-md border border-line bg-sheet p-3">
            {/* eslint-disable-next-line @next/next/no-img-element -- картинки отдаёт API, а не Next */}
            <img src={api.figureUrl(f)} alt={CAPTIONS[f] ?? f} loading="lazy" className="w-full" />
            <figcaption className="mt-2 text-sm text-ink-soft">{CAPTIONS[f] ?? f}</figcaption>
          </figure>
        ))}
      </div>
      {pages.length > 0 && (
        <ul className="mt-6 space-y-1">
          {pages.map((f) => (
            <li key={f}>
              <a
                href={api.figureUrl(f)}
                target="_blank"
                rel="noreferrer"
                className="text-ultramarine underline underline-offset-4"
              >
                {CAPTIONS[f] ?? f}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
