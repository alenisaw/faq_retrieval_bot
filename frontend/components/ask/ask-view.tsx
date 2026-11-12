"use client";

import { useEffect, useRef, useState } from "react";
import { AnswerSheet } from "@/components/ask/answer-sheet";
import { TopicBrowser } from "@/components/ask/topic-browser";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, type ChatResponse, type MethodInfo } from "@/lib/api";
import { queryStems } from "@/lib/highlight";
import { methodLabel } from "@/lib/utils";

const EXAMPLES = [
  "общага есть?",
  "проходной балл на computer science",
  "сколько стоит обучение",
  "есть военная кафедра?",
];

export function AskView() {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState<string>("");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [resp, setResp] = useState<ChatResponse | null>(null);
  const [asked, setAsked] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .methods()
      .then(setMethods)
      .catch(() => setMethods([]));
  }, []);

  async function search(text: string) {
    const q = text.trim();
    if (!q) {
      inputRef.current?.focus();
      return;
    }
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      const r = await api.chat(q, method || null, 3);
      setResp(r);
      setAsked(q);
    } catch (e) {
      setResp(null);
      setError(e instanceof ApiError ? e.message : "Не получилось получить ответ.");
    } finally {
      setLoading(false);
    }
  }

  const neuralNote = methods.find((m) => m.name === method)?.family === "neural";

  return (
    <div>
      <h1 className="max-w-[20ch] font-display text-[28px] font-semibold leading-[1.15] tracking-tight sm:text-[40px]">
        Что хотите узнать о поступлении в AITU?
      </h1>

      <form
        className="mt-8"
        onSubmit={(e) => {
          e.preventDefault();
          search(query);
        }}
      >
        <label htmlFor="q" className="sr-only">
          Ваш вопрос
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            id="q"
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            maxLength={500}
            autoComplete="off"
            placeholder="Например: есть ли общежитие?"
            className="h-14 w-full rounded-[10px] border border-line bg-sheet px-4 text-lg placeholder:text-ink-soft/70 focus:border-ultramarine focus:outline-none"
          />
          <Button type="submit" disabled={loading} className="h-14 px-6 text-base">
            {loading ? "Ищем…" : "Найти ответ"}
          </Button>
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => search(ex)}
                className="rounded-full border border-line bg-sheet px-3 py-1 text-sm text-ink-soft hover:border-ink/40 hover:text-ink"
              >
                {ex}
              </button>
            ))}
          </div>
          {methods.length > 0 && (
            <label className="flex items-center gap-2 text-sm text-ink-soft">
              Метод поиска
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="h-9 rounded-md border border-line bg-sheet px-2 text-ink"
              >
                <option value="">{methodLabel(methods.find((m) => m.is_default)?.name ?? "")} (по умолчанию)</option>
                {methods
                  .filter((m) => !m.is_default)
                  .map((m) => (
                    <option key={m.name} value={m.name}>
                      {methodLabel(m.name)}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>
        {neuralNote && (
          <p className="mt-2 text-sm text-ink-soft">
            Нейросетевой метод: первый вопрос может занять до минуты, пока сервер загрузит модель.
          </p>
        )}
      </form>

      <div className="mt-10">
        {loading && (
          <div className="space-y-3 rounded-md border border-line bg-sheet p-6 sm:p-8" aria-label="Загрузка">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-11/12" />
            <Skeleton className="h-5 w-2/3" />
          </div>
        )}
        {!loading && error && (
          <div role="alert" className="rounded-md border border-signal/40 bg-sheet p-6">
            <p className="text-signal">{error}</p>
            <Button variant="outline" className="mt-4" onClick={() => search(query)}>
              Повторить
            </Button>
          </div>
        )}
        {!loading && !error && resp && <AnswerSheet resp={resp} stems={queryStems(asked)} />}
      </div>

      {!resp && !loading && <TopicBrowser onPick={search} />}
    </div>
  );
}
