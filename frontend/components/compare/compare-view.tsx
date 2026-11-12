"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScoreBar } from "@/components/ui/score-bar";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, type CompareResponse, type MethodInfo } from "@/lib/api";
import { cn, formatMs, methodLabel } from "@/lib/utils";

// Цвета для ответов, которые нашли несколько методов. Светлые, чтобы текст оставался читаемым.
const CLUSTER_TINTS = ["#e7ecff", "#fff3c4", "#dff3ea", "#fde4e1", "#efe4fb", "#e2f1f8"];

export function CompareView() {
  const [query, setQuery] = useState("проходной балл на software engineering");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [withNeural, setWithNeural] = useState(false);
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .methods()
      .then(setMethods)
      .catch(() => setMethods([]));
  }, []);

  async function run() {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const chosen = withNeural ? methods.map((m) => m.name) : undefined; // undefined -> все лексические
      setData(await api.compare(q, chosen, 3));
    } catch (e) {
      setData(null);
      setError(e instanceof ApiError ? e.message : "Не получилось сравнить методы.");
    } finally {
      setLoading(false);
    }
  }

  // Кластер ответа -> цвет, только если ответ нашли хотя бы два метода.
  const tints = useMemo(() => {
    const seen = new Map<number, number>();
    data?.results.forEach((r) =>
      new Set(r.hits.map((h) => h.answer_cluster ?? -h.faq_id)).forEach((c) => seen.set(c, (seen.get(c) ?? 0) + 1)),
    );
    const map = new Map<number, string>();
    [...seen.entries()]
      .filter(([, n]) => n > 1)
      .sort((a, b) => b[1] - a[1])
      .forEach(([c], i) => i < CLUSTER_TINTS.length && map.set(c, CLUSTER_TINTS[i]));
    return map;
  }, [data]);

  const describe = (name: string) => methods.find((m) => m.name === name)?.description;

  // Итог: какой ответ чаще всего стоит первым и кто с этим не согласен
  const consensus = useMemo(() => {
    const tops = (data?.results ?? [])
      .filter((r) => r.hits[0])
      .map((r) => ({ method: r.method, key: r.hits[0].answer_cluster ?? -r.hits[0].faq_id, q: r.hits[0].question }));
    if (!tops.length) return null;
    const counts = new Map<number, number>();
    tops.forEach((t) => counts.set(t.key, (counts.get(t.key) ?? 0) + 1));
    const [key, n] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    return {
      n,
      total: tops.length,
      question: tops.find((t) => t.key === key)!.q,
      tint: tints.get(key),
      dissent: tops.filter((t) => t.key !== key).map((t) => methodLabel(t.method)),
    };
  }, [data, tints]);

  return (
    <div>
      <h1 className="font-display text-[28px] font-semibold leading-[1.15] tracking-tight sm:text-[36px]">
        Один вопрос, разные методы поиска
      </h1>
      <p className="mt-3 max-w-[68ch] text-ink-soft">
        Каждая колонка — отдельный метод и его три лучших ответа. Одинаковым цветом отмечен один и тот же ответ,
        найденный разными методами; где цвета расходятся, методы не согласны.
      </p>

      <form
        className="mt-8 flex flex-col gap-3 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <label htmlFor="cq" className="sr-only">
          Вопрос для сравнения
        </label>
        <input
          id="cq"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={500}
          className="h-12 w-full rounded-[10px] border border-line bg-sheet px-4 text-base focus:border-ultramarine focus:outline-none"
        />
        <Button type="submit" disabled={loading} className="h-12 px-6">
          {loading ? "Сравниваем…" : "Сравнить методы"}
        </Button>
      </form>
      <label className="mt-3 flex items-center gap-2 text-sm text-ink-soft">
        <input
          type="checkbox"
          checked={withNeural}
          onChange={(e) => setWithNeural(e.target.checked)}
          className="size-4 accent-[var(--ultramarine)]"
        />
        Добавить нейросетевые методы (e5, MiniLM, гибрид) — первый запуск может занять минуту
      </label>

      <div className="mt-10">
        {error && (
          <p role="alert" className="rounded-md border border-signal/40 bg-sheet p-5 text-signal">
            {error}
          </p>
        )}

        {loading && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="space-y-3 rounded-md border border-line bg-sheet p-5">
                <Skeleton className="h-5 w-1/2" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
              </div>
            ))}
          </div>
        )}

        {!loading && data && (
          <>
            {consensus && (
              <div className="mb-8" aria-live="polite">
                <p className="font-display text-xl font-semibold tracking-tight sm:text-2xl">
                  {consensus.n === consensus.total
                    ? `Все ${consensus.total} методов первым ставят один ответ`
                    : `${consensus.n} из ${consensus.total} методов первым ставят один ответ`}
                </p>
                <p className="mt-2 max-w-[68ch]">
                  <span
                    className="rounded px-1.5 py-0.5"
                    style={consensus.tint ? { background: consensus.tint } : undefined}
                  >
                    «{consensus.question}»
                  </span>
                </p>
                {consensus.dissent.length > 0 && (
                  <p className="mt-2 text-sm text-ink-soft">Не согласны: {consensus.dissent.join(", ")}.</p>
                )}
              </div>
            )}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.results.map((r) => (
                <section key={r.method} className="rounded-md border border-line bg-sheet p-5">
                  <header className="flex items-baseline justify-between gap-3">
                    <h2 className="font-semibold">{methodLabel(r.method)}</h2>
                    <span className="text-sm tabular-nums text-ink-soft">{formatMs(r.latency_ms)}</span>
                  </header>
                  {describe(r.method) && <p className="mt-1 text-sm text-ink-soft">{describe(r.method)}</p>}
                  {!r.answered && (
                    <p className="mt-3 text-sm text-signal">Метод отказался отвечать: совпадение ниже порога.</p>
                  )}
                  <ol className="mt-4 space-y-2">
                    {/* шкалы в колонке — относительно лучшего ответа метода: у BM25 и правил score не в [0, 1] */}
                    {r.hits.map((h, i) => {
                      const tint = tints.get(h.answer_cluster ?? -h.faq_id);
                      return (
                        <li
                          key={h.faq_id}
                          className={cn("rounded-md border px-3 py-2", tint ? "border-transparent" : "border-line")}
                          style={tint ? { background: tint } : undefined}
                        >
                          <div className="flex items-start gap-2">
                            <span className="mt-px text-sm tabular-nums text-ink-soft">{i + 1}</span>
                            <div className="min-w-0">
                              <p className="text-[15px] leading-snug">{h.question}</p>
                              <div className="mt-1 flex flex-wrap items-center gap-x-3 text-xs">
                                <ScoreBar score={h.score} max={Math.max(1, r.hits[0]?.score ?? 1)} />
                                {h.topic_name && <span className="text-ink-soft">{h.topic_name}</span>}
                              </div>
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                </section>
              ))}
            </div>

            {Object.keys(data.errors).length > 0 && (
              <div className="mt-6 rounded-md border border-line bg-sheet p-5 text-sm">
                <p className="font-medium">Эти методы сейчас недоступны:</p>
                <ul className="mt-2 space-y-1 text-ink-soft">
                  {Object.entries(data.errors).map(([m, msg]) => (
                    <li key={m}>
                      <span className="text-ink">{methodLabel(m)}</span>: {msg}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {!loading && !data && !error && (
          <p className="text-ink-soft">Нажмите «Сравнить методы», чтобы увидеть ответы всех лексических методов.</p>
        )}
      </div>
    </div>
  );
}
