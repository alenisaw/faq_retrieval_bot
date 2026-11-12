"use client";

import { useEffect, useRef, useState } from "react";
import { api, type Faq, type Topic } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

/** Пустое состояние: темы базы и вопросы внутри них. Клик по вопросу — сразу поиск. */
export function TopicBrowser({ onPick }: { onPick: (question: string) => void }) {
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [faqs, setFaqs] = useState<Faq[] | null>(null);
  const [failed, setFailed] = useState(false);
  const latest = useRef<string | null>(null);

  useEffect(() => {
    api
      .topics()
      .then(setTopics)
      .catch(() => setFailed(true));
  }, []);

  function toggle(slug: string) {
    if (active === slug) {
      setActive(null);
      return;
    }
    setActive(slug);
    setFaqs(null);
    latest.current = slug;
    api
      .faqs({ topic: slug, limit: 8 })
      .then((page) => latest.current === slug && setFaqs(page.items)) // ответ на старый клик не перетирает новый
      .catch(() => latest.current === slug && setFaqs([]));
  }

  if (failed) return null;

  return (
    <section aria-labelledby="topics-title" className="mt-14">
      <h2 id="topics-title" className="font-display text-lg font-semibold">
        Или выберите тему
      </h2>
      <div className="mt-4 grid gap-x-10 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
        {topics === null
          ? Array.from({ length: 12 }).map((_, i) => <Skeleton key={i} className="my-2 h-5 w-3/4" />)
          : topics.map((t) => (
              <button
                key={t.slug}
                type="button"
                onClick={() => toggle(t.slug)}
                aria-expanded={active === t.slug}
                className={cn(
                  "flex items-baseline justify-between gap-3 border-b py-2 text-left transition-colors",
                  active === t.slug ? "border-ultramarine text-ink" : "border-line text-ink hover:border-ink/40",
                )}
              >
                <span>{t.name}</span>
                <span className="tabular-nums text-sm text-ink-soft">{t.n_faqs}</span>
              </button>
            ))}
      </div>

      {active && (
        <div className="mt-8" aria-live="polite">
          <h3 className="font-medium">{topics?.find((t) => t.slug === active)?.name}: вопросы из базы</h3>
          <ul className="mt-3 space-y-1">
            {faqs === null
              ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="my-2 h-5 w-2/3" />)
              : faqs.map((f) => (
                  <li key={f.faq_id}>
                    <button
                      type="button"
                      onClick={() => onPick(f.question)}
                      className="rounded py-1 text-left text-ultramarine underline decoration-ultramarine/30 underline-offset-4 hover:decoration-ultramarine"
                    >
                      {f.question}
                    </button>
                  </li>
                ))}
          </ul>
        </div>
      )}
    </section>
  );
}
