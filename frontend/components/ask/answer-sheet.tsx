import { Highlighted } from "@/components/highlighted";
import { ScoreBar } from "@/components/ui/score-bar";
import type { ChatResponse } from "@/lib/api";
import { confidence } from "@/lib/confidence";
import { cn, formatMs, methodLabel } from "@/lib/utils";

const DOT: Record<string, string> = {
  clear: "bg-ultramarine",
  single: "bg-ultramarine",
  likely: "bg-ultramarine/50",
  close: "bg-signal",
};

export function AnswerSheet({ resp, stems }: { resp: ChatResponse; stems: string[] }) {
  const [best, ...rest] = resp.hits;
  const scale = Math.max(1, best?.score ?? 1); // BM25 и правила дают score > 1
  const conf = confidence(resp.hits);

  if (!resp.answered || !best) {
    return (
      <section aria-live="polite" className="rounded-md border border-signal/40 bg-sheet p-6 sm:p-8">
        <h2 className="font-display text-lg font-semibold text-signal">В базе нет ответа на этот вопрос</h2>
        <p className="mt-2 max-w-[68ch] text-ink">{resp.message}</p>
        <p className="mt-3 text-sm text-ink-soft">Попробуйте спросить иначе или посмотрите вопросы по темам ниже.</p>
      </section>
    );
  }

  return (
    <section aria-live="polite" aria-label="Ответ" className="space-y-6">
      <article className="rounded-md border border-line bg-sheet p-6 shadow-[0_1px_0_rgba(23,32,51,0.04)] sm:p-8">
        {conf.text && (
          <p className={cn("mb-4 flex items-center gap-2 text-sm", conf.level === "close" ? "text-signal" : "text-ink-soft")}>
            <span aria-hidden className={cn("size-2 rounded-full", DOT[conf.level])} />
            {conf.text}
          </p>
        )}
        <p className="max-w-[68ch] whitespace-pre-line text-[18px] leading-[1.65]">
          <Highlighted text={best.answer} stems={stems} />
        </p>

        {best.is_dated && (
          <p className="mt-4 max-w-[68ch] text-sm text-signal">
            В ответе есть даты или старое название столицы — информация могла устареть.
          </p>
        )}

        <dl className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line pt-4 text-sm">
          {best.topic_name && (
            <div className="flex gap-1.5">
              <dt className="text-ink-soft">Тема</dt>
              <dd>{best.topic_name}</dd>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <dt className="text-ink-soft">Совпадение</dt>
            <dd>
              <ScoreBar score={best.score} max={scale} />
            </dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-ink-soft">Метод</dt>
            <dd>{methodLabel(resp.method)}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-ink-soft">Время</dt>
            <dd className="tabular-nums">{formatMs(resp.latency_ms)}</dd>
          </div>
        </dl>
        <p className="mt-3 text-sm text-ink-soft">
          Найдено по вопросу из базы: «<Highlighted text={best.question} stems={stems} />»
        </p>
      </article>

      {rest.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-ink-soft">Если это не то, посмотрите похожие ответы</h2>
          <ul className="mt-2 divide-y divide-line border-y border-line">
            {rest.map((hit) => (
              <li key={hit.faq_id}>
                <details className="group py-3">
                  <summary className="flex cursor-pointer list-none items-start justify-between gap-4 rounded [&::-webkit-details-marker]:hidden">
                    <span>{hit.question}</span>
                    <span className="flex shrink-0 items-center gap-3 text-sm">
                      <ScoreBar score={hit.score} max={scale} />
                      <span aria-hidden className="text-ink-soft transition-transform group-open:rotate-90">
                        ›
                      </span>
                    </span>
                  </summary>
                  <p className="mt-2 max-w-[68ch] whitespace-pre-line text-ink-soft">
                    <Highlighted text={hit.answer} stems={stems} />
                  </p>
                </details>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
