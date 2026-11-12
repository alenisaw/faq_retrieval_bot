import { cn } from "@/lib/utils";

/** Шкала совпадения 0..1. Подпись — для экранных читалок и для тех, кто хочет число. */
export function ScoreBar({ score, className, max = 1 }: { score: number; className?: string; max?: number }) {
  const ratio = Math.max(0, Math.min(1, score / max));
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span className="relative h-1.5 w-20 overflow-hidden rounded-full bg-line" aria-hidden>
        <span className="absolute inset-y-0 left-0 rounded-full bg-ultramarine" style={{ width: `${ratio * 100}%` }} />
      </span>
      <span className="tabular-nums text-ink-soft">{score.toFixed(2)}</span>
    </span>
  );
}
