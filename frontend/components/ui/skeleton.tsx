import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div aria-hidden className={cn("animate-pulse rounded bg-ink/[0.07] motion-reduce:animate-none", className)} />
  );
}
