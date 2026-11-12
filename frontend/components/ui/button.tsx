import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "outline" | "ghost";

const variants: Record<Variant, string> = {
  primary: "bg-ultramarine text-white hover:bg-[#2537b8] disabled:bg-ultramarine/50",
  outline: "border border-line bg-sheet text-ink hover:border-ink/40 disabled:opacity-50",
  ghost: "text-ink-soft hover:text-ink hover:bg-ink/5 disabled:opacity-50",
};

export function Button({
  className,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-4 font-medium transition-colors",
        "h-11 text-[15px] disabled:cursor-not-allowed",
        variants[variant],
        className,
      )}
      {...props}
    />
  );
}
