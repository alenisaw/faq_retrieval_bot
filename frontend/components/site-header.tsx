"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Спросить" },
  { href: "/compare", label: "Сравнить методы" },
  { href: "/insights", label: "Как это работает" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b border-line bg-paper/90 backdrop-blur supports-[backdrop-filter]:bg-paper/75 sticky top-0 z-10">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-2 px-5 py-3">
        <Link href="/" className="font-display text-[15px] font-semibold tracking-tight">
          AITU FAQ
        </Link>
        <nav aria-label="Разделы" className="-mx-2 flex gap-1 overflow-x-auto">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "whitespace-nowrap border-b-2 px-2 py-1.5 text-[15px] transition-colors",
                  active ? "border-ultramarine text-ink" : "border-transparent text-ink-soft hover:text-ink",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
