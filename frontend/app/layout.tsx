import type { Metadata } from "next";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

export const metadata: Metadata = {
  title: "AITU FAQ — ответы для абитуриентов",
  description:
    "Учебный проект: поиск ответа в базе вопросов приёмной комиссии Astana IT University и сравнение методов поиска.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ru" className="h-full">
      <body className="flex min-h-full flex-col">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-sheet focus:px-3 focus:py-2"
        >
          Перейти к содержанию
        </a>
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-5 pb-16 pt-10 sm:pt-14">
          {children}
        </main>
        <footer className="border-t border-line">
          <p className="mx-auto max-w-6xl px-5 py-6 text-sm text-ink-soft">
            Учебный проект магистрантов AITU, группа AAI-2602. Это не официальный сервис университета: часть ответов в
            базе актуальна на 2023 год, проверяйте важное на{" "}
            <a className="text-ultramarine underline underline-offset-2" href="https://admission.astanait.edu.kz">
              admission.astanait.edu.kz
            </a>
            .
          </p>
        </footer>
      </body>
    </html>
  );
}
