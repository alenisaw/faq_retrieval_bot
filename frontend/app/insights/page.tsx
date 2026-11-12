import type { Metadata } from "next";
import { FiguresSection } from "@/components/insights/figures-section";
import { MetricsSection } from "@/components/insights/metrics-section";
import { ReportsSection } from "@/components/insights/reports-section";

export const metadata: Metadata = { title: "Как это работает — AITU FAQ" };

export default function Page() {
  return (
    <div>
      <h1 className="font-display text-[28px] font-semibold leading-[1.15] tracking-tight sm:text-[36px]">
        Как бот находит ответ
      </h1>
      <p className="mt-3 max-w-[68ch] text-ink-soft">
        В базе 535 вопросов приёмной комиссии после очистки. Бот сравнивает ваш вопрос с каждым из них и показывает
        ответ самого похожего. Ниже — насколько хорошо с этим справляются разные методы, данные из SQL-отчётов и графики
        исследования корпуса.
      </p>

      <section aria-labelledby="m" className="mt-12">
        <h2 id="m" className="font-display text-xl font-semibold">
          Качество методов
        </h2>
        <div className="mt-5">
          <MetricsSection />
        </div>
      </section>

      <section aria-labelledby="r" className="mt-16">
        <h2 id="r" className="font-display text-xl font-semibold">
          SQL-отчёты из базы
        </h2>
        <div className="mt-5">
          <ReportsSection />
        </div>
      </section>

      <section aria-labelledby="f" className="mt-16">
        <h2 id="f" className="font-display text-xl font-semibold">
          Что внутри корпуса
        </h2>
        <div className="mt-5">
          <FiguresSection />
        </div>
      </section>
    </div>
  );
}
