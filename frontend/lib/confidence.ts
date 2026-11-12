import type { Hit } from "@/lib/api";

export type Confidence = { level: "clear" | "likely" | "close" | "single"; text: string };

/**
 * Насколько первый ответ оторвался от второго. Считаем по отношению score, а не по самому score:
 * у BM25 и правил шкала не 0..1, а отношение сравнимо между методами.
 * Это подсказка пользователю, а не вероятность — пороги подобраны на глаз по разбору примеров.
 */
export function confidence(hits: Hit[]): Confidence {
  const [a, b] = hits;
  if (!a) return { level: "close", text: "" };
  if (!b || b.score <= 0) return { level: "single", text: "Подходящий ответ один" };
  if (a.answer_cluster !== null && a.answer_cluster === b.answer_cluster)
    return { level: "clear", text: "Два лучших совпадения ведут к одному ответу" };
  const ratio = a.score / b.score;
  if (ratio >= 1.5) return { level: "clear", text: "Явный лидер: другие ответы сильно дальше" };
  if (ratio >= 1.15) return { level: "likely", text: "Скорее всего, это он" };
  return { level: "close", text: "Есть близкие варианты — загляните в похожие ответы ниже" };
}
