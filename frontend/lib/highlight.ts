// Подсветка слов ответа, совпавших с вопросом пользователя.
// Сравниваем по началу слова (грубая «основа»), чтобы «общежития» совпало с «общежитие».
// Это подсказка для человека, а не объяснение работы модели: эмбеддинги находят ответ и без общих слов.

const STOP = new Set([
  "как",
  "что",
  "где",
  "когда",
  "какой",
  "какая",
  "какие",
  "каков",
  "есть",
  "это",
  "для",
  "или",
  "при",
  "можно",
  "нужно",
  "ли",
  "мне",
  "вас",
  "вы",
  "наш",
  "ваш",
  "все",
  "так",
  "там",
  "тут",
  "the",
  "and",
  "aitu",
  "аиту",
  "университет",
  "университете",
  "университета",
]);

const WORD = /[\p{L}\p{N}]+/gu;

function stem(word: string): string {
  const w = word.toLowerCase().replace(/ё/g, "е");
  if (w.length <= 4) return w;
  return w.slice(0, Math.max(4, Math.min(6, w.length - 2)));
}

export function queryStems(query: string): string[] {
  const words = query.match(WORD) ?? [];
  return [...new Set(words.filter((w) => w.length >= 3 && !STOP.has(w.toLowerCase())).map(stem))];
}

export type Segment = { text: string; hit: boolean };

/** Делит текст на куски: совпавшие слова (hit) и всё остальное. */
export function splitHighlights(text: string, stems: string[]): Segment[] {
  if (!stems.length) return [{ text, hit: false }];
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(WORD)) {
    const word = m[0];
    const w = word.toLowerCase().replace(/ё/g, "е");
    const isHit = w.length >= 3 && stems.some((s) => w.startsWith(s));
    if (!isHit) continue;
    const start = m.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start), hit: false });
    out.push({ text: word, hit: true });
    last = start + word.length;
  }
  if (last < text.length) out.push({ text: text.slice(last), hit: false });
  return out;
}
