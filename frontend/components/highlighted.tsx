import { splitHighlights } from "@/lib/highlight";

/** Текст ответа с маркером на словах, совпавших с вопросом. */
export function Highlighted({ text, stems }: { text: string; stems: string[] }) {
  return (
    <>
      {splitHighlights(text, stems).map((seg, i) =>
        seg.hit ? (
          <mark key={i} className="hl">
            {seg.text}
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </>
  );
}
