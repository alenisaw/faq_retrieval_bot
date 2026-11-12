"use client";

import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type ReportInfo, type ReportResult } from "@/lib/api";

function cell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

export function ReportsSection() {
  const [reports, setReports] = useState<ReportInfo[] | null>(null);
  const [name, setName] = useState("");
  const [result, setResult] = useState<ReportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  function load(next: string) {
    setName(next);
    setResult(null);
    setError(null);
    api
      .report(next)
      .then(setResult)
      .catch((e) => setError(e.message));
  }

  useEffect(() => {
    api
      .reports()
      .then((list) => {
        setReports(list);
        if (list[0]) load(list[0].name);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (!reports && !error) return <Skeleton className="h-40 w-full" />;

  return (
    <div>
      {reports && (
        <label className="flex flex-wrap items-center gap-3 text-sm text-ink-soft">
          Отчёт
          <select
            value={name}
            onChange={(e) => load(e.target.value)}
            className="h-10 max-w-full rounded-md border border-line bg-sheet px-2 text-[15px] text-ink"
          >
            {reports.map((r) => (
              <option key={r.name} value={r.name}>
                {r.title}
              </option>
            ))}
          </select>
        </label>
      )}
      {error && <p className="mt-4 text-signal">{error}</p>}
      {result && (
        <>
          <p className="mt-3 max-w-[68ch] text-sm text-ink-soft">{result.description}</p>
          {result.rows.length === 0 ? (
            <p className="mt-4 text-ink-soft">Запрос выполнился, но строк нет — для него ещё не хватает данных.</p>
          ) : (
            <div className="mt-4 max-h-[420px] overflow-auto rounded-md border border-line bg-sheet">
              <table className="w-full text-sm">
                <thead className="sticky top-0 border-b border-line bg-sheet text-left text-ink-soft">
                  <tr>
                    {result.columns.map((c) => (
                      <th key={c} scope="col" className="whitespace-nowrap px-3 py-2 font-medium">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {result.rows.slice(0, 200).map((row, i) => (
                    <tr key={i} className="border-b border-line last:border-0">
                      {result.columns.map((c) => (
                        <td key={c} className="max-w-[420px] px-3 py-2 align-top">
                          {cell(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
