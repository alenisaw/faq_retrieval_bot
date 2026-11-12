"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type MetricsTable, type Row } from "@/lib/api";
import { formatMs, methodLabel, pct } from "@/lib/utils";

const INK_SOFT = "#4a5568";
const LINE = "#d5dbe5";
const ULTRA = "#2f45d8";
const ULTRA_SOFT = "#9aa6ee";

const PROTOCOL_TEXT: Record<string, string> = {
  test: "ручной тестовый набор (протокол A, запросы, которых нет в базе)",
  loo: "leave-one-out по перефразам из самой базы (протокол B)",
};

const num = (r: Row, k: string) => (typeof r[k] === "number" ? (r[k] as number) : NaN);

export function MetricsSection() {
  const [table, setTable] = useState<MetricsTable | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const test = await api.metrics("test");
        setTable(test.rows.length ? test : await api.metrics("loo"));
      } catch (e) {
        setFailed(e instanceof Error ? e.message : "Метрики недоступны.");
      }
    })();
  }, []);

  if (failed) return <p className="text-signal">{failed}</p>;
  if (!table) return <Skeleton className="h-72 w-full" />;
  if (!table.rows.length)
    return <p className="text-ink-soft">Прогонов ещё нет: запустите пункт 4 меню или `python -m src.experiments`.</p>;

  const rows = [...table.rows].sort((a, b) => num(b, "mrr@10") - num(a, "mrr@10"));
  const best = rows[0];
  const baseline = rows.find((r) => r.method === "rule_based");
  const chartData = rows.map((r) => ({
    name: methodLabel(String(r.method)),
    "MRR@10": +num(r, "mrr@10").toFixed(3),
    "P@1": +num(r, "p@1").toFixed(3),
  }));
  const scatterData = rows.map((r) => ({
    name: methodLabel(String(r.method)),
    latency: Math.max(num(r, "latency_mean_ms"), 0.01),
    mrr: +num(r, "mrr@10").toFixed(3),
  }));
  const lat = scatterData.map((d) => d.latency);
  const logScale = Math.max(...lat) / Math.min(...lat) > 20; // лог-шкала, только когда в данных есть нейросети
  const protocol = String(best.protocol ?? table.protocol ?? "");

  return (
    <div className="space-y-12">
      <p className="max-w-[68ch] text-lg">
        Лучший метод сейчас — <strong>{methodLabel(String(best.method))}</strong>: правильный ответ первым в{" "}
        {pct(num(best, "p@1"))} вопросов, MRR@10 = {num(best, "mrr@10").toFixed(2)}
        {baseline && baseline !== best && (
          <> против {num(baseline, "mrr@10").toFixed(2)} у baseline на правилах</>
        )}.{" "}
        <span className="text-ink-soft">
          Оценка: {PROTOCOL_TEXT[protocol] ?? protocol}, {String(best.n_queries)} запросов.
        </span>
      </p>

      <figure>
        <figcaption className="font-medium">Качество поиска по методам</figcaption>
        <p className="text-sm text-ink-soft">Доля от 0 до 1, чем длиннее полоса, тем лучше</p>
        <div className="mt-4 h-[360px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 24, right: 32 }}>
              <CartesianGrid horizontal={false} stroke={LINE} />
              <XAxis type="number" domain={[0, 1]} tick={{ fill: INK_SOFT, fontSize: 12 }} stroke={LINE} />
              <YAxis
                type="category"
                dataKey="name"
                width={150}
                tick={{ fill: "#172033", fontSize: 13 }}
                stroke={LINE}
              />
              <Tooltip cursor={{ fill: "rgba(23,32,51,0.04)" }} />
              <Legend wrapperStyle={{ fontSize: 13 }} />
              <Bar dataKey="MRR@10" fill={ULTRA} radius={[0, 3, 3, 0]} barSize={12} />
              <Bar dataKey="P@1" fill={ULTRA_SOFT} radius={[0, 3, 3, 0]} barSize={12} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <figure>
        <figcaption className="font-medium">Скорость против качества</figcaption>
        <p className="text-sm text-ink-soft">Средняя задержка на один вопрос и MRR@10. Лучшие методы — вверху слева.</p>
        <div className="mt-4 h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ left: 8, right: 120, top: 16, bottom: 24 }}>
              <CartesianGrid stroke={LINE} />
              <XAxis
                type="number"
                dataKey="latency"
                scale={logScale ? "log" : "linear"}
                domain={logScale ? ["auto", "auto"] : [0, "auto"]}
                tickFormatter={(v: number) => (v < 1 ? v.toFixed(2) : String(Math.round(v)))}
                name="Задержка, мс"
                tick={{ fill: INK_SOFT, fontSize: 12 }}
                stroke={LINE}
                label={{
                  value: "задержка, мс",
                  position: "insideBottomRight",
                  offset: -8,
                  fill: INK_SOFT,
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="mrr"
                domain={[0, 1]}
                name="MRR@10"
                tick={{ fill: INK_SOFT, fontSize: 12 }}
                stroke={LINE}
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={scatterData} fill={ULTRA}>
                <LabelList dataKey="name" position="right" style={{ fill: "#172033", fontSize: 12 }} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <div>
        <h3 className="font-medium">Все метрики</h3>
        <div className="mt-3 overflow-x-auto rounded-md border border-line bg-sheet">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="border-b border-line text-left text-ink-soft">
              <tr>
                {["Метод", "P@1", "Hit@3", "Hit@5", "MRR@10", "Задержка", "Индекс, с", "Память, МБ"].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {rows.map((r) => (
                <tr key={String(r.method)} className="border-b border-line last:border-0">
                  <th scope="row" className="px-3 py-2 text-left font-normal">
                    {methodLabel(String(r.method))}
                  </th>
                  <td className="px-3 py-2">{pct(num(r, "p@1"))}</td>
                  <td className="px-3 py-2">{pct(num(r, "hit@3"))}</td>
                  <td className="px-3 py-2">{pct(num(r, "hit@5"))}</td>
                  <td className="px-3 py-2 font-medium">{num(r, "mrr@10").toFixed(3)}</td>
                  <td className="px-3 py-2">{formatMs(num(r, "latency_mean_ms"))}</td>
                  <td className="px-3 py-2">{num(r, "index_time_s").toFixed(2)}</td>
                  <td className="px-3 py-2">{num(r, "peak_mem_mb").toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
