// Клиент API бэкенда. Типы повторяют backend/src/api/schemas.py — меняем оба файла одним PR.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Hit = {
  faq_id: number;
  question: string;
  answer: string;
  score: number;
  answer_cluster: number | null;
  topic: string | null;
  topic_name: string | null;
  is_dated: boolean;
};

export type ChatResponse = {
  query: string;
  answered: boolean;
  message: string;
  hits: Hit[];
  method: string;
  threshold: number | null;
  predicted_topic: string | null;
  latency_ms: number;
};

export type CompareResponse = {
  query: string;
  results: ChatResponse[];
  errors: Record<string, string>;
};

export type MethodInfo = {
  name: string;
  family: "lexical" | "neural";
  is_default: boolean;
  loaded: boolean;
  description: string;
};

export type Topic = { slug: string; name: string; n_faqs: number };

export type Faq = {
  faq_id: number;
  question: string;
  answer: string;
  topic: string | null;
  topic_name: string | null;
  answer_cluster: number;
  is_dated: boolean;
  orig_class: string | null;
};

export type FaqPage = { total: number; limit: number; offset: number; items: Faq[] };

export type Row = Record<string, string | number | boolean | null>;
export type MetricsTable = { protocol: string | null; columns: string[]; rows: Row[] };
export type ReportInfo = { name: string; title: string; description: string };
export type ReportResult = ReportInfo & { columns: string[]; rows: Row[] };

export type Health = {
  status: "ok";
  n_faqs: number;
  n_clusters: number;
  default_method: string;
  threshold: number | null;
  database: boolean;
  loaded_methods: string[];
};

/** Ошибка с текстом, который можно показать человеку как есть. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(`Нет связи с сервером ${API_URL}. Проверьте, что бэкенд запущен.`);
  }
  if (!res.ok) {
    let detail = `Сервер ответил ошибкой ${res.status}.`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      /* тело не JSON — оставляем общий текст */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  methods: () => request<MethodInfo[]>("/api/methods"),
  topics: () => request<Topic[]>("/api/topics"),
  faqs: (params: { topic?: string; q?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && qs.set(k, String(v)));
    return request<FaqPage>(`/api/faqs?${qs}`);
  },
  chat: (query: string, method?: string | null, k = 3) =>
    request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify({ query, method: method || null, k }) }),
  compare: (query: string, methods?: string[], k = 3) =>
    request<CompareResponse>("/api/compare", {
      method: "POST",
      body: JSON.stringify({ query, methods: methods?.length ? methods : null, k }),
    }),
  metrics: (protocol?: "loo" | "test" | "dev" | "all") =>
    request<MetricsTable>(`/api/metrics${protocol ? `?protocol=${protocol}` : ""}`),
  reports: () => request<ReportInfo[]>("/api/reports"),
  report: (name: string) => request<ReportResult>(`/api/reports/${encodeURIComponent(name)}`),
  figures: () => request<string[]>("/api/figures"),
  figureUrl: (file: string) => `${API_URL}/figures/${encodeURIComponent(file)}`,
};
