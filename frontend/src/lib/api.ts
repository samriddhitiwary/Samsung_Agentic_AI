export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type QueryResult = {
  rank: number;
  score: number;
  path: string;
  start_line?: number;
  end_line?: number;
  symbol?: string;
  symbol_type?: string;
  commit?: string;
  snippet?: string;
  evidence_type?: string;
  referenced_symbol?: string;
  ordered_calls?: Array<{ name?: string; raw?: string; line?: number }>;
  limitation?: string;
};

export type QueryResponse = {
  query: string;
  query_type: string;
  repo_id: string;
  commit: string;
  timing: Record<string, number | null>;
  trace: Array<Record<string, unknown>>;
  results: QueryResult[];
};

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // keep HTTP status
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

