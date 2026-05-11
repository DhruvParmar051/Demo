import type {
  QueryResponse,
  TicketRecord,
  HealthResponse,
  IngestResponse,
  ModelTag,
  ConversationHistory,
} from "./types";

export const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000") + "/api/v1";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function querySync(
  query: string,
  modelTag: ModelTag,
  conversationHistory: ConversationHistory[] = []
): Promise<QueryResponse> {
  return fetchJSON<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ query, model_tag: modelTag, conversation_history: conversationHistory }),
  });
}

export async function queryBaseline(
  query: string,
  baseline: "b1" | "b2" | "b3"
): Promise<QueryResponse> {
  return fetchJSON<QueryResponse>(`/query/baseline?baseline=${baseline}`, {
    method: "POST",
    body: JSON.stringify({ query, model_tag: baseline }),
  });
}

export async function queryUserDocs(
  query: string,
  collectionId: string,
  modelTag: ModelTag = "m5"
): Promise<QueryResponse> {
  return fetchJSON<QueryResponse>(`/query/user_docs?collection_id=${encodeURIComponent(collectionId)}`, {
    method: "POST",
    body: JSON.stringify({ query, model_tag: modelTag }),
  });
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>("/health");
}

export async function getTickets(): Promise<{ tickets: TicketRecord[] }> {
  return fetchJSON<{ tickets: TicketRecord[] }>("/tickets");
}

export async function deleteSession(collectionId: string): Promise<{ status: string }> {
  return fetchJSON<{ status: string }>(`/sessions/${encodeURIComponent(collectionId)}`, {
    method: "DELETE",
  });
}

export async function ingestFiles(
  files: File[],
  collectionId: string
): Promise<IngestResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE_URL}/ingest`, {
    method: "POST",
    headers: { "X-Collection-ID": collectionId },
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<IngestResponse>;
}

