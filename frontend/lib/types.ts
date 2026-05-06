// TypeScript types mirroring src/data/schema.py

export interface Citation {
  doc_id: string;
  chunk_id: string;
  span_start: number;
  span_end: number;
  cited_text: string;
  source: string;
  page_number?: number | null;
  source_url?: string | null;
  verified?: boolean | null;
}

export interface ToolCall {
  tool_name: "SearchKB" | "GetPolicy" | "CreateTicket" | "AnswerVerify" | "AnswerDirect";
  args: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  latency_ms: number;
  iteration: number;
  confidence_before?: number | null;
  confidence_after?: number | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  tool_calls: ToolCall[];
  confidence: number;
  cgal_iterations: number;
  fcrs?: number | null;
  ticket_id?: string | null;
  latency_ms: number;
  ttft_ms?: number | null;
  session_id: string;
  decomposed: boolean;
  sub_queries: string[];
  alpha?: number | null;
  verify_verdict?: "pass" | "partial" | "fail" | "skipped" | null;
  model_tag: string;
}

export interface TicketRecord {
  ticket_id: string;
  session_id: string;
  query: string;
  category: "billing" | "technical" | "account" | "policy" | "other";
  severity: "low" | "medium" | "high" | "critical";
  estimated_response_time: string;
  created_at?: string;
}

export type ModelTag = "b1" | "b2" | "b3" | "m1" | "m2" | "m3" | "m4" | "m5";
export type SearchMode = "system_kb" | "user_docs";

// SSE event types
export interface TokenEvent { type: "token"; text: string }
export interface CitationEvent { type: "citation"; data: Citation }
export interface ToolCallEvent { type: "tool_call"; data: ToolCall }
export interface VerifyStartEvent { type: "verify_start" }
export interface VerifyResultEvent {
  type: "verify_result";
  verdict: string;
  grounding_score?: number;
  ungrounded_claims?: string[];
}
export interface DoneEvent { type: "done"; data: Partial<QueryResponse> }
export interface ErrorEvent { type: "error"; message: string; error_type?: string }

export type SSEEvent =
  | TokenEvent
  | CitationEvent
  | ToolCallEvent
  | VerifyStartEvent
  | VerifyResultEvent
  | DoneEvent
  | ErrorEvent;

// Chat message
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  toolCalls?: ToolCall[];
  confidence?: number;
  verifyVerdict?: "pass" | "partial" | "fail" | "skipped" | null;
  decomposed?: boolean;
  subQueries?: string[];
  latencyMs?: number;
  ttftMs?: number | null;
  cgalIterations?: number;
  alpha?: number | null;
  modelTag?: string;
  ticketId?: string | null;
  isStreaming?: boolean;
  error?: string;
}

export interface ConversationHistory {
  role: "user" | "assistant";
  content: string;
}

export interface HealthResponse {
  status: string;
  device: string;
  models_cached: string[];
}

export interface MetricsStats {
  total_queries: number;
  escalations: number;
  avg_latency_ms: number;
  avg_confidence: number;
  queries_by_tag: Record<string, number>;
}

export interface IngestResponse {
  status: string;
  files_accepted: string[];
  files_rejected: { filename: string; reason: string }[];
  chunks_added: number;
  doc_ids: string[];
  latency_ms: number;
}
