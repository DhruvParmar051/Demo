"use client";

import { useState, useCallback, useRef } from "react";
import type { Message, Citation, ToolCall, ModelTag, ConversationHistory, QueryResponse } from "@/lib/types";
import { generateId } from "@/lib/utils";

// Re-export BASE_URL from utils won't work — get it from env directly
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type ChatStatus = "idle" | "streaming" | "done" | "error";

interface UseSSEChatReturn {
  messages: Message[];
  status: ChatStatus;
  sendMessage: (query: string, modelTag: ModelTag, searchMode: "system_kb" | "user_docs", collectionId?: string) => Promise<void>;
  clearMessages: () => void;
  stopStreaming: () => void;
}

export function useSSEChat(): UseSSEChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const abortRef = useRef<AbortController | null>(null);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStatus("idle");
  }, []);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  }, []);

  const buildHistory = (msgs: Message[]): ConversationHistory[] =>
    msgs
      .filter((m) => !m.isStreaming && !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

  const sendMessage = useCallback(
    async (
      query: string,
      modelTag: ModelTag,
      searchMode: "system_kb" | "user_docs",
      collectionId?: string
    ) => {
      if (status === "streaming") return;

      const userMsg: Message = { id: generateId(), role: "user", content: query };
      const assistantId = generateId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        toolCalls: [],
        isStreaming: true,
        modelTag,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setStatus("streaming");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        let url: string;
        let body: object;

        if (searchMode === "user_docs" && collectionId) {
          // User docs: sync endpoint (no streaming available)
          url = `${API_URL}/query/user_docs?collection_id=${encodeURIComponent(collectionId)}`;
          body = { query, model_tag: modelTag };
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
          });
          const data: QueryResponse = await res.json();
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: data.answer,
                    citations: data.citations ?? [],
                    toolCalls: data.tool_calls ?? [],
                    confidence: data.confidence,
                    verifyVerdict: data.verify_verdict,
                    latencyMs: data.latency_ms,
                    isStreaming: false,
                  }
                : m
            )
          );
          setStatus("done");
          return;
        }

        // Streaming endpoint
        const history = buildHistory(messages);
        url = `${API_URL}/query/stream`;
        body = { query, model_tag: modelTag, conversation_history: history };

        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentContent = "";
        const currentCitations: Citation[] = [];
        const currentToolCalls: ToolCall[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          let eventType = "";
          let eventData = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              eventData = line.slice(6).trim();
            } else if (line === "" && eventType && eventData) {
              // Process event
              try {
                const parsed = JSON.parse(eventData);
                if (eventType === "token") {
                  currentContent += parsed.text ?? "";
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId ? { ...m, content: currentContent } : m
                    )
                  );
                } else if (eventType === "citation") {
                  currentCitations.push(parsed as Citation);
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId ? { ...m, citations: [...currentCitations] } : m
                    )
                  );
                } else if (eventType === "tool_call") {
                  currentToolCalls.push(parsed as ToolCall);
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId ? { ...m, toolCalls: [...currentToolCalls] } : m
                    )
                  );
                } else if (eventType === "verify_result") {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId ? { ...m, verifyVerdict: parsed.verdict } : m
                    )
                  );
                } else if (eventType === "done") {
                  const d = parsed as Partial<QueryResponse>;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? {
                            ...m,
                            content: d.answer ?? currentContent,
                            citations: d.citations ?? currentCitations,
                            toolCalls: d.tool_calls ?? currentToolCalls,
                            confidence: d.confidence,
                            verifyVerdict: d.verify_verdict,
                            decomposed: d.decomposed,
                            subQueries: d.sub_queries,
                            latencyMs: d.latency_ms,
                            ttftMs: d.ttft_ms,
                            cgalIterations: d.cgal_iterations,
                            alpha: d.alpha,
                            ticketId: d.ticket_id,
                            isStreaming: false,
                          }
                        : m
                    )
                  );
                  setStatus("done");
                } else if (eventType === "error") {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, error: parsed.message, isStreaming: false }
                        : m
                    )
                  );
                  setStatus("error");
                }
              } catch {
                // ignore parse errors for heartbeat comments
              }
              eventType = "";
              eventData = "";
            } else if (line.startsWith(": ")) {
              // SSE comment (heartbeat) — ignore
            }
          }
        }

        // Ensure streaming flag is cleared
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId && m.isStreaming ? { ...m, isStreaming: false } : m
          )
        );
        setStatus("done");
      } catch (err: unknown) {
        if ((err as Error).name === "AbortError") {
          setStatus("idle");
          return;
        }
        const msg = err instanceof Error ? err.message : "Unknown error";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, error: msg, isStreaming: false, content: m.content || "An error occurred." }
              : m
          )
        );
        setStatus("error");
      }
    },
    [messages, status]
  );

  return { messages, status, sendMessage, clearMessages, stopStreaming };
}
