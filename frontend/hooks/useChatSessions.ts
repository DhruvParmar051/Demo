"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type { Message, Citation, ToolCall, ModelTag, QueryResponse } from "@/lib/types";
import { generateId } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORAGE_KEY = "aegis_sessions_v2";
const ACTIVE_KEY = "aegis_active_v2";
const FIXED_MODEL: ModelTag = "m5";

type ChatStatus = "idle" | "streaming" | "done" | "error";

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: Message[];
  collectionId: string;
}

// ── localStorage helpers ────────────────────────────────────────────────────

function readSessions(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const sessions: ChatSession[] = JSON.parse(raw);
    // Clear any messages stuck in streaming state from a previous refresh
    return sessions.map((s) => ({
      ...s,
      messages: s.messages.map((m) =>
        m.isStreaming ? { ...m, isStreaming: false } : m
      ),
    }));
  } catch {
    return [];
  }
}

function writeSessions(sessions: ChatSession[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function readActiveId(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(ACTIVE_KEY) ?? "";
}

function writeActiveId(id: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem(ACTIVE_KEY, id);
}

function makeSession(): ChatSession {
  return {
    id: generateId(),
    title: "New Chat",
    createdAt: Date.now(),
    messages: [],
    collectionId: "",
  };
}

// ── Hook ────────────────────────────────────────────────────────────────────

interface UseChatSessionsReturn {
  sessions: ChatSession[];
  activeSessionId: string;
  messages: Message[];
  collectionId: string;
  status: ChatStatus;
  sendMessage: (query: string) => Promise<void>;
  stopStreaming: () => void;
  createNewChat: () => void;
  switchChat: (id: string) => void;
  setCollectionId: (id: string) => void;
}

export function useChatSessions(): UseChatSessionsReturn {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [status, setStatus] = useState<ChatStatus>("idle");
  const abortRef = useRef<AbortController | null>(null);
  const sessionsRef = useRef<ChatSession[]>([]);

  // Keep ref in sync so callbacks always see current sessions
  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  // Load from localStorage once on mount
  useEffect(() => {
    const stored = readSessions();
    let activeId = readActiveId();

    if (stored.length === 0) {
      const fresh = makeSession();
      writeSessions([fresh]);
      writeActiveId(fresh.id);
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
    } else {
      const exists = stored.some((s) => s.id === activeId);
      if (!exists) activeId = stored[0].id;
      setSessions(stored);
      setActiveSessionId(activeId);
      writeActiveId(activeId);
    }
  }, []);

  // ── derived active session ────────────────────────────────────────────────

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;
  const messages = activeSession?.messages ?? [];
  const collectionId = activeSession?.collectionId ?? "";

  // ── helpers ───────────────────────────────────────────────────────────────

  const updateSession = useCallback(
    (id: string, updater: (s: ChatSession) => ChatSession) => {
      setSessions((prev) => {
        const next = prev.map((s) => (s.id === id ? updater(s) : s));
        writeSessions(next);
        return next;
      });
    },
    []
  );

  const updateMessages = useCallback(
    (sessionId: string, updater: (msgs: Message[]) => Message[]) => {
      updateSession(sessionId, (s) => ({ ...s, messages: updater(s.messages) }));
    },
    [updateSession]
  );

  // ── public actions ────────────────────────────────────────────────────────

  const createNewChat = useCallback(() => {
    abortRef.current?.abort();
    const session = makeSession();
    setSessions((prev) => {
      const next = [session, ...prev];
      writeSessions(next);
      return next;
    });
    setActiveSessionId(session.id);
    writeActiveId(session.id);
    setStatus("idle");
  }, []);

  const switchChat = useCallback((id: string) => {
    abortRef.current?.abort();
    setActiveSessionId(id);
    writeActiveId(id);
    setStatus("idle");
  }, []);

  const setCollectionId = useCallback(
    (id: string) => {
      updateSession(activeSessionId, (s) => ({ ...s, collectionId: id }));
    },
    [activeSessionId, updateSession]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
    setSessions((prev) => {
      const next = prev.map((s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.isStreaming ? { ...m, isStreaming: false } : m
        ),
      }));
      writeSessions(next);
      return next;
    });
  }, []);

  // ── sendMessage ───────────────────────────────────────────────────────────

  const sendMessage = useCallback(
    async (query: string) => {
      if (status === "streaming") return;

      const sessionId = activeSessionId;
      const session = sessionsRef.current.find((s) => s.id === sessionId);
      if (!session) return;

      const userMsg: Message = { id: generateId(), role: "user", content: query };
      const assistantId = generateId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        toolCalls: [],
        isStreaming: true,
        modelTag: FIXED_MODEL,
      };

      // Set title from first user message
      const isFirstMessage = session.messages.length === 0;
      updateSession(sessionId, (s) => ({
        ...s,
        title: isFirstMessage ? query.slice(0, 50) : s.title,
        messages: [...s.messages, userMsg, assistantMsg],
      }));
      setStatus("streaming");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        // Use user_docs endpoint when collection is set, otherwise streaming system KB
        if (session.collectionId) {
          const url = `${API_URL}/query/user_docs?collection_id=${encodeURIComponent(session.collectionId)}`;
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, model_tag: FIXED_MODEL }),
            signal: controller.signal,
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data: QueryResponse = await res.json();

          updateMessages(sessionId, (msgs) =>
            msgs.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: data.answer,
                    citations: data.citations ?? [],
                    confidence: data.confidence,
                    verifyVerdict: data.verify_verdict,
                    ticketId: data.ticket_id,
                    isStreaming: false,
                  }
                : m
            )
          );
          setStatus("done");
          return;
        }

        // Streaming system KB
        const history = session.messages
          .filter((m) => !m.isStreaming && !m.error)
          .map((m) => ({ role: m.role, content: m.content }));

        const res = await fetch(`${API_URL}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            model_tag: FIXED_MODEL,
            conversation_history: history,
          }),
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
          const normalized = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
          const lines = normalized.split("\n");
          buffer = lines.pop() ?? "";

          let eventType = "";
          let eventData = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              eventData = line.slice(6).trim();
            } else if (line === "" && eventType && eventData) {
              try {
                const parsed = JSON.parse(eventData);
                if (eventType === "token") {
                  currentContent += parsed.text ?? "";
                  updateMessages(sessionId, (msgs) =>
                    msgs.map((m) =>
                      m.id === assistantId ? { ...m, content: currentContent } : m
                    )
                  );
                } else if (eventType === "citation") {
                  currentCitations.push(parsed as Citation);
                  updateMessages(sessionId, (msgs) =>
                    msgs.map((m) =>
                      m.id === assistantId
                        ? { ...m, citations: [...currentCitations] }
                        : m
                    )
                  );
                } else if (eventType === "tool_call") {
                  currentToolCalls.push(parsed as ToolCall);
                } else if (eventType === "verify_result") {
                  updateMessages(sessionId, (msgs) =>
                    msgs.map((m) =>
                      m.id === assistantId
                        ? { ...m, verifyVerdict: parsed.verdict }
                        : m
                    )
                  );
                } else if (eventType === "done") {
                  const d = parsed as Partial<QueryResponse>;
                  updateMessages(sessionId, (msgs) =>
                    msgs.map((m) =>
                      m.id === assistantId
                        ? {
                            ...m,
                            content: d.answer ?? currentContent,
                            citations: d.citations ?? currentCitations,
                            confidence: d.confidence,
                            verifyVerdict: d.verify_verdict,
                            ticketId: d.ticket_id,
                            isStreaming: false,
                          }
                        : m
                    )
                  );
                  setStatus("done");
                } else if (eventType === "error") {
                  updateMessages(sessionId, (msgs) =>
                    msgs.map((m) =>
                      m.id === assistantId
                        ? { ...m, error: parsed.message, isStreaming: false }
                        : m
                    )
                  );
                  setStatus("error");
                }
              } catch {
                // ignore parse errors / heartbeat comments
              }
              eventType = "";
              eventData = "";
            }
          }
        }

        // Ensure streaming flag cleared if stream ended without done event
        updateMessages(sessionId, (msgs) =>
          msgs.map((m) =>
            m.id === assistantId && m.isStreaming
              ? { ...m, isStreaming: false }
              : m
          )
        );
        setStatus("done");
      } catch (err: unknown) {
        if ((err as Error).name === "AbortError") {
          setStatus("idle");
          return;
        }
        const msg = err instanceof Error ? err.message : "Unknown error";
        updateMessages(sessionId, (msgs) =>
          msgs.map((m) =>
            m.id === assistantId
              ? { ...m, error: msg, isStreaming: false, content: m.content || "" }
              : m
          )
        );
        setStatus("error");
      }
    },
    [status, activeSessionId, updateSession, updateMessages]
  );

  return {
    sessions,
    activeSessionId,
    messages,
    collectionId,
    status,
    sendMessage,
    stopStreaming,
    createNewChat,
    switchChat,
    setCollectionId,
  };
}
