"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type { Message, ModelTag, QueryResponse } from "@/lib/types";
import { generateId } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORAGE_KEY = "aegis_sessions_v2";
const ACTIVE_KEY = "aegis_active_v2";
const FIXED_MODEL: ModelTag = "m5";

const CHITCHAT_RESPONSES: Record<string, string> = {
  greeting: "Hello! I'm AegisRAG, your document-grounded assistant. Upload a document using the + button and I'll answer questions based on its contents.",
  howAreYou: "I'm doing great, thanks for asking! Upload a document and I'll get to work answering your questions.",
  thanks: "You're welcome! Let me know if you have any other questions about your documents.",
  bye: "Goodbye! Come back anytime you need help with your documents.",
  whoAreYou: "I'm AegisRAG — an evidence-based RAG assistant. I answer questions grounded in documents you upload. Use the + button to get started.",
  whatCanYouDo: "I can answer questions based on documents you upload (PDF, DOCX, TXT, MD). Upload a file using the + button, then ask me anything about it.",
  help: "To get started: click the + button to upload a document, then type your question. I'll answer based only on what's in your document.",
};

function getChitchatResponse(query: string): string | null {
  const q = query.trim().toLowerCase();
  if (/^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))/i.test(q)) return CHITCHAT_RESPONSES.greeting;
  if (/^how are you/i.test(q)) return CHITCHAT_RESPONSES.howAreYou;
  if (/^(thanks|thank you|thx|ty)/i.test(q)) return CHITCHAT_RESPONSES.thanks;
  if (/^(bye|goodbye|see you|take care)/i.test(q)) return CHITCHAT_RESPONSES.bye;
  if (/^who are you/i.test(q)) return CHITCHAT_RESPONSES.whoAreYou;
  if (/^what (are|can) you do/i.test(q)) return CHITCHAT_RESPONSES.whatCanYouDo;
  if (/^help[!?.]?$/i.test(q)) return CHITCHAT_RESPONSES.help;
  if (/^(ok|okay|got it|understood|sounds good|great|cool|nice|yes|no|sure|maybe|absolutely|definitely)[!.,]?$/i.test(q)) return "Got it! Feel free to ask me anything about your uploaded documents.";
  if (/^what('s| is) up/i.test(q)) return CHITCHAT_RESPONSES.greeting;
  return null;
}

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
  sendMessage: (query: string, attachedFiles?: { name: string; size: number; type: string }[]) => Promise<void>;
  stopStreaming: () => void;
  createNewChat: () => void;
  switchChat: (id: string) => void;
  setCollectionId: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;
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

  const renameSession = useCallback(
    (id: string, title: string) => {
      updateSession(id, (s) => ({ ...s, title: title.trim() || s.title }));
    },
    [updateSession]
  );

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const next = prev.filter((s) => s.id !== id);
        writeSessions(next);
        // If we deleted the active session, switch to the first remaining one (or create new)
        if (id === activeSessionId) {
          if (next.length > 0) {
            writeActiveId(next[0].id);
            setActiveSessionId(next[0].id);
          } else {
            const fresh = makeSession();
            writeSessions([fresh]);
            writeActiveId(fresh.id);
            setActiveSessionId(fresh.id);
            return [fresh];
          }
        }
        return next;
      });
    },
    [activeSessionId]
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
    async (query: string, attachedFiles?: { name: string; size: number; type: string }[]) => {
      if (status === "streaming") return;

      const sessionId = activeSessionId;
      const session = sessionsRef.current.find((s) => s.id === sessionId);
      if (!session) return;

      const userMsg: Message = { id: generateId(), role: "user", content: query, attachedFiles };
      const assistantId = generateId();

      // Set title from first user message
      const isFirstMessage = session.messages.length === 0;

      // ── Chitchat fast-path (client-side, no API call) ──────────────────────
      const chitchatReply = getChitchatResponse(query);
      if (chitchatReply) {
        updateSession(sessionId, (s) => ({
          ...s,
          title: isFirstMessage ? query.slice(0, 50) : s.title,
          messages: [...s.messages, userMsg, {
            id: assistantId,
            role: "assistant" as const,
            content: chitchatReply,
            citations: [],
            toolCalls: [],
            isStreaming: false,
            modelTag: FIXED_MODEL,
          }],
        }));
        return;
      }

      // ── No document uploaded — prompt user to upload ────────────────────────
      if (!session.collectionId) {
        updateSession(sessionId, (s) => ({
          ...s,
          title: isFirstMessage ? query.slice(0, 50) : s.title,
          messages: [...s.messages, userMsg, {
            id: assistantId,
            role: "assistant" as const,
            content: "To answer your question, I need a document to reference. Please upload a PDF, DOCX, TXT, or MD file using the **+** button, then ask your question again.",
            citations: [],
            toolCalls: [],
            isStreaming: false,
            modelTag: FIXED_MODEL,
          }],
        }));
        return;
      }

      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        toolCalls: [],
        isStreaming: true,
        modelTag: FIXED_MODEL,
      };

      updateSession(sessionId, (s) => ({
        ...s,
        title: isFirstMessage ? query.slice(0, 50) : s.title,
        messages: [...s.messages, userMsg, assistantMsg],
      }));
      setStatus("streaming");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        // Use user_docs endpoint when collection is set
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
    renameSession,
    deleteSession,
  };
}
