"use client";

import { createContext, useContext } from "react";
import { useChatSessions } from "@/hooks/useChatSessions";
import type { ChatSession } from "@/hooks/useChatSessions";
import type { Message } from "@/lib/types";

interface ChatSessionsContextValue {
  sessions: ChatSession[];
  activeSessionId: string;
  messages: Message[];
  collectionId: string;
  status: "idle" | "streaming" | "done" | "error";
  sendMessage: (query: string) => Promise<void>;
  stopStreaming: () => void;
  createNewChat: () => void;
  switchChat: (id: string) => void;
  setCollectionId: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;
}

const ChatSessionsContext = createContext<ChatSessionsContextValue | null>(null);

export function ChatSessionsProvider({ children }: { children: React.ReactNode }) {
  const value = useChatSessions();
  return (
    <ChatSessionsContext.Provider value={value}>
      {children}
    </ChatSessionsContext.Provider>
  );
}

export function useChatSessionsContext() {
  const ctx = useContext(ChatSessionsContext);
  if (!ctx) throw new Error("useChatSessionsContext must be used inside ChatSessionsProvider");
  return ctx;
}
