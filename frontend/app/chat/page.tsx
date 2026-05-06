"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { InputBar } from "@/components/chat/InputBar";
import { useSSEChat } from "@/hooks/useSSEChat";
import type { ModelTag, SearchMode } from "@/lib/types";
import { getHealth } from "@/lib/api";

export default function ChatPage() {
  const { messages, status, sendMessage, clearMessages, stopStreaming } = useSSEChat();
  const [modelTag, setModelTag] = useState<ModelTag>("m5");
  const [searchMode, setSearchMode] = useState<SearchMode>("system_kb");
  const [collectionId, setCollectionId] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem("aegis_collection_id") : null;
    if (stored) setCollectionId(stored);

    const check = () =>
      getHealth()
        .then(() => setIsConnected(true))
        .catch(() => setIsConnected(false));

    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  const handleCollectionChange = (id: string) => {
    setCollectionId(id);
    if (typeof window !== "undefined") localStorage.setItem("aegis_collection_id", id);
  };

  const handleSend = (query: string) => {
    sendMessage(query, modelTag, searchMode, collectionId || undefined);
  };

  return (
    <div
      className="flex h-full w-full"
      style={{ background: "var(--page-bg)" }}
    >
      <Sidebar onNewChat={clearMessages} />

      <div className="flex flex-col flex-1 min-w-0 min-h-0 overflow-hidden">
        <TopBar
          title="Chat"
          modelTag={modelTag}
          onModelChange={setModelTag}
          searchMode={searchMode}
          onSearchModeChange={setSearchMode}
          isConnected={isConnected}
        />

        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          <ChatWindow messages={messages} onSuggestion={handleSend} />
          <InputBar
            onSend={handleSend}
            onStop={stopStreaming}
            isStreaming={status === "streaming"}
            disabled={false}
            collectionId={collectionId}
            onCollectionChange={handleCollectionChange}
            placeholder={
              searchMode === "user_docs" && !collectionId
                ? "Upload a document first using (+), then switch to My Docs…"
                : undefined
            }
          />
        </div>
      </div>
    </div>
  );
}
