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

    getHealth()
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));
  }, []);

  const handleSend = (query: string) => {
    sendMessage(query, modelTag, searchMode, collectionId || undefined);
  };

  return (
    <div className="flex h-full" style={{ background: "#0a0a0f" }}>
      <Sidebar onNewChat={clearMessages} />

      <div className="flex flex-col flex-1 min-w-0">
        <TopBar
          title="Chat"
          modelTag={modelTag}
          onModelChange={setModelTag}
          searchMode={searchMode}
          onSearchModeChange={setSearchMode}
          isConnected={isConnected}
        />

        <div className="flex-1 flex flex-col min-h-0">
          <ChatWindow messages={messages} />
          <InputBar
            onSend={handleSend}
            onStop={stopStreaming}
            isStreaming={status === "streaming"}
            disabled={!isConnected}
            placeholder={
              searchMode === "user_docs" && !collectionId
                ? "Upload documents first to use My Docs mode…"
                : undefined
            }
          />
        </div>
      </div>
    </div>
  );
}
