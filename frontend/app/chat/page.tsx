"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { InputBar } from "@/components/chat/InputBar";
import { useChatSessions } from "@/hooks/useChatSessions";
import { getHealth } from "@/lib/api";

export default function ChatPage() {
  const {
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
  } = useChatSessions();

  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const check = () =>
      getHealth()
        .then(() => setIsConnected(true))
        .catch(() => setIsConnected(false));
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex h-full w-full" style={{ background: "var(--page-bg)" }}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={createNewChat}
        onSwitchChat={switchChat}
      />

      <div className="flex flex-col flex-1 min-w-0 min-h-0 overflow-hidden">
        <TopBar title="Chat" isConnected={isConnected} />

        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          <ChatWindow messages={messages} onSuggestion={sendMessage} />
          <InputBar
            onSend={sendMessage}
            onStop={stopStreaming}
            isStreaming={status === "streaming"}
            collectionId={collectionId}
            onCollectionChange={setCollectionId}
          />
        </div>
      </div>
    </div>
  );
}
