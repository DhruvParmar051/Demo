"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import { Zap, MessageSquare } from "lucide-react";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages: Message[];
}

export function ChatWindow({ messages }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-6 p-8">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center shadow-glow">
          <Zap size={28} className="text-white" />
        </div>
        <div className="text-center max-w-sm">
          <h2 className="text-xl font-semibold text-white/90 mb-2">AegisRAG Copilot</h2>
          <p className="text-sm text-muted leading-relaxed">
            Confidence-gated retrieval with NLI verification. Ask about CMS, FSA, IRS, or VA policies.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 w-full max-w-sm mt-2">
          {[
            "What is the CMS reimbursement rate for telehealth?",
            "Explain FSA eligible expenses",
            "How do I appeal a VA benefits decision?",
            "IRS rules for home office deduction",
          ].map((q) => (
            <div key={q} className="glass-hover rounded-xl px-3 py-2.5 cursor-pointer">
              <div className="flex items-start gap-2">
                <MessageSquare size={12} className="text-accent-3 mt-0.5 flex-shrink-0" />
                <span className="text-xs text-muted-2 leading-relaxed">{q}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6 no-scrollbar">
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  );
}
