"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Zap, MessageSquare } from "lucide-react";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

const SUGGESTIONS = [
  "What is the CMS reimbursement rate for telehealth?",
  "Explain FSA eligible expenses",
  "How do I appeal a VA benefits decision?",
  "IRS rules for home office deduction",
];

interface ChatWindowProps {
  messages: Message[];
  onSuggestion?: (q: string) => void;
}

export function ChatWindow({ messages, onSuggestion }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-5 p-6 sm:p-8 overflow-y-auto">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4, ease: [0.34, 1.56, 0.64, 1] }}
          className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center shadow-[var(--shadow-glow)]"
        >
          <Zap size={28} className="text-white" />
        </motion.div>

        <motion.div
          initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="text-center max-w-sm"
        >
          <h2 className="text-xl font-semibold text-[var(--fg)] mb-2">AegisRAG Copilot</h2>
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            Confidence-gated retrieval with NLI verification. Ask about CMS, FSA, IRS, or VA policies — or upload your own documents with the <span className="font-mono text-accent">+</span> button.
          </p>
        </motion.div>

        <motion.div
          initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.3 }}
          className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-sm sm:max-w-md"
        >
          {SUGGESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => onSuggestion?.(q)}
              className="glass-hover text-left px-3 py-2.5"
            >
              <div className="flex items-start gap-2">
                <MessageSquare size={12} className="text-accent-3 mt-0.5 flex-shrink-0" />
                <span className="text-xs text-[var(--muted-2)] leading-relaxed">{q}</span>
              </div>
            </button>
          ))}
        </motion.div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 sm:py-6 space-y-5 no-scrollbar">
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </AnimatePresence>
      <div ref={bottomRef} className="h-px" />
    </div>
  );
}
