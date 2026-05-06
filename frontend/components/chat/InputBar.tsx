"use client";

import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Send, Square, Paperclip } from "lucide-react";
import { cn } from "@/lib/utils";

interface InputBarProps {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function InputBar({ onSend, onStop, isStreaming, disabled, placeholder }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const q = value.trim();
    if (!q || isStreaming) return;
    onSend(q);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const canSend = value.trim().length > 0 && !isStreaming && !disabled;

  return (
    <div className="px-4 py-4 border-t border-white/[0.06]" style={{ background: "rgba(10,10,18,0.95)" }}>
      <div className="max-w-4xl mx-auto">
        <div className={cn(
          "glass-card flex items-end gap-3 px-4 py-3",
          "focus-within:border-accent/30 transition-all duration-200"
        )}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={placeholder ?? "Ask about CMS, FSA, IRS, VA policies… (Enter to send, Shift+Enter for newline)"}
            rows={1}
            disabled={disabled}
            className={cn(
              "flex-1 bg-transparent text-sm text-white/90 placeholder:text-muted resize-none outline-none",
              "min-h-[24px] max-h-[200px] leading-relaxed font-sans",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          />

          <div className="flex items-center gap-2 flex-shrink-0">
            {isStreaming ? (
              <motion.button
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                onClick={onStop}
                className="w-8 h-8 rounded-lg bg-danger/15 border border-danger/20 flex items-center justify-center text-danger hover:bg-danger/25 transition-colors"
              >
                <Square size={13} />
              </motion.button>
            ) : (
              <motion.button
                whileTap={{ scale: 0.92 }}
                onClick={handleSend}
                disabled={!canSend}
                className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200",
                  canSend
                    ? "bg-gradient-to-br from-accent to-accent-2 text-white shadow-glow-sm hover:shadow-glow"
                    : "bg-white/[0.06] text-muted cursor-not-allowed"
                )}
              >
                <Send size={13} />
              </motion.button>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-muted mt-2">
          AegisRAG may produce errors. Verify important information.
        </p>
      </div>
    </div>
  );
}
