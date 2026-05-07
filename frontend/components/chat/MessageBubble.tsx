"use client";

import { motion } from "framer-motion";
import { User, Zap, AlertCircle, TicketCheck } from "lucide-react";
import type { Message } from "@/lib/types";
import { cn, formatConfidence } from "@/lib/utils";
import { StreamingMessage } from "./StreamingMessage";
import { VerifiedBadge } from "./VerifiedBadge";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      className={cn(
        "flex gap-2.5 w-full",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5",
        isUser
          ? "bg-[var(--glass-bg)] border border-[var(--glass-border)]"
          : "bg-gradient-to-br from-accent to-accent-2 border border-accent/30 shadow-[var(--shadow-glow-sm)]"
      )}>
        {isUser
          ? <User size={13} className="text-[var(--muted-2)]" />
          : <Zap size={13} className="text-white" />
        }
      </div>

      {/* Content */}
      <div className={cn(
        "flex flex-col min-w-0",
        isUser ? "items-end max-w-[65%] sm:max-w-[60%]" : "items-start max-w-[80%] sm:max-w-[75%]"
      )}>
        {/* Label row */}
        <div className={cn("flex items-center gap-1.5 mb-1", isUser && "flex-row-reverse")}>
          <span className="text-xs font-medium text-[var(--muted-2)]">
            {isUser ? "You" : "AegisRAG"}
          </span>
          {!isUser && <VerifiedBadge verdict={message.verifyVerdict} />}
          {!isUser && message.ticketId && (
            <span className="flex items-center gap-1 text-xs text-warning border border-warning/20 bg-warning/10 px-1.5 py-0.5 rounded-md">
              <TicketCheck size={10} /> Escalated
            </span>
          )}
        </div>

        {/* Bubble */}
        <div className={cn(
          "rounded-2xl px-3.5 py-2.5 w-full",
          isUser
            ? "bg-[var(--bubble-user)] border border-accent/15 text-[var(--fg)] text-sm"
            : "glass-card text-[var(--fg)] text-sm"
        )}>
          {isUser ? (
            <p className="leading-relaxed whitespace-pre-wrap break-words text-sm">{message.content}</p>
          ) : (
            <>
              {message.error ? (
                <div className="flex items-start gap-2 text-danger">
                  <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
                  <span className="text-sm break-words">{message.error}</span>
                </div>
              ) : (
                <StreamingMessage
                  content={message.content}
                  citations={message.citations ?? []}
                  isStreaming={message.isStreaming ?? false}
                />
              )}
            </>
          )}
        </div>

        {/* Meta — confidence only */}
        {!isUser && !message.isStreaming && message.confidence !== undefined && !isNaN(message.confidence) && (
          <div className="flex items-center gap-2 mt-1 px-1">
            <span className={cn(
              "text-xs font-mono",
              message.confidence >= 0.85 ? "text-success"
                : message.confidence >= 0.75 ? "text-warning"
                : "text-[var(--muted)]"
            )}>
              {formatConfidence(message.confidence)} conf.
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
