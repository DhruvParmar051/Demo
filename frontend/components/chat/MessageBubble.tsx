"use client";

import { motion } from "framer-motion";
import { User, Zap, AlertCircle, TicketCheck } from "lucide-react";
import type { Message } from "@/lib/types";
import { cn, formatLatency, formatConfidence } from "@/lib/utils";
import { StreamingMessage } from "./StreamingMessage";
import { VerifiedBadge } from "./VerifiedBadge";
import { CGALTrace } from "./CGALTrace";
import { DecompIndicator } from "./DecompIndicator";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className={cn(
        "flex gap-3 w-full",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5",
        isUser
          ? "bg-[var(--glass-bg)] border border-[var(--glass-border)]"
          : "bg-gradient-to-br from-accent to-accent-2 border border-accent/30 shadow-[var(--shadow-glow-sm)]"
      )}>
        {isUser
          ? <User size={14} className="text-[var(--muted-2)]" />
          : <Zap size={14} className="text-white" />
        }
      </div>

      {/* Content */}
      <div className={cn(
        "flex flex-col min-w-0",
        isUser ? "items-end max-w-[80%] sm:max-w-[75%]" : "items-start max-w-[92%] sm:max-w-[85%]"
      )}>
        {/* Label row */}
        <div className={cn("flex items-center gap-2 mb-1.5", isUser && "flex-row-reverse")}>
          <span className="text-xs font-medium text-[var(--muted-2)]">
            {isUser ? "You" : "AegisRAG"}
          </span>
          {!isUser && message.modelTag && (
            <span className="text-xs text-[var(--muted)] font-mono px-1.5 py-0.5 rounded-md bg-[var(--glass-bg)] border border-[var(--glass-border)]">
              {message.modelTag}
            </span>
          )}
          {!isUser && <VerifiedBadge verdict={message.verifyVerdict} />}
          {!isUser && message.ticketId && (
            <span className="flex items-center gap-1 text-xs text-warning border border-warning/20 bg-warning/10 px-1.5 py-0.5 rounded-md">
              <TicketCheck size={10} /> Escalated
            </span>
          )}
        </div>

        {/* Bubble */}
        <div className={cn(
          "rounded-2xl px-4 py-3 w-full",
          isUser
            ? "bg-[var(--bubble-user)] border border-accent/15 text-[var(--fg)] text-sm"
            : "glass-card text-[var(--fg)] text-sm"
        )}>
          {isUser ? (
            <p className="leading-relaxed whitespace-pre-wrap break-words">{message.content}</p>
          ) : (
            <>
              {message.decomposed && message.subQueries?.length
                ? <DecompIndicator subQueries={message.subQueries} />
                : null}
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
              {message.toolCalls && message.toolCalls.length > 0 && (
                <CGALTrace
                  toolCalls={message.toolCalls}
                  iterations={message.cgalIterations ?? 1}
                  confidence={message.confidence}
                  alpha={message.alpha}
                />
              )}
            </>
          )}
        </div>

        {/* Meta */}
        {!isUser && !message.isStreaming && (message.latencyMs || message.confidence !== undefined) && (
          <div className="flex items-center gap-3 mt-1.5 px-1 flex-wrap">
            {message.latencyMs && (
              <span className="text-xs text-[var(--muted)] font-mono">{formatLatency(message.latencyMs)}</span>
            )}
            {message.ttftMs && (
              <span className="text-xs text-[var(--muted)]">TTFT {formatLatency(message.ttftMs)}</span>
            )}
            {message.confidence !== undefined && (
              <span className={cn(
                "text-xs font-mono",
                message.confidence >= 0.85 ? "text-success"
                  : message.confidence >= 0.75 ? "text-warning"
                  : "text-danger"
              )}>
                {formatConfidence(message.confidence)} conf.
              </span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
