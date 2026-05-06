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
      className={cn("flex gap-3 max-w-4xl", isUser ? "ml-auto flex-row-reverse" : "")}
    >
      {/* Avatar */}
      <div className={cn(
        "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5",
        isUser
          ? "bg-white/[0.08] border border-white/[0.1]"
          : "bg-gradient-to-br from-accent to-accent-2 border border-accent/30 shadow-glow-sm"
      )}>
        {isUser ? <User size={14} className="text-white/70" /> : <Zap size={14} className="text-white" />}
      </div>

      {/* Bubble */}
      <div className={cn(
        "flex-1 min-w-0 max-w-[85%]",
        isUser ? "flex flex-col items-end" : ""
      )}>
        {/* Role label */}
        <div className={cn(
          "flex items-center gap-2 mb-1.5",
          isUser ? "flex-row-reverse" : ""
        )}>
          <span className="text-xs font-medium text-muted-2">
            {isUser ? "You" : "AegisRAG"}
          </span>
          {!isUser && message.modelTag && (
            <span className="text-xs text-muted font-mono px-1.5 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06]">
              {message.modelTag}
            </span>
          )}
          {!isUser && <VerifiedBadge verdict={message.verifyVerdict} />}
          {!isUser && message.ticketId && (
            <span className="flex items-center gap-1 text-xs text-warning border border-warning/20 bg-warning/10 px-1.5 py-0.5 rounded-md">
              <TicketCheck size={10} />
              Escalated
            </span>
          )}
        </div>

        <div className={cn(
          "rounded-2xl px-4 py-3",
          isUser
            ? "bg-accent/15 border border-accent/20 text-white/90 text-sm"
            : "glass-card text-white/85 text-sm"
        )}>
          {isUser ? (
            <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            <>
              {message.decomposed && message.subQueries?.length ? (
                <DecompIndicator subQueries={message.subQueries} />
              ) : null}

              {message.error ? (
                <div className="flex items-start gap-2 text-danger">
                  <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
                  <span className="text-sm">{message.error}</span>
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

        {/* Meta row */}
        {!isUser && !message.isStreaming && (message.latencyMs || message.confidence !== undefined) && (
          <div className="flex items-center gap-3 mt-1.5 px-1">
            {message.latencyMs && (
              <span className="text-xs text-muted font-mono">{formatLatency(message.latencyMs)}</span>
            )}
            {message.ttftMs && (
              <span className="text-xs text-muted">TTFT {formatLatency(message.ttftMs)}</span>
            )}
            {message.confidence !== undefined && (
              <span className={cn(
                "text-xs font-mono",
                message.confidence >= 0.85 ? "text-success"
                : message.confidence >= 0.75 ? "text-warning"
                : "text-danger"
              )}>
                {formatConfidence(message.confidence)} confidence
              </span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
