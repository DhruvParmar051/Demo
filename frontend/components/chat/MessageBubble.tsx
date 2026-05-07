"use client";

import { motion } from "framer-motion";
import { User, Zap, AlertCircle, FileText, TicketCheck } from "lucide-react";
import type { Message } from "@/lib/types";
import { cn, formatConfidence } from "@/lib/utils";
import { StreamingMessage } from "./StreamingMessage";
import { VerifiedBadge } from "./VerifiedBadge";

function formatBytes(b: number) {
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function fileIcon(type: string) {
  if (type.includes("pdf")) return "PDF";
  if (type.includes("word") || type.includes("docx")) return "DOC";
  if (type.includes("text") || type.includes("plain")) return "TXT";
  if (type.includes("markdown")) return "MD";
  return "FILE";
}

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const hasFiles = isUser && message.attachedFiles && message.attachedFiles.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      className={cn("flex gap-2.5 w-full", isUser ? "flex-row-reverse" : "flex-row")}
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
        {/* Label */}
        <div className={cn("flex items-center gap-1.5 mb-1", isUser && "flex-row-reverse")}>
          <span className="text-xs font-medium text-[var(--muted-2)]">
            {isUser ? "You" : "AegisRAG"}
          </span>
          {!isUser && <VerifiedBadge verdict={message.verifyVerdict} />}
        </div>

        {/* Attached files — shown above the text bubble for user messages */}
        {hasFiles && (
          <div className="flex flex-wrap gap-2 mb-2 justify-end">
            {message.attachedFiles!.map((f, i) => (
              <div key={i} className="flex items-center gap-2 px-2.5 py-2 rounded-xl bg-[var(--glass-bg)] border border-[var(--glass-border)] max-w-[200px]">
                <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center flex-shrink-0">
                  <FileText size={14} className="text-accent-3" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-[var(--fg)] truncate leading-tight">{f.name}</p>
                  <p className="text-[10px] text-[var(--muted)] mt-0.5">
                    {fileIcon(f.type)} · {formatBytes(f.size)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

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
                  citations={message.ticketId ? [] : (message.citations ?? [])}
                  isStreaming={message.isStreaming ?? false}
                />
              )}

              {/* Ticket card */}
              {!message.isStreaming && message.ticketId && (
                <div className="mt-3 flex items-start gap-2.5 px-3 py-2.5 rounded-xl bg-warning/8 border border-warning/25">
                  <div className="w-7 h-7 rounded-lg bg-warning/15 border border-warning/25 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <TicketCheck size={13} className="text-warning" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-semibold text-warning mb-0.5">Support Ticket Created</p>
                    <p className="text-[11px] text-[var(--muted-2)] font-mono break-all">{message.ticketId}</p>
                    <p className="text-[11px] text-[var(--muted)] mt-1">A specialist will review and follow up with you.</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Confidence */}
        {!isUser && !message.isStreaming && message.confidence !== undefined && !isNaN(message.confidence) && message.confidence > 0 && (
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
