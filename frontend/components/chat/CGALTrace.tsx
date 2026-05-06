"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Wrench, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { ToolCall } from "@/lib/types";
import { cn, formatLatency, formatConfidence } from "@/lib/utils";

const TOOL_COLORS: Record<string, string> = {
  SearchKB:    "text-blue-400 bg-blue-400/10 border-blue-400/20",
  GetPolicy:   "text-purple-400 bg-purple-400/10 border-purple-400/20",
  CreateTicket:"text-red-400 bg-red-400/10 border-red-400/20",
  AnswerVerify:"text-green-400 bg-green-400/10 border-green-400/20",
  AnswerDirect:"text-accent-3 bg-accent/10 border-accent/20",
};

interface CGALTraceProps {
  toolCalls: ToolCall[];
  iterations: number;
  confidence?: number;
  alpha?: number | null;
}

export function CGALTrace({ toolCalls, iterations, confidence, alpha }: CGALTraceProps) {
  const [open, setOpen] = useState(false);
  if (!toolCalls?.length) return null;

  return (
    <div className="mt-3 rounded-xl border border-[var(--glass-border)] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-[var(--glass-bg)] hover:bg-[var(--glass-hover)] transition-colors text-xs text-[var(--muted-2)]"
      >
        <div className="flex items-center gap-2 flex-wrap">
          <Wrench size={12} className="text-accent-3 flex-shrink-0" />
          <span className="font-medium">
            CGAL Loop — {iterations} iter, {toolCalls.length} tool call{toolCalls.length !== 1 ? "s" : ""}
          </span>
          {confidence !== undefined && (
            <span className={cn(
              "px-1.5 py-0.5 rounded-md font-mono border",
              confidence >= 0.85 ? "text-success bg-success/10 border-success/20"
                : confidence >= 0.75 ? "text-warning bg-warning/10 border-warning/20"
                : "text-danger bg-danger/10 border-danger/20"
            )}>
              {formatConfidence(confidence)}
            </span>
          )}
        </div>
        <ChevronDown size={13} className={cn("transition-transform duration-200 flex-shrink-0", open && "rotate-180")} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-3.5 space-y-2.5 border-t border-[var(--glass-border)]">
              {alpha !== null && alpha !== undefined && (
                <div className="flex items-center gap-2 text-xs text-[var(--muted)] px-1">
                  <span className="flex-shrink-0">Adaptive α:</span>
                  <div className="flex-1 h-1.5 bg-[var(--glass-bg)] border border-[var(--glass-border)] rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-accent to-accent-2 rounded-full" style={{ width: `${alpha * 100}%` }} />
                  </div>
                  <span className="font-mono text-[var(--muted-2)] flex-shrink-0">{alpha.toFixed(2)}</span>
                </div>
              )}
              {toolCalls.map((tc, i) => {
                const colorClass = TOOL_COLORS[tc.tool_name] ?? "text-[var(--muted-2)] bg-[var(--glass-bg)] border-[var(--glass-border)]";
                const confDelta = tc.confidence_after != null && tc.confidence_before != null
                  ? tc.confidence_after - tc.confidence_before : null;
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex flex-col items-center gap-1 mt-0.5">
                      <div className="w-5 h-5 rounded-full bg-[var(--glass-bg)] border border-[var(--glass-border)] flex items-center justify-center text-[10px] text-[var(--muted-2)] font-mono flex-shrink-0">
                        {tc.iteration}
                      </div>
                      {i < toolCalls.length - 1 && <div className="w-px h-4 bg-[var(--glass-border)]" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={cn("px-2 py-0.5 rounded-md text-xs font-medium border", colorClass)}>
                          {tc.tool_name}
                        </span>
                        <span className="text-xs text-[var(--muted)] font-mono">{formatLatency(tc.latency_ms)}</span>
                        {confDelta !== null && (
                          <span className={cn("flex items-center gap-0.5 text-xs",
                            confDelta > 0 ? "text-success" : confDelta < 0 ? "text-danger" : "text-[var(--muted)]"
                          )}>
                            {confDelta > 0 ? <TrendingUp size={10} /> : confDelta < 0 ? <TrendingDown size={10} /> : <Minus size={10} />}
                            {confDelta > 0 ? "+" : ""}{(confDelta * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
