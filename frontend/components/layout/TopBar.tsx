"use client";

import { motion } from "framer-motion";
import { Activity, Cpu, WifiOff } from "lucide-react";
import type { ModelTag, SearchMode } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODEL_OPTIONS: { value: ModelTag; label: string; description: string }[] = [
  { value: "b1", label: "B1", description: "No retrieval" },
  { value: "b2", label: "B2", description: "BM25 sparse" },
  { value: "b3", label: "B3", description: "Dense only" },
  { value: "m1", label: "M1", description: "SFT only" },
  { value: "m2", label: "M2", description: "SFT + DPO" },
  { value: "m3", label: "M3", description: "+ Confidence" },
  { value: "m4", label: "M4", description: "+ Alpha" },
  { value: "m5", label: "M5 ✦", description: "Full AegisRAG" },
];

interface TopBarProps {
  title: string;
  modelTag: ModelTag;
  onModelChange: (m: ModelTag) => void;
  searchMode: SearchMode;
  onSearchModeChange: (m: SearchMode) => void;
  isConnected?: boolean;
}

export function TopBar({ title, modelTag, onModelChange, searchMode, onSearchModeChange, isConnected }: TopBarProps) {
  return (
    <header
      className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-[var(--glass-border)] backdrop-blur-xl z-10 flex-shrink-0"
      style={{ background: "var(--topbar-bg)" }}
    >
      {/* Title + status */}
      <div className="flex items-center gap-2 md:gap-3 pl-10 md:pl-0">
        <h1 className="text-sm font-semibold text-[var(--fg)]">{title}</h1>
        <div className="flex items-center gap-1.5">
          {isConnected ? (
            <>
              <motion.div
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ repeat: Infinity, duration: 2.5 }}
                className="w-1.5 h-1.5 rounded-full bg-success"
              />
              <span className="hidden sm:inline text-xs text-[var(--muted)]">Online</span>
            </>
          ) : (
            <>
              <WifiOff size={11} className="text-[var(--muted)]" />
              <span className="hidden sm:inline text-xs text-[var(--muted)]">Backend offline</span>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 md:gap-3">
        {/* Search mode — hidden on very small screens */}
        <div className="hidden sm:flex items-center gap-0.5 p-1 rounded-xl bg-[var(--glass-bg)] border border-[var(--glass-border)]">
          {(["system_kb", "user_docs"] as SearchMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => onSearchModeChange(mode)}
              className={cn(
                "px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 whitespace-nowrap",
                searchMode === mode
                  ? "bg-accent/15 text-accent border border-accent/20"
                  : "text-[var(--muted)] hover:text-[var(--fg)]"
              )}
            >
              {mode === "system_kb" ? "System KB" : "My Docs"}
            </button>
          ))}
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-1.5">
          <Cpu size={13} className="text-[var(--muted)] hidden sm:block" />
          <select
            value={modelTag}
            onChange={(e) => onModelChange(e.target.value as ModelTag)}
            className={cn(
              "text-xs rounded-xl px-2.5 py-1.5 outline-none cursor-pointer transition-colors font-medium",
              "bg-[var(--glass-bg)] border border-[var(--glass-border)] text-[var(--fg)]",
              "focus:border-accent/40"
            )}
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value} style={{ background: "var(--surface)" }}>
                {o.label} — {o.description}
              </option>
            ))}
          </select>
        </div>

        {/* Latency badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-[var(--glass-bg)] border border-[var(--glass-border)]">
          <Activity size={12} className="text-accent-3" />
          <span className="text-xs text-[var(--muted-2)] font-mono">~2.0s p50</span>
        </div>
      </div>
    </header>
  );
}
