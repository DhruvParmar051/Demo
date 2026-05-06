"use client";

import { motion } from "framer-motion";
import { Activity, Cpu } from "lucide-react";
import type { ModelTag, SearchMode } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODEL_OPTIONS: { value: ModelTag; label: string; description: string }[] = [
  { value: "b1", label: "B1", description: "Baseline 1 (no retrieval)" },
  { value: "b2", label: "B2", description: "Baseline 2 (BM25)" },
  { value: "b3", label: "B3", description: "Baseline 3 (dense)" },
  { value: "m1", label: "M1", description: "SFT only" },
  { value: "m2", label: "M2", description: "SFT + DPO" },
  { value: "m3", label: "M3", description: "+ Confidence head" },
  { value: "m4", label: "M4", description: "+ Adaptive alpha" },
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

export function TopBar({
  title,
  modelTag,
  onModelChange,
  searchMode,
  onSearchModeChange,
  isConnected,
}: TopBarProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-white/[0.06] glass-card rounded-none"
      style={{ background: "rgba(10,10,18,0.8)", backdropFilter: "blur(20px)" }}>
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-white/90">{title}</h1>
        <div className="flex items-center gap-1.5">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              isConnected ? "bg-success" : "bg-muted"
            )}
          />
          <span className="text-xs text-muted">{isConnected ? "Backend Online" : "Connecting..."}</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Search mode toggle */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.08]">
          {(["system_kb", "user_docs"] as SearchMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => onSearchModeChange(mode)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200",
                searchMode === mode
                  ? "bg-accent/20 text-accent border border-accent/20"
                  : "text-muted hover:text-white"
              )}
            >
              {mode === "system_kb" ? "System KB" : "My Docs"}
            </button>
          ))}
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-muted" />
          <select
            value={modelTag}
            onChange={(e) => onModelChange(e.target.value as ModelTag)}
            className="text-xs bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-1.5 text-white/90 outline-none focus:border-accent/40 transition-colors cursor-pointer"
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value} style={{ background: "#111118" }}>
                {o.label} — {o.description}
              </option>
            ))}
          </select>
        </div>

        {/* Live status */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/[0.06]">
          <Activity size={13} className="text-accent-3" />
          <span className="text-xs text-muted-2 font-mono">~2.0s p50</span>
        </div>
      </div>
    </header>
  );
}
