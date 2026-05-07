"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3,
  ChevronLeft, ChevronRight,
  Zap, Plus, Clock, Pencil, Trash2, Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatSession } from "@/hooks/useChatSessions";

interface SidebarProps {
  sessions?: ChatSession[];
  activeSessionId?: string;
  onNewChat?: () => void;
  onSwitchChat?: (id: string) => void;
  onRenameSession?: (id: string, title: string) => void;
  onDeleteSession?: (id: string) => void;
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function Sidebar({ sessions = [], activeSessionId = "", onNewChat, onSwitchChat, onRenameSession, onDeleteSession }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const pathname = usePathname();
  const isOnChat = pathname === "/chat" || pathname.startsWith("/chat/");
  const effectiveActiveId = isOnChat ? activeSessionId : "";

  const startRename = (s: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(s.id);
    setRenameValue(s.title);
  };

  const commitRename = (id: string) => {
    if (renameValue.trim()) onRenameSession?.(id, renameValue.trim());
    setRenamingId(null);
  };

  const handleRenameKey = (e: React.KeyboardEvent, id: string) => {
    if (e.key === "Enter") commitRename(id);
    if (e.key === "Escape") setRenamingId(null);
  };

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className="hidden md:flex flex-col h-full flex-shrink-0 overflow-hidden border-r border-[var(--glass-border)]"
      style={{ background: "var(--sidebar-bg)" }}
    >
      {/* Logo + collapse toggle */}
      <div className="flex items-center gap-3 px-4 py-[18px] border-b border-[var(--glass-border)]">
        {/* Icon: click to expand when collapsed, click to create new chat when expanded */}
        <button
          onClick={() => collapsed ? setCollapsed(false) : onNewChat?.()}
          className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center flex-shrink-0 shadow-[var(--shadow-glow-sm)] hover:opacity-85 transition-opacity"
          title={collapsed ? "Expand sidebar" : "New chat"}
        >
          <Zap size={16} className="text-white" />
        </button>
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -6 }}
              transition={{ duration: 0.15 }}
              className="font-semibold text-sm whitespace-nowrap gradient-text flex-1"
            >
              AegisRAG
            </motion.span>
          )}
        </AnimatePresence>
        {!collapsed && (
          <button
            onClick={() => setCollapsed(true)}
            className="w-6 h-6 flex items-center justify-center rounded-lg text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)] transition-all duration-200 flex-shrink-0"
            title="Collapse sidebar"
          >
            <ChevronLeft size={14} />
          </button>
        )}
      </div>

      {/* New Chat */}
      {!collapsed && (
        <div className="px-3 py-3">
          <Link
            href="/chat"
            onClick={onNewChat}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl bg-accent/10 border border-accent/20 text-accent hover:bg-accent/20 transition-all duration-200 text-sm font-medium"
          >
            <Plus size={15} className="flex-shrink-0" />
            <span className="whitespace-nowrap">New Chat</span>
          </Link>
        </div>
      )}

      {/* Chat history */}
      {!collapsed && sessions.some((s) => s.messages.length > 0) && (
        <div className="px-3 pb-2 flex-1 overflow-hidden flex flex-col min-h-0">
          <div className="flex items-center gap-1.5 px-1 mb-1.5">
            <Clock size={11} className="text-[var(--muted)]" />
            <span className="text-[10px] font-medium text-[var(--muted)] uppercase tracking-wide">History</span>
          </div>
          <div className="space-y-0.5 overflow-y-auto no-scrollbar flex-1">
            {[...sessions]
              .filter((s) => s.messages.length > 0)
              .sort((a, b) => b.createdAt - a.createdAt)
              .map((s) => (
                <div
                  key={s.id}
                  className={cn(
                    "group relative rounded-lg text-xs transition-all duration-150",
                    s.id === effectiveActiveId
                      ? "bg-accent/12 border border-accent/20"
                      : "hover:bg-[var(--glass-bg)]"
                  )}
                >
                  {renamingId === s.id ? (
                    <div className="flex items-center gap-1 px-2 py-1.5">
                      <input
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => handleRenameKey(e, s.id)}
                        onBlur={() => commitRename(s.id)}
                        className="flex-1 bg-transparent text-xs text-[var(--fg)] outline-none border-b border-accent/40 pb-0.5 min-w-0"
                      />
                      <button
                        onMouseDown={(e) => { e.preventDefault(); commitRename(s.id); }}
                        className="text-success flex-shrink-0 p-0.5"
                      >
                        <Check size={11} />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => onSwitchChat?.(s.id)}
                      className={cn(
                        "w-full text-left px-2.5 py-2 rounded-lg",
                        s.id === effectiveActiveId ? "text-accent" : "text-[var(--muted-2)] hover:text-[var(--fg)]"
                      )}
                    >
                      <p className="truncate font-medium leading-snug pr-10">{s.title}</p>
                      <p className="text-[10px] text-[var(--muted)] mt-0.5">{formatRelativeTime(s.createdAt)}</p>
                    </button>
                  )}

                  {renamingId !== s.id && (
                    <div className="absolute right-1.5 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-0.5">
                      <button
                        onClick={(e) => startRename(s, e)}
                        className="p-1 rounded-md text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)] transition-colors"
                        title="Rename"
                      >
                        <Pencil size={11} />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); onDeleteSession?.(s.id); }}
                        className="p-1 rounded-md text-[var(--muted)] hover:text-danger hover:bg-danger/10 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Spacer when no history or collapsed */}
      {(collapsed || !sessions.some((s) => s.messages.length > 0)) && <div className="flex-1" />}

      {/* Bottom section: Analytics */}
      <div className="px-3 pb-3 border-t border-[var(--glass-border)] pt-3">
        {/* Analytics link */}
        <Link
          href="/analytics"
          className={cn(
            "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200",
            pathname.startsWith("/analytics")
              ? "bg-accent/12 text-accent border border-accent/20 font-medium"
              : "text-[var(--muted-2)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)]",
            collapsed && "justify-center px-2"
          )}
        >
          <BarChart3 size={17} className="flex-shrink-0" />
          <AnimatePresence>
            {!collapsed && (
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="whitespace-nowrap">
                Analytics
              </motion.span>
            )}
          </AnimatePresence>
        </Link>

      </div>
    </motion.aside>
  );
}
