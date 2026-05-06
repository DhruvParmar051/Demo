"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Upload,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Zap,
  Plus,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/chat", icon: MessageSquare, label: "Chat" },
  { href: "/upload", icon: Upload, label: "Documents" },
  { href: "/analytics", icon: BarChart3, label: "Analytics" },
];

interface SidebarProps {
  onNewChat?: () => void;
}

export function Sidebar({ onNewChat }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className="relative flex flex-col h-full glass border-r border-white/[0.06] overflow-hidden flex-shrink-0"
      style={{ background: "rgba(10,10,18,0.95)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-white/[0.06]">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center flex-shrink-0 shadow-glow-sm">
          <Zap size={16} className="text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden"
            >
              <span className="font-semibold text-sm text-white whitespace-nowrap gradient-text">
                AegisRAG
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* New Chat Button */}
      <div className="px-3 py-3">
        <button
          onClick={onNewChat}
          className={cn(
            "w-full flex items-center gap-2.5 px-3 py-2 rounded-xl",
            "bg-accent/10 border border-accent/20 text-accent hover:bg-accent/20",
            "transition-all duration-200 text-sm font-medium",
            collapsed && "justify-center px-2"
          )}
        >
          <Plus size={15} className="flex-shrink-0" />
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="whitespace-nowrap"
              >
                New Chat
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>

      {/* Nav Items */}
      <nav className="flex-1 px-3 py-2 space-y-1">
        {navItems.map(({ href, icon: Icon, label }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200",
                active
                  ? "bg-accent/15 text-white border border-accent/20"
                  : "text-muted-2 hover:text-white hover:bg-white/[0.05]",
                collapsed && "justify-center px-2"
              )}
            >
              <Icon size={17} className="flex-shrink-0" />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="whitespace-nowrap font-medium"
                  >
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-3 border-t border-white/[0.06]">
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] mb-2"
            >
              <Sparkles size={13} className="text-accent-3" />
              <span className="text-xs text-muted-2 whitespace-nowrap">DS 615 Project</span>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center p-2 rounded-xl text-muted hover:text-white hover:bg-white/[0.05] transition-all duration-200"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </motion.aside>
  );
}
