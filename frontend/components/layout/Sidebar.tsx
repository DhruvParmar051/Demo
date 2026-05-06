"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare, Upload, BarChart3,
  ChevronLeft, ChevronRight,
  Zap, Plus, Sparkles, X, Menu,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/ui/ThemeToggle";

const navItems = [
  { href: "/chat",      icon: MessageSquare, label: "Chat" },
  { href: "/upload",    icon: Upload,        label: "Documents" },
  { href: "/analytics", icon: BarChart3,     label: "Analytics" },
];

interface SidebarProps {
  onNewChat?: () => void;
}

export function Sidebar({ onNewChat }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();

  // Close mobile sidebar on route change
  useEffect(() => { setMobileOpen(false); }, [pathname]);

  // Close on outside tap
  useEffect(() => {
    if (!mobileOpen) return;
    const handler = (e: TouchEvent | MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-sidebar]")) setMobileOpen(false);
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
    };
  }, [mobileOpen]);

  const SidebarContent = ({ mobile = false }: { mobile?: boolean }) => (
    <div
      data-sidebar
      className={cn(
        "flex flex-col h-full border-r transition-colors duration-200",
        "border-[var(--glass-border)]",
        mobile ? "w-64" : collapsed ? "w-16" : "w-[220px]"
      )}
      style={{ background: "var(--sidebar-bg)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-[18px] border-b border-[var(--glass-border)]">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center flex-shrink-0 shadow-[var(--shadow-glow-sm)]">
          <Zap size={16} className="text-white" />
        </div>
        <AnimatePresence>
          {(!collapsed || mobile) && (
            <motion.span
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -6 }}
              transition={{ duration: 0.15 }}
              className="font-semibold text-sm whitespace-nowrap gradient-text"
            >
              AegisRAG
            </motion.span>
          )}
        </AnimatePresence>
        {mobile && (
          <button onClick={() => setMobileOpen(false)} className="ml-auto text-[var(--muted)] hover:text-[var(--fg)] transition-colors">
            <X size={18} />
          </button>
        )}
      </div>

      {/* New Chat */}
      <div className="px-3 py-3">
        <button
          onClick={() => { onNewChat?.(); setMobileOpen(false); }}
          className={cn(
            "w-full flex items-center gap-2.5 px-3 py-2 rounded-xl",
            "bg-accent/10 border border-accent/20 text-accent hover:bg-accent/20",
            "transition-all duration-200 text-sm font-medium",
            !mobile && collapsed && "justify-center px-2"
          )}
        >
          <Plus size={15} className="flex-shrink-0" />
          <AnimatePresence>
            {(!collapsed || mobile) && (
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="whitespace-nowrap">
                New Chat
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>

      {/* Nav */}
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
                  ? "bg-accent/12 text-accent border border-accent/20 font-medium"
                  : "text-[var(--muted-2)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)]",
                !mobile && collapsed && "justify-center px-2"
              )}
            >
              <Icon size={17} className="flex-shrink-0" />
              <AnimatePresence>
                {(!collapsed || mobile) && (
                  <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="whitespace-nowrap">
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-3 border-t border-[var(--glass-border)] space-y-1">
        <AnimatePresence>
          {(!collapsed || mobile) && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex items-center gap-2 px-3 py-2 rounded-xl bg-[var(--glass-bg)] border border-[var(--glass-border)] mb-1"
            >
              <Sparkles size={12} className="text-accent-3 flex-shrink-0" />
              <span className="text-xs text-[var(--muted-2)] whitespace-nowrap">DS 615 Project</span>
            </motion.div>
          )}
        </AnimatePresence>

        <ThemeToggle collapsed={!mobile && collapsed} />

        {!mobile && (
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center p-2 rounded-xl text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)] transition-all duration-200"
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: collapsed ? 64 : 220 }}
        transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
        className="hidden md:flex flex-col h-full flex-shrink-0 overflow-hidden"
      >
        <SidebarContent />
      </motion.aside>

      {/* Mobile hamburger button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="md:hidden fixed top-3 left-3 z-50 w-9 h-9 rounded-xl glass-card flex items-center justify-center text-[var(--muted)] hover:text-[var(--fg)] transition-colors"
        aria-label="Open menu"
      >
        <Menu size={18} />
      </button>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm md:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.div
              initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }}
              transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
              className="fixed left-0 top-0 bottom-0 z-50 md:hidden"
            >
              <SidebarContent mobile />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
