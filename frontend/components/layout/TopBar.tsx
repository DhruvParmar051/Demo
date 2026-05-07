"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { WifiOff, Sun, Moon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface TopBarProps {
  title: string;
  isConnected?: boolean;
}

export function TopBar({ title, isConnected }: TopBarProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const isDark = theme === "dark";

  return (
    <header
      className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-[var(--glass-border)] backdrop-blur-xl z-10 flex-shrink-0"
      style={{ background: "var(--topbar-bg)" }}
    >
      {/* Title + connection status */}
      <div className="flex items-center gap-2 md:gap-3">
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
              <span className="hidden sm:inline text-xs text-[var(--muted)]">Offline</span>
            </>
          )}
        </div>
      </div>

      {/* Theme toggle — icon only */}
      {mounted && (
        <button
          onClick={() => setTheme(isDark ? "light" : "dark")}
          aria-label="Toggle theme"
          className="w-8 h-8 flex items-center justify-center rounded-xl text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--glass-bg)] border border-[var(--glass-border)] transition-all duration-200"
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={theme}
              initial={{ rotate: -90, opacity: 0, scale: 0.7 }}
              animate={{ rotate: 0, opacity: 1, scale: 1 }}
              exit={{ rotate: 90, opacity: 0, scale: 0.7 }}
              transition={{ duration: 0.18 }}
            >
              {isDark
                ? <Moon size={14} />
                : <Sun size={14} className="text-amber-500" />
              }
            </motion.div>
          </AnimatePresence>
        </button>
      )}
    </header>
  );
}
