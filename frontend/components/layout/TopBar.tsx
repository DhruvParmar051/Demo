"use client";

import { WifiOff } from "lucide-react";
import { motion } from "framer-motion";

interface TopBarProps {
  title: string;
  isConnected?: boolean;
}

export function TopBar({ title, isConnected }: TopBarProps) {
  return (
    <header
      className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-[var(--glass-border)] backdrop-blur-xl z-10 flex-shrink-0"
      style={{ background: "var(--topbar-bg)" }}
    >
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
              <span className="hidden sm:inline text-xs text-[var(--muted)]">Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
