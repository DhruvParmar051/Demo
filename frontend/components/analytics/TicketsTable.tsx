"use client";

import { motion } from "framer-motion";
import type { TicketRecord } from "@/lib/types";
import { cn } from "@/lib/utils";
import { TicketCheck } from "lucide-react";

const SEVERITY_STYLE: Record<string, string> = {
  low: "text-success bg-success/10 border-success/20",
  medium: "text-warning bg-warning/10 border-warning/20",
  high: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  critical: "text-danger bg-danger/10 border-danger/20",
};

const CATEGORY_STYLE: Record<string, string> = {
  billing: "text-blue-400 bg-blue-400/10 border-blue-400/20",
  technical: "text-purple-400 bg-purple-400/10 border-purple-400/20",
  account: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  policy: "text-accent-3 bg-accent/10 border-accent/20",
  other: "text-muted-2 bg-white/[0.04] border-white/[0.08]",
};

interface TicketsTableProps {
  tickets: TicketRecord[];
  loading?: boolean;
}

export function TicketsTable({ tickets, loading }: TicketsTableProps) {
  if (loading) {
    return (
      <div className="space-y-2">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="shimmer-bg h-12 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!tickets.length) {
    return (
      <div className="flex flex-col items-center gap-3 py-12 text-muted">
        <TicketCheck size={32} className="opacity-30" />
        <p className="text-sm">No escalation tickets yet</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="text-left border-b border-white/[0.06]">
            {["Ticket ID", "Query", "Category", "Severity", "ETA"].map((h) => (
              <th key={h} className="pb-3 pr-4 text-xs font-medium text-muted uppercase tracking-wider">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tickets.map((t, i) => (
            <motion.tr
              key={t.ticket_id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors"
            >
              <td className="py-3 pr-4">
                <span className="text-xs font-mono text-muted-2">{t.ticket_id.slice(0, 12)}…</span>
              </td>
              <td className="py-3 pr-4 max-w-[260px]">
                <p className="text-xs text-white/70 truncate">{t.query}</p>
              </td>
              <td className="py-3 pr-4">
                <span className={cn("px-2 py-0.5 rounded-md text-xs font-medium border capitalize", CATEGORY_STYLE[t.category] ?? CATEGORY_STYLE.other)}>
                  {t.category}
                </span>
              </td>
              <td className="py-3 pr-4">
                <span className={cn("px-2 py-0.5 rounded-md text-xs font-medium border capitalize", SEVERITY_STYLE[t.severity] ?? SEVERITY_STYLE.low)}>
                  {t.severity}
                </span>
              </td>
              <td className="py-3">
                <span className="text-xs text-muted font-mono">{t.estimated_response_time}</span>
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
