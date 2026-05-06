"use client";

import { motion } from "framer-motion";
import { MessageSquare, Clock, Shield, TicketCheck } from "lucide-react";
import { cn } from "@/lib/utils";

function Card({ icon: Icon, label, value, sub, colorClass, loading, delay }: {
  icon: React.ElementType; label: string; value: string; sub?: string;
  colorClass: string; loading?: boolean; delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="glass-card p-4 sm:p-5"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] sm:text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">{label}</span>
        <div className={cn("w-7 h-7 sm:w-8 sm:h-8 rounded-xl flex items-center justify-center border", colorClass)}>
          <Icon size={14} />
        </div>
      </div>
      {loading
        ? <div className="shimmer-bg h-7 w-20 rounded-lg" />
        : <p className="text-xl sm:text-2xl font-bold text-[var(--fg)] tracking-tight">{value}</p>
      }
      {sub && !loading && <p className="text-xs text-[var(--muted)] mt-0.5">{sub}</p>}
    </motion.div>
  );
}

interface MetricsCardsProps {
  totalQueries: number;
  avgLatencyMs: number;
  avgConfidence: number;
  escalations: number;
  loading?: boolean;
}

export function MetricsCards({ totalQueries, avgLatencyMs, avgConfidence, escalations, loading }: MetricsCardsProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
      <Card icon={MessageSquare} label="Total Queries" value={totalQueries.toLocaleString()}
        colorClass="text-accent-3 bg-accent/10 border-accent/20" loading={loading} delay={0} />
      <Card icon={Clock} label="Avg Latency"
        value={avgLatencyMs < 1000 ? `${Math.round(avgLatencyMs)}ms` : `${(avgLatencyMs / 1000).toFixed(1)}s`}
        sub="End-to-end p50" colorClass="text-blue-400 bg-blue-400/10 border-blue-400/20" loading={loading} delay={0.05} />
      <Card icon={Shield} label="Avg Confidence" value={`${Math.round(avgConfidence * 100)}%`}
        sub="CGAL head score"
        colorClass={avgConfidence >= 0.85 ? "text-success bg-success/10 border-success/20"
          : avgConfidence >= 0.75 ? "text-warning bg-warning/10 border-warning/20"
          : "text-danger bg-danger/10 border-danger/20"}
        loading={loading} delay={0.1} />
      <Card icon={TicketCheck} label="Escalations" value={escalations.toLocaleString()} sub="CreateTicket calls"
        colorClass="text-warning bg-warning/10 border-warning/20" loading={loading} delay={0.15} />
    </div>
  );
}
