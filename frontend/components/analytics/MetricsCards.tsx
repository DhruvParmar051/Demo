"use client";

import { motion } from "framer-motion";
import { MessageSquare, Clock, Shield, TicketCheck } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsCardsProps {
  totalQueries: number;
  avgLatencyMs: number;
  avgConfidence: number;
  escalations: number;
  loading?: boolean;
}

function Card({
  icon: Icon,
  label,
  value,
  sub,
  color,
  loading,
  delay,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  sub?: string;
  color: string;
  loading?: boolean;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="glass-card p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-medium text-muted-2 uppercase tracking-wider">{label}</span>
        <div className={cn("w-8 h-8 rounded-xl flex items-center justify-center border", color)}>
          <Icon size={15} />
        </div>
      </div>
      {loading ? (
        <div className="shimmer-bg h-7 w-24 rounded-lg" />
      ) : (
        <p className="text-2xl font-bold text-white/90 tracking-tight">{value}</p>
      )}
      {sub && !loading && <p className="text-xs text-muted mt-1">{sub}</p>}
    </motion.div>
  );
}

export function MetricsCards({ totalQueries, avgLatencyMs, avgConfidence, escalations, loading }: MetricsCardsProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Card
        icon={MessageSquare}
        label="Total Queries"
        value={totalQueries.toLocaleString()}
        color="text-accent-3 bg-accent/10 border-accent/20"
        loading={loading}
        delay={0}
      />
      <Card
        icon={Clock}
        label="Avg Latency"
        value={avgLatencyMs < 1000 ? `${Math.round(avgLatencyMs)}ms` : `${(avgLatencyMs / 1000).toFixed(1)}s`}
        sub="End-to-end p50"
        color="text-blue-400 bg-blue-400/10 border-blue-400/20"
        loading={loading}
        delay={0.05}
      />
      <Card
        icon={Shield}
        label="Avg Confidence"
        value={`${Math.round(avgConfidence * 100)}%`}
        sub="CGAL head score"
        color={avgConfidence >= 0.85 ? "text-success bg-success/10 border-success/20"
          : avgConfidence >= 0.75 ? "text-warning bg-warning/10 border-warning/20"
          : "text-danger bg-danger/10 border-danger/20"}
        loading={loading}
        delay={0.1}
      />
      <Card
        icon={TicketCheck}
        label="Escalations"
        value={escalations.toLocaleString()}
        sub="CreateTicket calls"
        color="text-warning bg-warning/10 border-warning/20"
        loading={loading}
        delay={0.15}
      />
    </div>
  );
}
