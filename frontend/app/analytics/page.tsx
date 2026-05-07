"use client";

import { useEffect, useState } from "react";
import { RefreshCw, Cpu, Server } from "lucide-react";
import { Sidebar } from "@/components/layout/Sidebar";
import { useChatSessionsContext } from "@/components/providers/ChatSessionsProvider";
import { MetricsCards } from "@/components/analytics/MetricsCards";
import { LatencyChart } from "@/components/analytics/LatencyChart";
import { TicketsTable } from "@/components/analytics/TicketsTable";
import { useMetrics } from "@/hooks/useMetrics";
import { cn } from "@/lib/utils";

interface DataPoint { time: string; latency: number; confidence: number }

export default function AnalyticsPage() {
  const { sessions, activeSessionId, createNewChat, switchChat, renameSession, deleteSession } = useChatSessionsContext();
  const { health, tickets, parsedMetrics, loading, error, refetch } = useMetrics(30000);
  const [chartData, setChartData] = useState<DataPoint[]>([]);

  useEffect(() => {
    if (!loading && parsedMetrics.aegisrag_avg_latency_ms !== undefined) {
      const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      setChartData((prev) => [
        ...prev.slice(-29),
        {
          time: now,
          latency: Math.round(parsedMetrics.aegisrag_avg_latency_ms ?? 0),
          confidence: Math.round((parsedMetrics.aegisrag_avg_confidence ?? 0) * 100),
        },
      ]);
    }
  }, [parsedMetrics, loading]);

  const totalQueries = parsedMetrics.aegisrag_queries_total ?? 0;
  const avgLatency   = parsedMetrics.aegisrag_avg_latency_ms ?? 0;
  const avgConf      = parsedMetrics.aegisrag_avg_confidence ?? 0;
  const escalations  = parsedMetrics.aegisrag_escalations_total ?? 0;

  return (
    <div className="flex h-full w-full" style={{ background: "var(--page-bg)" }}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={() => { createNewChat(); window.location.href = "/chat"; }}
        onSwitchChat={(id) => { switchChat(id); window.location.href = "/chat"; }}
        onRenameSession={renameSession}
        onDeleteSession={deleteSession}
      />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <div className="px-4 sm:px-8 py-5 sm:py-7 border-b border-[var(--glass-border)] pt-14 md:pt-6">
          <div className="flex items-center justify-between max-w-5xl flex-wrap gap-3">
            <div>
              <h1 className="text-xl font-semibold text-[var(--fg)] mb-0.5">Analytics</h1>
              <p className="text-sm text-[var(--muted)]">Live system metrics · auto-refreshes every 30 s</p>
            </div>
            <div className="flex items-center gap-2">
              {health && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl glass-card text-xs">
                  <Server size={12} className="text-accent-3" />
                  <span className="text-[var(--muted-2)] hidden sm:inline">{health.device}</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-success" />
                </div>
              )}
              <button
                onClick={refetch}
                disabled={loading}
                className={cn("flex items-center gap-1.5 px-3 py-2 rounded-xl glass-hover text-xs text-[var(--muted-2)]", loading && "opacity-50 cursor-not-allowed")}
              >
                <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 px-4 sm:px-8 py-5 sm:py-6 max-w-5xl w-full space-y-6 sm:space-y-8">
          {error && (
            <div className="px-4 py-3 rounded-xl bg-danger/8 border border-danger/20 text-danger text-sm">
              {error} — Is the backend running on <span className="font-mono">localhost:8000</span>?
            </div>
          )}

          <section>
            <h2 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-3">Key Metrics</h2>
            <MetricsCards totalQueries={totalQueries} avgLatencyMs={avgLatency} avgConfidence={avgConf} escalations={escalations} loading={loading} />
          </section>

          {health && (
            <section>
              <h2 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-3">Backend Status</h2>
              <div className="glass-card p-5">
                <div className="flex items-center gap-6 flex-wrap">
                  <div>
                    <p className="text-xs text-[var(--muted)] mb-1">Device</p>
                    <div className="flex items-center gap-2">
                      <Cpu size={13} className="text-accent-3" />
                      <span className="text-sm text-[var(--fg)] font-mono">{health.device}</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-[var(--muted)] mb-1">Cached Models</p>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {health.models_cached.length > 0
                        ? health.models_cached.map((m) => (
                            <span key={m} className="px-2 py-0.5 rounded-md text-xs font-mono text-accent-3 bg-accent/10 border border-accent/20">{m}</span>
                          ))
                        : <span className="text-xs text-[var(--muted)]">None loaded yet</span>
                      }
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          <section>
            <h2 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-3">Latency Over Time</h2>
            <div className="glass-card p-4 sm:p-5">
              <LatencyChart data={chartData} />
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Escalation Tickets</h2>
              <span className="text-xs text-[var(--muted)] font-mono">{tickets.length} total</span>
            </div>
            <div className="glass-card p-4 sm:p-5 overflow-x-auto">
              <TicketsTable tickets={tickets} loading={loading} />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
