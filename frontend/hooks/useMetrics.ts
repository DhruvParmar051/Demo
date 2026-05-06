"use client";

import { useState, useEffect, useCallback } from "react";
import type { HealthResponse, TicketRecord } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface MetricsData {
  health: HealthResponse | null;
  tickets: TicketRecord[];
  rawMetrics: string | null;
  parsedMetrics: Record<string, number>;
  loading: boolean;
  error: string | null;
}

function parsePrometheus(raw: string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const line of raw.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const match = line.match(/^(\w+)(?:\{[^}]*\})?\s+([\d.]+)/);
    if (match) {
      const existing = result[match[1]];
      result[match[1]] = existing !== undefined
        ? existing + parseFloat(match[2])
        : parseFloat(match[2]);
    }
  }
  return result;
}

export function useMetrics(pollIntervalMs = 30000): MetricsData & { refetch: () => void } {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [tickets, setTickets] = useState<TicketRecord[]>([]);
  const [rawMetrics, setRawMetrics] = useState<string | null>(null);
  const [parsedMetrics, setParsedMetrics] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthRes, ticketsRes, metricsRes] = await Promise.allSettled([
        fetch(`${API_URL}/health`).then((r) => r.json() as Promise<HealthResponse>),
        fetch(`${API_URL}/tickets`).then((r) => r.json() as Promise<{ tickets: TicketRecord[] }>),
        fetch(`${API_URL}/metrics`).then((r) => r.text()),
      ]);

      if (healthRes.status === "fulfilled") setHealth(healthRes.value);
      if (ticketsRes.status === "fulfilled") setTickets(ticketsRes.value.tickets ?? []);
      if (metricsRes.status === "fulfilled") {
        setRawMetrics(metricsRes.value);
        setParsedMetrics(parsePrometheus(metricsRes.value));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch metrics");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, pollIntervalMs);
    return () => clearInterval(id);
  }, [fetchAll, pollIntervalMs]);

  return { health, tickets, rawMetrics, parsedMetrics, loading, error, refetch: fetchAll };
}
