"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

interface DataPoint {
  time: string;
  latency: number;
  confidence: number;
}

interface LatencyChartProps {
  data: DataPoint[];
}

const CustomTooltip = ({ active, payload, label }: Record<string, unknown>) => {
  if (!active || !(payload as unknown[])?.length) return null;
  const p = payload as { name: string; value: number; color: string }[];
  return (
    <div className="glass-card px-3 py-2.5 text-xs">
      <p className="text-muted-2 mb-1.5">{String(label)}</p>
      {p.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: entry.color }} />
          <span className="text-muted">{entry.name}:</span>
          <span className="text-white/80 font-mono">{entry.value}</span>
        </div>
      ))}
    </div>
  );
};

export function LatencyChart({ data }: LatencyChartProps) {
  if (!data.length) {
    return (
      <div className="h-48 flex items-center justify-center text-sm text-muted">
        No data yet. Submit some queries to see metrics.
      </div>
    );
  }

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="latency"
            name="Latency (ms)"
            stroke="#6366f1"
            strokeWidth={2}
            fill="url(#latencyGrad)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
