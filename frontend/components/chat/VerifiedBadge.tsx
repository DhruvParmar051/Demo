"use client";

import { CheckCircle, AlertTriangle, XCircle, SkipForward } from "lucide-react";
import { cn } from "@/lib/utils";

interface VerifiedBadgeProps {
  verdict: "pass" | "partial" | "fail" | "skipped" | null | undefined;
}

export function VerifiedBadge({ verdict }: VerifiedBadgeProps) {
  if (!verdict || verdict === "skipped") return null;

  const config = {
    pass: { icon: CheckCircle, label: "Verified", className: "text-success bg-success/10 border-success/20" },
    partial: { icon: AlertTriangle, label: "Partial", className: "text-warning bg-warning/10 border-warning/20" },
    fail: { icon: XCircle, label: "Unverified", className: "text-danger bg-danger/10 border-danger/20" },
  }[verdict];

  if (!config) return null;
  const { icon: Icon, label, className } = config;

  return (
    <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border", className)}>
      <Icon size={11} />
      {label}
    </span>
  );
}
