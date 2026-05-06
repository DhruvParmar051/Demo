"use client";

import { Scissors } from "lucide-react";

interface DecompIndicatorProps {
  subQueries: string[];
}

export function DecompIndicator({ subQueries }: DecompIndicatorProps) {
  if (!subQueries?.length) return null;

  return (
    <div className="flex items-start gap-2 px-3 py-2.5 rounded-xl bg-accent/5 border border-accent/15 mb-3">
      <Scissors size={13} className="text-accent-3 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <p className="text-xs font-medium text-accent-3 mb-1">
          Decomposed into {subQueries.length} sub-queries
        </p>
        <ol className="space-y-0.5">
          {subQueries.map((q, i) => (
            <li key={i} className="text-xs text-muted-2">
              <span className="text-muted mr-1.5">{i + 1}.</span>{q}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
