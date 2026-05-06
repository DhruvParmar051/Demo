"use client";

import { Copy, Trash2, CheckCircle } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface CollectionBadgeProps {
  collectionId: string;
  onDelete: () => void;
}

export function CollectionBadge({ collectionId, onDelete }: CollectionBadgeProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(collectionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl glass-card">
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
        <span className="text-xs text-muted-2">Active collection:</span>
        <span className="text-xs text-white/90 font-mono truncate">{collectionId}</span>
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-lg hover:bg-white/[0.06] text-muted hover:text-white transition-colors"
        >
          {copied ? <CheckCircle size={13} className="text-success" /> : <Copy size={13} />}
        </button>
        <button
          onClick={onDelete}
          className="p-1.5 rounded-lg hover:bg-danger/10 text-muted hover:text-danger transition-colors"
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  );
}
