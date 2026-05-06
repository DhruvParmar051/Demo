"use client";

import { useState } from "react";
import { Copy, Trash2, CheckCircle } from "lucide-react";

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
        <div className="w-2 h-2 rounded-full bg-success animate-pulse flex-shrink-0" />
        <span className="text-xs text-[var(--muted-2)] flex-shrink-0">Collection:</span>
        <span className="text-xs text-[var(--fg)] font-mono truncate">{collectionId}</span>
      </div>
      <div className="flex items-center gap-1 flex-shrink-0">
        <button onClick={handleCopy} className="p-1.5 rounded-lg hover:bg-[var(--glass-bg)] text-[var(--muted)] hover:text-[var(--fg)] transition-colors">
          {copied ? <CheckCircle size={13} className="text-success" /> : <Copy size={13} />}
        </button>
        <button onClick={onDelete} className="p-1.5 rounded-lg hover:bg-danger/10 text-[var(--muted)] hover:text-danger transition-colors">
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  );
}
