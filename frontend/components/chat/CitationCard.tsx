"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, ExternalLink, X, CheckCircle } from "lucide-react";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: CitationCardProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={cn(
          "inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-xs font-mono font-medium",
          "bg-accent/10 border border-accent/20 text-accent-3 hover:bg-accent/20",
          "transition-all duration-150 cursor-pointer align-middle mx-0.5",
          citation.verified && "border-success/30 bg-success/5 text-success"
        )}
      >
        {citation.verified && <CheckCircle size={9} />}
        [{index + 1}]
      </button>

      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
              onClick={() => setOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.15 }}
              className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg"
            >
              <div className="glass-card p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-accent/15 border border-accent/20 flex items-center justify-center">
                      <FileText size={13} className="text-accent-3" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white/90">Source [{index + 1}]</p>
                      <p className="text-xs text-muted truncate max-w-[280px]">{citation.source}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setOpen(false)}
                    className="text-muted hover:text-white transition-colors p-1"
                  >
                    <X size={14} />
                  </button>
                </div>

                <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3 mb-3">
                  <p className="text-sm text-white/80 leading-relaxed line-clamp-6">{citation.cited_text}</p>
                </div>

                <div className="flex items-center justify-between text-xs text-muted">
                  <div className="flex items-center gap-3">
                    <span>Doc: <span className="text-muted-2 font-mono">{citation.doc_id.slice(0, 16)}…</span></span>
                    {citation.page_number && <span>Page {citation.page_number}</span>}
                    <span>Span: {citation.span_start}–{citation.span_end}</span>
                  </div>
                  {citation.source_url && (
                    <a
                      href={citation.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-accent hover:text-accent-3 transition-colors"
                    >
                      <ExternalLink size={11} />
                      Open source
                    </a>
                  )}
                </div>

                {citation.verified !== null && citation.verified !== undefined && (
                  <div className={cn(
                    "mt-3 flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg",
                    citation.verified
                      ? "bg-success/10 text-success border border-success/20"
                      : "bg-danger/10 text-danger border border-danger/20"
                  )}>
                    <CheckCircle size={11} />
                    {citation.verified ? "NLI verified — grounded claim" : "NLI failed — ungrounded claim"}
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
