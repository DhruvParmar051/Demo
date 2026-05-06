"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, CheckCircle, XCircle, Loader2, FileText, Trash2, ArrowRight } from "lucide-react";
import Link from "next/link";
import { Sidebar } from "@/components/layout/Sidebar";
import { DropZone } from "@/components/upload/DropZone";
import { CollectionBadge } from "@/components/upload/CollectionBadge";
import { useUpload } from "@/hooks/useUpload";
import { deleteSession } from "@/lib/api";
import { cn } from "@/lib/utils";

function generateCollectionId() {
  return "user_" + Math.random().toString(36).slice(2, 12);
}
function formatBytes(b: number) {
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / (1024 * 1024)).toFixed(1)}MB`;
}

export default function UploadPage() {
  const { files, addFiles, upload, removeFile, clearFiles, uploading } = useUpload();
  const [collectionId, setCollectionId] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("aegis_collection_id");
    if (stored) setCollectionId(stored);
    else {
      const id = generateCollectionId();
      setCollectionId(id);
      localStorage.setItem("aegis_collection_id", id);
    }
  }, []);

  const handleDeleteSession = async () => {
    try { await deleteSession(collectionId); } catch { /* ignore */ }
    const id = generateCollectionId();
    setCollectionId(id);
    localStorage.setItem("aegis_collection_id", id);
    clearFiles();
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;
  const doneCount = files.filter((f) => f.status === "done").length;

  return (
    <div className="flex h-full w-full" style={{ background: "var(--page-bg)" }}>
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <div className="px-4 sm:px-8 py-6 sm:py-8 border-b border-[var(--glass-border)] pt-14 md:pt-8">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-9 h-9 rounded-xl bg-accent/12 border border-accent/20 flex items-center justify-center">
                <Upload size={17} className="text-accent-3" />
              </div>
              <h1 className="text-xl font-semibold text-[var(--fg)]">Document Upload</h1>
            </div>
            <p className="text-sm text-[var(--muted)] leading-relaxed">
              Upload documents to build a personal knowledge base. Use the <span className="font-mono text-accent">+</span> button in chat, or upload here in bulk.
            </p>
          </div>
        </div>

        <div className="flex-1 px-4 sm:px-8 py-6 max-w-2xl space-y-5">
          {collectionId && (
            <CollectionBadge collectionId={collectionId} onDelete={handleDeleteSession} />
          )}

          <DropZone onFiles={addFiles} disabled={uploading} />

          <AnimatePresence>
            {files.length > 0 && (
              <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-[var(--fg)]">{files.length} file{files.length !== 1 ? "s" : ""} selected</span>
                  <button onClick={clearFiles} className="text-xs text-[var(--muted)] hover:text-[var(--fg)] transition-colors">Clear all</button>
                </div>

                {files.map((f) => (
                  <motion.div
                    key={f.id}
                    initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 6 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-xl glass-card"
                  >
                    <div className="w-8 h-8 rounded-lg bg-[var(--glass-bg)] border border-[var(--glass-border)] flex items-center justify-center flex-shrink-0">
                      <FileText size={14} className="text-[var(--muted-2)]" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-[var(--fg)] truncate font-medium">{f.file.name}</p>
                      <p className="text-xs text-[var(--muted)]">{formatBytes(f.file.size)}</p>
                    </div>
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      {f.status === "pending" && (
                        <button onClick={() => removeFile(f.id)} className="p-1.5 rounded-lg hover:bg-danger/10 text-[var(--muted)] hover:text-danger transition-colors">
                          <Trash2 size={13} />
                        </button>
                      )}
                      {f.status === "uploading" && <Loader2 size={14} className="text-accent-3 animate-spin" />}
                      {f.status === "done" && <CheckCircle size={14} className="text-success" />}
                      {f.status === "error" && (
                        <div className="flex items-center gap-1">
                          <XCircle size={14} className="text-danger" />
                          <span className="text-xs text-danger truncate max-w-[80px]">{f.error}</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {pendingCount > 0 && (
            <motion.button
              initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
              onClick={() => upload(collectionId)}
              disabled={uploading}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm",
                "bg-gradient-to-r from-accent to-accent-2 text-white",
                "hover:opacity-90 active:scale-[0.99] transition-all duration-200 shadow-[var(--shadow-glow)]",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {uploading
                ? <><Loader2 size={15} className="animate-spin" />Uploading &amp; indexing…</>
                : <><Upload size={15} />Upload {pendingCount} file{pendingCount !== 1 ? "s" : ""}</>
              }
            </motion.button>
          )}

          {doneCount > 0 && !uploading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex items-center justify-between px-4 py-3 rounded-xl bg-success/8 border border-success/20"
            >
              <div className="flex items-center gap-2">
                <CheckCircle size={15} className="text-success" />
                <span className="text-sm text-success font-medium">{doneCount} file{doneCount !== 1 ? "s" : ""} indexed</span>
              </div>
              <Link href="/chat" className="flex items-center gap-1 text-xs text-success hover:text-[var(--fg)] transition-colors">
                Go to chat <ArrowRight size={12} />
              </Link>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
