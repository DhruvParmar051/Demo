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

function generateCollectionId(): string {
  return "user_" + Math.random().toString(36).slice(2, 12);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export default function UploadPage() {
  const { files, addFiles, upload, removeFile, clearFiles, uploading } = useUpload();
  const [collectionId, setCollectionId] = useState<string>("");
  const [deleteStatus, setDeleteStatus] = useState<"idle" | "loading" | "done">("idle");

  useEffect(() => {
    const stored = localStorage.getItem("aegis_collection_id");
    if (stored) setCollectionId(stored);
    else {
      const newId = generateCollectionId();
      setCollectionId(newId);
      localStorage.setItem("aegis_collection_id", newId);
    }
  }, []);

  const handleUpload = async () => {
    await upload(collectionId);
  };

  const handleDeleteSession = async () => {
    setDeleteStatus("loading");
    try {
      await deleteSession(collectionId);
    } catch {
      // ignore if collection doesn't exist yet
    }
    const newId = generateCollectionId();
    setCollectionId(newId);
    localStorage.setItem("aegis_collection_id", newId);
    clearFiles();
    setDeleteStatus("done");
    setTimeout(() => setDeleteStatus("idle"), 2000);
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;
  const doneCount = files.filter((f) => f.status === "done").length;
  const errorCount = files.filter((f) => f.status === "error").length;

  return (
    <div className="flex h-full" style={{ background: "#0a0a0f" }}>
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Header */}
        <div className="px-8 py-8 border-b border-white/[0.06]">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-9 h-9 rounded-xl bg-accent/15 border border-accent/20 flex items-center justify-center">
                <Upload size={17} className="text-accent-3" />
              </div>
              <h1 className="text-xl font-semibold text-white/90">Document Upload</h1>
            </div>
            <p className="text-sm text-muted leading-relaxed">
              Upload your documents to create a personal knowledge base. Switch to "My Docs" in the chat to query them.
            </p>
          </div>
        </div>

        <div className="flex-1 px-8 py-6 max-w-2xl space-y-6">
          {/* Collection badge */}
          {collectionId && (
            <CollectionBadge collectionId={collectionId} onDelete={handleDeleteSession} />
          )}

          {/* Drop zone */}
          <DropZone onFiles={addFiles} disabled={uploading} />

          {/* File list */}
          <AnimatePresence>
            {files.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-2"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-white/80">{files.length} file{files.length !== 1 ? "s" : ""} selected</span>
                  <button onClick={clearFiles} className="text-xs text-muted hover:text-white transition-colors">
                    Clear all
                  </button>
                </div>

                {files.map((f) => (
                  <motion.div
                    key={f.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 8 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-xl glass-card"
                  >
                    <div className="w-8 h-8 rounded-lg bg-white/[0.06] border border-white/[0.08] flex items-center justify-center flex-shrink-0">
                      <FileText size={14} className="text-muted-2" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white/80 truncate font-medium">{f.file.name}</p>
                      <p className="text-xs text-muted">{formatBytes(f.file.size)}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {f.status === "pending" && (
                        <button
                          onClick={() => removeFile(f.id)}
                          className="p-1.5 rounded-lg hover:bg-white/[0.06] text-muted hover:text-white transition-colors"
                        >
                          <Trash2 size={13} />
                        </button>
                      )}
                      {f.status === "uploading" && (
                        <Loader2 size={14} className="text-accent-3 animate-spin" />
                      )}
                      {f.status === "done" && (
                        <CheckCircle size={14} className="text-success" />
                      )}
                      {f.status === "error" && (
                        <div className="flex items-center gap-1.5">
                          <XCircle size={14} className="text-danger" />
                          <span className="text-xs text-danger">{f.error}</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Upload button */}
          {pendingCount > 0 && (
            <motion.button
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              onClick={handleUpload}
              disabled={uploading}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-3 rounded-xl font-medium text-sm",
                "bg-gradient-to-r from-accent to-accent-2 text-white",
                "hover:opacity-90 active:scale-[0.99] transition-all duration-200",
                "shadow-glow disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {uploading ? (
                <>
                  <Loader2 size={15} className="animate-spin" />
                  Uploading & indexing…
                </>
              ) : (
                <>
                  <Upload size={15} />
                  Upload {pendingCount} file{pendingCount !== 1 ? "s" : ""}
                </>
              )}
            </motion.button>
          )}

          {/* Success state */}
          {doneCount > 0 && !uploading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center justify-between px-4 py-3 rounded-xl bg-success/10 border border-success/20"
            >
              <div className="flex items-center gap-2">
                <CheckCircle size={15} className="text-success" />
                <span className="text-sm text-success font-medium">
                  {doneCount} file{doneCount !== 1 ? "s" : ""} indexed successfully
                </span>
              </div>
              <Link
                href="/chat"
                className="flex items-center gap-1 text-xs text-success hover:text-white transition-colors"
              >
                Go to chat <ArrowRight size={12} />
              </Link>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
