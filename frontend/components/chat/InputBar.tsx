"use client";

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Square, Plus, X, Upload, FileText, CheckCircle, Loader2, AlertCircle } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";
import { ingestFiles } from "@/lib/api";

const ACCEPTED = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
};

type FileUploadState = { file: File; status: "pending" | "uploading" | "done" | "error"; error?: string };

interface InputBarProps {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  collectionId: string;
  onCollectionChange: (id: string) => void;
}

function formatBytes(b: number) {
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / (1024 * 1024)).toFixed(1)}MB`;
}

export function InputBar({
  onSend, onStop, isStreaming, collectionId, onCollectionChange,
}: InputBarProps) {
  const [value, setValue] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [files, setFiles] = useState<FileUploadState[]>([]);
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const q = value.trim();
    if (!q || isStreaming) return;
    onSend(q);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  };

  // Dropzone for the inline upload popover
  const onDrop = useCallback((accepted: File[]) => {
    setFiles((prev) => [
      ...prev,
      ...accepted.map((f) => ({ file: f, status: "pending" as const })),
    ]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: 25 * 1024 * 1024,
    disabled: uploading,
    multiple: true,
  });

  const handleUpload = async () => {
    const pending = files.filter((f) => f.status === "pending");
    if (!pending.length) return;
    setUploading(true);
    setFiles((prev) => prev.map((f) => f.status === "pending" ? { ...f, status: "uploading" } : f));

    // Ensure we have a collection ID
    let cid = collectionId;
    if (!cid) {
      cid = "user_" + Math.random().toString(36).slice(2, 12);
      onCollectionChange(cid);
      if (typeof window !== "undefined") localStorage.setItem("aegis_collection_id", cid);
    }

    try {
      const result = await ingestFiles(pending.map((f) => f.file), cid);
      setFiles((prev) =>
        prev.map((f) => {
          if (f.status !== "uploading") return f;
          const rej = result.files_rejected.find((r) => r.filename === f.file.name);
          return rej ? { ...f, status: "error", error: rej.reason } : { ...f, status: "done" };
        })
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setFiles((prev) => prev.map((f) => f.status === "uploading" ? { ...f, status: "error", error: msg } : f));
    } finally {
      setUploading(false);
    }
  };

  const clearDone = () => setFiles((prev) => prev.filter((f) => f.status !== "done"));
  const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i));
  const canSend = value.trim().length > 0 && !isStreaming;
  const pendingCount = files.filter((f) => f.status === "pending").length;
  const doneCount = files.filter((f) => f.status === "done").length;

  return (
    <div
      className="px-3 sm:px-4 md:px-6 py-3 md:py-4 border-t border-[var(--glass-border)] flex-shrink-0"
      style={{ background: "var(--topbar-bg)" }}
    >
      <div className="max-w-4xl mx-auto relative">
        {/* Inline upload popover */}
        <AnimatePresence>
          {uploadOpen && (
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.97 }}
              transition={{ duration: 0.18 }}
              className="absolute bottom-full mb-3 left-0 right-0 glass-card p-4 z-20"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Upload size={14} className="text-accent-3" />
                  <span className="text-sm font-semibold text-[var(--fg)]">Quick Upload</span>
                  {collectionId && (
                    <span className="text-xs font-mono text-[var(--muted)] bg-[var(--glass-bg)] border border-[var(--glass-border)] px-1.5 py-0.5 rounded-md">
                      {collectionId.slice(0, 16)}…
                    </span>
                  )}
                </div>
                <button onClick={() => setUploadOpen(false)} className="text-[var(--muted)] hover:text-[var(--fg)] transition-colors">
                  <X size={15} />
                </button>
              </div>

              {/* Drop zone */}
              <div
                {...getRootProps()}
                className={cn(
                  "rounded-xl border-2 border-dashed p-5 text-center cursor-pointer transition-all duration-200",
                  isDragActive ? "border-accent/60 bg-accent/5" : "border-[var(--glass-border)] hover:border-accent/30 hover:bg-accent/5"
                )}
              >
                <input {...getInputProps()} />
                <Upload size={18} className="mx-auto mb-1.5 text-[var(--muted)]" />
                <p className="text-xs text-[var(--muted-2)]">
                  {isDragActive ? "Drop files here" : "Drop files or click · PDF, DOCX, TXT, MD · 25 MB max"}
                </p>
              </div>

              {/* File list */}
              {files.length > 0 && (
                <div className="mt-3 space-y-1.5 max-h-36 overflow-y-auto no-scrollbar">
                  {files.map((f, i) => (
                    <div key={i} className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-[var(--glass-bg)] border border-[var(--glass-border)]">
                      <FileText size={13} className="text-[var(--muted-2)] flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-[var(--fg)] truncate font-medium">{f.file.name}</p>
                        <p className="text-[10px] text-[var(--muted)]">{formatBytes(f.file.size)}</p>
                      </div>
                      {f.status === "pending" && (
                        <button onClick={() => removeFile(i)} className="text-[var(--muted)] hover:text-danger transition-colors flex-shrink-0">
                          <X size={12} />
                        </button>
                      )}
                      {f.status === "uploading" && <Loader2 size={13} className="text-accent-3 animate-spin flex-shrink-0" />}
                      {f.status === "done" && <CheckCircle size={13} className="text-success flex-shrink-0" />}
                      {f.status === "error" && (
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <AlertCircle size={13} className="text-danger" />
                          <span className="text-[10px] text-danger">{f.error}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-2 mt-3">
                {pendingCount > 0 && (
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-semibold",
                      "bg-gradient-to-r from-accent to-accent-2 text-white",
                      "hover:opacity-90 transition-opacity disabled:opacity-50"
                    )}
                  >
                    {uploading ? <><Loader2 size={12} className="animate-spin" />Indexing…</> : <><Upload size={12} />Upload {pendingCount} file{pendingCount !== 1 ? "s" : ""}</>}
                  </button>
                )}
                {doneCount > 0 && (
                  <button onClick={clearDone} className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs text-[var(--muted)] hover:text-[var(--fg)] bg-[var(--glass-bg)] border border-[var(--glass-border)] transition-colors">
                    <CheckCircle size={12} className="text-success" /> {doneCount} indexed
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input row */}
        <div
          className={cn(
            "flex items-end gap-2 px-3 py-2.5 rounded-2xl border transition-all duration-200",
            "bg-[var(--input-bg)] border-[var(--input-border)]",
            "focus-within:border-accent/40 focus-within:shadow-[0_0_0_3px_rgba(99,102,241,0.08)]"
          )}
        >
          {/* + upload button */}
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={() => setUploadOpen(!uploadOpen)}
            className={cn(
              "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-200",
              uploadOpen
                ? "bg-accent/20 border border-accent/30 text-accent"
                : "bg-[var(--glass-bg)] border border-[var(--glass-border)] text-[var(--muted)] hover:text-accent hover:bg-accent/10 hover:border-accent/20"
            )}
            aria-label="Upload documents"
          >
            <Plus size={15} className={cn("transition-transform duration-200", uploadOpen && "rotate-45")} />
          </motion.button>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={collectionId ? "Ask about your uploaded document… (Enter to send)" : "Ask a question… (Enter to send)"}
            rows={1}
            disabled={false}
            className={cn(
              "flex-1 bg-transparent text-sm text-[var(--fg)] placeholder:text-[var(--muted)]",
              "resize-none outline-none min-h-[24px] max-h-[180px] leading-relaxed",
              "disabled:opacity-40 disabled:cursor-not-allowed"
            )}
          />

          {/* Stop / Send */}
          {isStreaming ? (
            <motion.button
              initial={{ scale: 0.8 }} animate={{ scale: 1 }}
              onClick={onStop}
              className="w-8 h-8 rounded-xl bg-danger/12 border border-danger/25 flex items-center justify-center text-danger hover:bg-danger/20 transition-colors flex-shrink-0"
            >
              <Square size={13} />
            </motion.button>
          ) : (
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={handleSend}
              disabled={!canSend}
              className={cn(
                "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-200",
                canSend
                  ? "bg-gradient-to-br from-accent to-accent-2 text-white shadow-[var(--shadow-glow-sm)] hover:shadow-[var(--shadow-glow)]"
                  : "bg-[var(--glass-bg)] border border-[var(--glass-border)] text-[var(--muted)] cursor-not-allowed"
              )}
            >
              <Send size={13} />
            </motion.button>
          )}
        </div>

        <p className="text-center text-[10px] sm:text-xs text-[var(--muted)] mt-2">
          AegisRAG may produce errors. Verify important information with official sources.
        </p>
      </div>
    </div>
  );
}
