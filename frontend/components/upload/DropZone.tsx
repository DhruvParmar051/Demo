"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { Upload, FileText, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPTED = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
};

interface DropZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

export function DropZone({ onFiles, disabled }: DropZoneProps) {
  const onDrop = useCallback((accepted: File[]) => { if (accepted.length) onFiles(accepted); }, [onFiles]);
  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: 25 * 1024 * 1024, disabled, multiple: true,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative rounded-2xl border-2 border-dashed p-8 sm:p-10 text-center cursor-pointer transition-all duration-200",
        isDragActive && !isDragReject && "border-accent/60 bg-accent/5",
        isDragReject && "border-danger/50 bg-danger/5",
        !isDragActive && !isDragReject && "border-[var(--glass-border)] hover:border-accent/30 hover:bg-accent/[0.03]",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input {...getInputProps()} />
      <motion.div
        animate={{ scale: isDragActive ? 1.04 : 1 }}
        transition={{ duration: 0.15 }}
        className="flex flex-col items-center gap-3"
      >
        <div className={cn(
          "w-12 h-12 rounded-2xl flex items-center justify-center border transition-colors",
          isDragReject  ? "bg-danger/12 border-danger/30 text-danger"
          : isDragActive ? "bg-accent/15 border-accent/30 text-accent-3"
          : "bg-[var(--glass-bg)] border-[var(--glass-border)] text-[var(--muted-2)]"
        )}>
          {isDragReject ? <AlertCircle size={20} /> : <Upload size={20} />}
        </div>
        <div>
          <p className="text-sm font-medium text-[var(--fg)] mb-0.5">
            {isDragActive ? "Drop files here" : "Drop files or click to browse"}
          </p>
          <p className="text-xs text-[var(--muted)]">PDF, DOCX, TXT, MD · Max 25 MB per file</p>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap justify-center">
          {[".pdf", ".docx", ".txt", ".md"].map((ext) => (
            <span key={ext} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--glass-bg)] border border-[var(--glass-border)] text-xs text-[var(--muted-2)]">
              <FileText size={10} />{ext}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
