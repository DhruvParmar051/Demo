"use client";

import { useState, useCallback } from "react";
import type { IngestResponse } from "@/lib/types";
import { ingestFiles } from "@/lib/api";

export type FileStatus = "pending" | "uploading" | "done" | "error";

export interface UploadFile {
  id: string;
  file: File;
  status: FileStatus;
  error?: string;
  result?: IngestResponse;
}

interface UseUploadReturn {
  files: UploadFile[];
  addFiles: (newFiles: File[]) => void;
  upload: (collectionId: string) => Promise<void>;
  removeFile: (id: string) => void;
  clearFiles: () => void;
  uploading: boolean;
}

export function useUpload(): UseUploadReturn {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);

  const addFiles = useCallback((newFiles: File[]) => {
    const entries: UploadFile[] = newFiles.map((f) => ({
      id: Math.random().toString(36).slice(2),
      file: f,
      status: "pending",
    }));
    setFiles((prev) => [...prev, ...entries]);
  }, []);

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const clearFiles = useCallback(() => setFiles([]), []);

  const upload = useCallback(
    async (collectionId: string) => {
      const pending = files.filter((f) => f.status === "pending");
      if (!pending.length) return;
      setUploading(true);

      setFiles((prev) =>
        prev.map((f) =>
          f.status === "pending" ? { ...f, status: "uploading" } : f
        )
      );

      try {
        const result = await ingestFiles(
          pending.map((f) => f.file),
          collectionId
        );
        setFiles((prev) =>
          prev.map((f) => {
            if (f.status !== "uploading") return f;
            const rejected = result.files_rejected.find(
              (r) => r.filename === f.file.name
            );
            if (rejected) return { ...f, status: "error", error: rejected.reason };
            return { ...f, status: "done", result };
          })
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setFiles((prev) =>
          prev.map((f) =>
            f.status === "uploading" ? { ...f, status: "error", error: msg } : f
          )
        );
      } finally {
        setUploading(false);
      }
    },
    [files]
  );

  return { files, addFiles, upload, removeFile, clearFiles, uploading };
}
