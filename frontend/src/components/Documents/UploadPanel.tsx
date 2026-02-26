"use client";

import { useState, useRef, useCallback, DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { uploadDocument } from "@/lib/api";

interface UploadPanelProps {
  onUploadComplete?: () => void;
}

const ACCEPTED_TYPES = ".txt,.md,.csv,.pdf,.docx";

export default function UploadPanel({ onUploadComplete }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadStatus(null);

      try {
        await uploadDocument(file);
        setUploadStatus({
          type: "success",
          message: `"${file.name}" uploaded successfully`,
        });
        onUploadComplete?.();
      } catch {
        setUploadStatus({
          type: "error",
          message: `Failed to upload "${file.name}"`,
        });
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadComplete]
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
      // Reset input so same file can be re-selected
      e.target.value = "";
    },
    [handleUpload]
  );

  return (
    <div className="p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">
        Upload Documents
      </h3>
      <motion.div
        className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          isDragging
            ? "border-indigo-500 bg-indigo-500/10"
            : "border-gray-700 hover:border-gray-600 bg-gray-900/50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        animate={
          isDragging
            ? { scale: 1.02, borderColor: "#6366f1" }
            : { scale: 1, borderColor: "#374151" }
        }
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        role="button"
        tabIndex={0}
        aria-label="Upload a document by clicking or dragging a file"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          onChange={handleFileSelect}
          className="hidden"
          aria-hidden="true"
        />

        {isUploading ? (
          <div className="flex flex-col items-center gap-2">
            <motion.div
              className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            <span className="text-sm text-gray-400">Uploading...</span>
          </div>
        ) : (
          <>
            <motion.div
              className="mx-auto mb-2 w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center"
              animate={isDragging ? { scale: 1.1 } : { scale: 1 }}
            >
              <svg
                className="w-5 h-5 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </motion.div>
            <p className="text-sm text-gray-400">
              Drop a file here or{" "}
              <span className="text-indigo-400 font-medium">browse</span>
            </p>
            <p className="text-xs text-gray-600 mt-1">
              PDF, DOCX, TXT, MD, CSV
            </p>
          </>
        )}
      </motion.div>

      {/* Status message */}
      <AnimatePresence>
        {uploadStatus && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className={`mt-2 px-3 py-2 rounded-lg text-xs ${
              uploadStatus.type === "success"
                ? "bg-green-500/10 text-green-400 border border-green-500/20"
                : "bg-red-500/10 text-red-400 border border-red-500/20"
            }`}
          >
            {uploadStatus.message}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
