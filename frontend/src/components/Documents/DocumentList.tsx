"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { listDocuments, deleteDocument } from "@/lib/api";

interface DocumentEntry {
  source: string;
  format?: string;
  document_id?: string;
  chunk_count?: number;
}

interface DocumentListProps {
  refreshTrigger?: number;
}

const listVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
  exit: {
    opacity: 0,
    x: 20,
    height: 0,
    marginBottom: 0,
    paddingTop: 0,
    paddingBottom: 0,
    transition: { duration: 0.2 },
  },
};

const FORMAT_COLORS: Record<string, string> = {
  pdf: "bg-red-500/20 text-red-400",
  docx: "bg-blue-500/20 text-blue-400",
  txt: "bg-gray-500/20 text-gray-400",
  md: "bg-green-500/20 text-green-400",
  csv: "bg-yellow-500/20 text-yellow-400",
};

export default function DocumentList({ refreshTrigger }: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const data = (await listDocuments()) as {
        documents?: DocumentEntry[];
      };
      setDocuments(data.documents || []);
    } catch {
      // API may not be available yet
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs, refreshTrigger]);

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== docId));
    } catch {
      // Silently fail on delete error
    }
  };

  const getFormat = (doc: DocumentEntry): string => {
    if (doc.format) return doc.format.toLowerCase();
    const ext = doc.source.split(".").pop()?.toLowerCase();
    return ext || "txt";
  };

  return (
    <div className="p-3 flex-1 overflow-y-auto">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">
        Documents
      </h3>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <motion.div
            className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        </div>
      ) : documents.length === 0 ? (
        <p className="text-xs text-gray-600 text-center py-4">
          No documents yet. Upload one above.
        </p>
      ) : (
        <motion.ul
          variants={listVariants}
          initial="hidden"
          animate="visible"
          className="space-y-1.5"
          role="list"
        >
          <AnimatePresence>
            {documents.map((doc) => {
              const format = getFormat(doc);
              const colorClass =
                FORMAT_COLORS[format] || "bg-gray-500/20 text-gray-400";
              const fileName = doc.source.split("/").pop() || doc.source;

              return (
                <motion.li
                  key={doc.document_id || doc.source}
                  variants={itemVariants}
                  exit="exit"
                  layout
                  className="group flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900/50 hover:bg-gray-800/70 transition-colors"
                >
                  {/* Format badge */}
                  <span
                    className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold uppercase ${colorClass}`}
                  >
                    {format}
                  </span>

                  {/* Filename */}
                  <span
                    className="flex-1 text-sm text-gray-300 truncate"
                    title={doc.source}
                  >
                    {fileName}
                  </span>

                  {/* Chunk count */}
                  {doc.chunk_count !== undefined && (
                    <span className="text-[10px] text-gray-600 font-mono">
                      {doc.chunk_count}ch
                    </span>
                  )}

                  {/* Delete button */}
                  {doc.document_id && (
                    <motion.button
                      onClick={() => handleDelete(doc.document_id!)}
                      className="flex-shrink-0 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 text-gray-600 hover:text-red-400 transition-all"
                      whileTap={{ scale: 0.9 }}
                      aria-label={`Delete ${fileName}`}
                      tabIndex={0}
                    >
                      <svg
                        className="w-3.5 h-3.5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </motion.button>
                  )}
                </motion.li>
              );
            })}
          </AnimatePresence>
        </motion.ul>
      )}
    </div>
  );
}
