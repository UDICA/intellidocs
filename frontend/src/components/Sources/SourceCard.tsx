"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Source } from "@/lib/types";

interface SourceCardProps {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const scorePercent = Math.round(source.score * 100);
  const sourceName =
    (source.metadata?.source as string) || `Source ${index + 1}`;
  const fileName = sourceName.split("/").pop() || sourceName;

  const scoreColor =
    source.score > 0.8
      ? "bg-green-500/20 text-green-400 border-green-500/30"
      : source.score > 0.5
        ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
        : "bg-red-500/20 text-red-400 border-red-500/30";

  const glowColor =
    source.score > 0.8
      ? "hover:shadow-green-500/10"
      : source.score > 0.5
        ? "hover:shadow-yellow-500/10"
        : "hover:shadow-red-500/10";

  return (
    <motion.div
      layout
      className={`rounded-xl border border-gray-800 bg-gray-900/80 overflow-hidden cursor-pointer hover:shadow-lg ${glowColor} transition-shadow`}
      onClick={() => setExpanded(!expanded)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setExpanded(!expanded);
        }
      }}
      role="button"
      tabIndex={0}
      aria-expanded={expanded}
      aria-label={`Source: ${fileName}, relevance ${scorePercent}%`}
      whileHover={{ scale: 1.01 }}
    >
      {/* Compact header */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <svg
          className="w-4 h-4 text-gray-500 flex-shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <span className="flex-1 text-sm text-gray-300 truncate">
          {fileName}
        </span>
        <span
          className={`px-2 py-0.5 rounded-full text-xs font-mono font-semibold border ${scoreColor}`}
        >
          {scorePercent}%
        </span>
        <motion.svg
          className="w-4 h-4 text-gray-500 flex-shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </motion.svg>
      </div>

      {/* Expanded content */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 border-t border-gray-800">
              <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-wrap">
                {source.content}
              </p>
              {/* Metadata */}
              {Object.keys(source.metadata).length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {Object.entries(source.metadata).map(([key, val]) => (
                    <span
                      key={key}
                      className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-gray-800 text-gray-500 font-mono"
                    >
                      {key}: {String(val)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
