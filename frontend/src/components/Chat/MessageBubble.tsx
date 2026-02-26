"use client";

import { motion } from "framer-motion";
import { Message } from "@/lib/types";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export default function MessageBubble({
  message,
  isStreaming,
}: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
    >
      <div
        className={`relative max-w-[80%] rounded-2xl px-4 py-3 shadow-lg ${
          isUser
            ? "bg-gradient-to-br from-indigo-500 to-violet-600 text-white"
            : "bg-gray-800 text-gray-100 border border-gray-700/50"
        }`}
      >
        {/* Message content */}
        <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
          {isStreaming && !isUser && (
            <motion.span
              className="inline-block ml-0.5 w-2 h-4 bg-indigo-400 rounded-sm"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
            />
          )}
        </div>

        {/* Source chips */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-700/50 flex flex-wrap gap-1.5">
            {message.sources.map((source, idx) => {
              const sourceName =
                (source.metadata?.source as string) || `Source ${idx + 1}`;
              const fileName = sourceName.split("/").pop() || sourceName;
              const scorePercent = Math.round(source.score * 100);
              const scoreColor =
                source.score > 0.8
                  ? "bg-green-500/20 text-green-400 border-green-500/30"
                  : source.score > 0.5
                    ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                    : "bg-red-500/20 text-red-400 border-red-500/30";

              return (
                <span
                  key={idx}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border ${scoreColor}`}
                >
                  <svg
                    className="w-3 h-3"
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
                  {fileName}
                  <span className="font-mono">{scorePercent}%</span>
                </span>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}
