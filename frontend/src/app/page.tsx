"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useChat } from "@/hooks/useChat";
import ChatWindow from "@/components/Chat/ChatWindow";
import InputBar from "@/components/Chat/InputBar";
import UploadPanel from "@/components/Documents/UploadPanel";
import DocumentList from "@/components/Documents/DocumentList";
import SettingsPanel from "@/components/Settings/SettingsPanel";
import SourceCard from "@/components/Sources/SourceCard";

export default function Home() {
  const { messages, isStreaming, currentSources, sendMessage, clearChat } =
    useChat();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [topK, setTopK] = useState(5);
  const [scoreThreshold, setScoreThreshold] = useState(0.3);
  const [refreshDocs, setRefreshDocs] = useState(0);

  const handleSend = useCallback(
    (message: string) => {
      sendMessage(message, topK);
    },
    [sendMessage, topK]
  );

  const handleUploadComplete = useCallback(() => {
    setRefreshDocs((prev) => prev + 1);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* Sidebar */}
      <AnimatePresence initial={false}>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="flex-shrink-0 border-r border-gray-800 bg-gray-950 flex flex-col overflow-hidden"
          >
            <div className="flex flex-col h-full w-80">
              {/* Sidebar header */}
              <div className="px-4 py-4 border-b border-gray-800">
                <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
                  IntelliDocs
                </h1>
                <p className="text-[11px] text-gray-600 mt-0.5">
                  Intelligent Document Understanding
                </p>
              </div>

              {/* Upload */}
              <UploadPanel onUploadComplete={handleUploadComplete} />

              {/* Document list */}
              <DocumentList refreshTrigger={refreshDocs} />

              {/* Settings */}
              <SettingsPanel
                topK={topK}
                onTopKChange={setTopK}
                scoreThreshold={scoreThreshold}
                onScoreThresholdChange={setScoreThreshold}
              />
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm">
          {/* Sidebar toggle */}
          <motion.button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
            whileTap={{ scale: 0.95 }}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              {sidebarOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </motion.button>

          {/* Title (shown when sidebar collapsed) */}
          <AnimatePresence>
            {!sidebarOpen && (
              <motion.h1
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="text-sm font-semibold bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent"
              >
                IntelliDocs
              </motion.h1>
            )}
          </AnimatePresence>

          <div className="flex-1" />

          {/* Sources toggle */}
          {currentSources.length > 0 && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                sourcesOpen
                  ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700"
              }`}
              whileTap={{ scale: 0.95 }}
              aria-label="Toggle sources panel"
              aria-pressed={sourcesOpen}
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Sources ({currentSources.length})
            </motion.button>
          )}

          {/* Clear chat */}
          {messages.length > 0 && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              onClick={clearChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-600 transition-colors"
              whileTap={{ scale: 0.95 }}
              aria-label="Clear chat history"
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
              Clear
            </motion.button>
          )}
        </header>

        {/* Chat + Sources layout */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat area */}
          <div className="flex-1 flex flex-col min-w-0">
            <ChatWindow messages={messages} isStreaming={isStreaming} />
            <InputBar onSend={handleSend} disabled={isStreaming} />
          </div>

          {/* Sources panel */}
          <AnimatePresence>
            {sourcesOpen && currentSources.length > 0 && (
              <motion.aside
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 320, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="flex-shrink-0 border-l border-gray-800 bg-gray-950 overflow-hidden"
              >
                <div className="w-80 h-full overflow-y-auto p-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">
                    Retrieved Sources
                  </h3>
                  <div className="space-y-2">
                    {currentSources.map((source, idx) => (
                      <SourceCard key={idx} source={source} index={idx} />
                    ))}
                  </div>
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
