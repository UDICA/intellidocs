"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface SettingsPanelProps {
  topK: number;
  onTopKChange: (value: number) => void;
  scoreThreshold: number;
  onScoreThresholdChange: (value: number) => void;
}

interface AccordionSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function AccordionSection({
  title,
  icon,
  children,
  defaultOpen = false,
}: AccordionSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-gray-800 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-gray-800/50 transition-colors"
        aria-expanded={isOpen}
      >
        <span className="text-gray-500">{icon}</span>
        <span className="flex-1 text-xs font-semibold uppercase tracking-wider text-gray-500">
          {title}
        </span>
        <motion.svg
          className="w-4 h-4 text-gray-600"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </motion.svg>
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function SettingsPanel({
  topK,
  onTopKChange,
  scoreThreshold,
  onScoreThresholdChange,
}: SettingsPanelProps) {
  return (
    <div className="border-t border-gray-800">
      <AccordionSection
        title="Model"
        icon={
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        }
      >
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-gray-900/50 border border-gray-800">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm text-gray-300">OpenRouter</span>
        </div>
        <p className="text-[10px] text-gray-600 mt-1.5">
          Configured via backend environment
        </p>
      </AccordionSection>

      <AccordionSection
        title="Retrieval"
        defaultOpen
        icon={
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        }
      >
        <div className="space-y-4">
          {/* Top-K slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label
                htmlFor="topK"
                className="text-xs text-gray-400"
              >
                Results (top_k)
              </label>
              <span className="text-xs font-mono text-indigo-400">{topK}</span>
            </div>
            <input
              id="topK"
              type="range"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
              className="w-full h-1.5 bg-gray-800 rounded-full appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-gray-700 mt-0.5">
              <span>1</span>
              <span>20</span>
            </div>
          </div>

          {/* Score threshold slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label
                htmlFor="scoreThreshold"
                className="text-xs text-gray-400"
              >
                Score threshold
              </label>
              <span className="text-xs font-mono text-indigo-400">
                {scoreThreshold.toFixed(2)}
              </span>
            </div>
            <input
              id="scoreThreshold"
              type="range"
              min={0}
              max={100}
              value={scoreThreshold * 100}
              onChange={(e) =>
                onScoreThresholdChange(Number(e.target.value) / 100)
              }
              className="w-full h-1.5 bg-gray-800 rounded-full appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-gray-700 mt-0.5">
              <span>0.00</span>
              <span>1.00</span>
            </div>
          </div>
        </div>
      </AccordionSection>
    </div>
  );
}
