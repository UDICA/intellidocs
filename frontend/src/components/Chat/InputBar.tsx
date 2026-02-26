"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { motion } from "framer-motion";

interface InputBarProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export default function InputBar({ onSend, disabled }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const maxHeight = 5 * 24; // ~5 rows
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    }
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-800 bg-gray-950/80 backdrop-blur-sm p-4">
      <div className="flex items-end gap-3 max-w-4xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              adjustHeight();
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            disabled={disabled}
            rows={1}
            className="w-full resize-none rounded-xl bg-gray-900 border border-gray-700 px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Message input"
          />
        </div>
        <motion.button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          whileTap={{ scale: 0.95 }}
          whileHover={{ scale: 1.05 }}
          className="flex-shrink-0 h-11 w-11 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 text-white flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed transition-opacity shadow-lg shadow-indigo-500/25"
          aria-label="Send message"
        >
          {disabled ? (
            <motion.svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="none"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <circle
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="3"
                strokeDasharray="42"
                strokeDashoffset="12"
                strokeLinecap="round"
              />
            </motion.svg>
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19V5m-7 7l7-7 7 7"
              />
            </svg>
          )}
        </motion.button>
      </div>
      <p className="text-center text-xs text-gray-600 mt-2">
        Press Enter to send, Shift+Enter for a new line
      </p>
    </div>
  );
}
