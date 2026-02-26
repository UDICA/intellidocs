"use client";

import { useCallback, useReducer, useRef } from "react";
import { streamChat } from "@/lib/api";
import { Message, Source } from "@/lib/types";

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  currentSources: Source[];
}

type ChatAction =
  | { type: "ADD_USER_MESSAGE"; content: string }
  | { type: "START_STREAMING" }
  | { type: "SET_SOURCES"; sources: Source[] }
  | { type: "APPEND_TOKEN"; token: string }
  | { type: "FINISH_STREAMING" }
  | { type: "CLEAR" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "ADD_USER_MESSAGE":
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: action.content,
          },
        ],
      };
    case "START_STREAMING":
      return {
        ...state,
        isStreaming: true,
        currentSources: [],
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "",
          },
        ],
      };
    case "SET_SOURCES":
      return {
        ...state,
        currentSources: action.sources,
        messages: state.messages.map((m, i) =>
          i === state.messages.length - 1
            ? { ...m, sources: action.sources }
            : m
        ),
      };
    case "APPEND_TOKEN": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      messages[messages.length - 1] = {
        ...last,
        content: last.content + action.token,
      };
      return { ...state, messages };
    }
    case "FINISH_STREAMING":
      return { ...state, isStreaming: false };
    case "CLEAR":
      return { messages: [], isStreaming: false, currentSources: [] };
    default:
      return state;
  }
}

export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, {
    messages: [],
    isStreaming: false,
    currentSources: [],
  });

  // Use a ref to track the latest messages for the sendMessage closure
  const messagesRef = useRef(state.messages);
  messagesRef.current = state.messages;

  const sendMessage = useCallback(
    async (content: string, topK?: number) => {
      const history = messagesRef.current.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      dispatch({ type: "ADD_USER_MESSAGE", content });
      dispatch({ type: "START_STREAMING" });

      try {
        for await (const { event, data } of streamChat(
          content,
          history,
          topK
        )) {
          if (event === "sources") {
            dispatch({
              type: "SET_SOURCES",
              sources: data.sources as Source[],
            });
          } else if (event === "token") {
            dispatch({
              type: "APPEND_TOKEN",
              token: data.token as string,
            });
          }
        }
      } catch {
        dispatch({
          type: "APPEND_TOKEN",
          token: "\n\n*Error: Failed to get response. Please try again.*",
        });
      } finally {
        dispatch({ type: "FINISH_STREAMING" });
      }
    },
    []
  );

  const clearChat = useCallback(() => dispatch({ type: "CLEAR" }), []);

  return {
    messages: state.messages,
    isStreaming: state.isStreaming,
    currentSources: state.currentSources,
    sendMessage,
    clearChat,
  };
}
