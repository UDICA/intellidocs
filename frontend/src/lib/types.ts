export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export interface Source {
  content: string;
  score: number;
  metadata: Record<string, string | number>;
}

export interface DocumentInfo {
  source: string;
  format?: string;
  document_id?: string;
}

export interface ChatRequest {
  query: string;
  conversation_history: { role: string; content: string }[];
  top_k?: number;
}
