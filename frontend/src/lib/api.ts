const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function* streamChat(
  query: string,
  conversationHistory: { role: string; content: string }[],
  topK?: number
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      conversation_history: conversationHistory,
      top_k: topK,
    }),
  });

  if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);
  if (!response.body) throw new Error("No response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEvent = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6).trim();
        try {
          yield { event: currentEvent, data: JSON.parse(data) };
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}

export async function uploadDocument(file: File): Promise<Record<string, unknown>> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
  return response.json();
}

export async function listDocuments(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/api/documents`);
  if (!response.ok) throw new Error(`List failed: ${response.status}`);
  return response.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`Delete failed: ${response.status}`);
}

export async function healthCheck(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}
