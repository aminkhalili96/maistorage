/**
 * API client — handles all backend communication including SSE streaming.
 *
 * The main function is streamChat() which opens a POST SSE connection to
 * /api/chat/stream and dispatches typed events to handler callbacks:
 *   - onTrace:       agent pipeline events (classification, retrieval, grading, etc.)
 *   - onCitation:    citation chips as they're created
 *   - onAnswerChunk: token-level streaming of the generated answer
 *   - onDone:        final payload with quality signals (confidence, response_mode, etc.)
 *   - onError:       recoverable/non-recoverable error events
 *
 * In dev mode, API_BASE_URL is empty (Vite proxies /api and /health to the backend).
 * In production, it's set via VITE_API_BASE_URL environment variable.
 */
import type { ChatDonePayload, ChatTurn, EvalRow, IngestionStatus, SourcesResponse, TraceEvent, Citation } from "./types";

const API_BASE_URL = import.meta.env.DEV ? "" : import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getSources(): Promise<SourcesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/sources`);
  return parseJson<SourcesResponse>(response);
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  const response = await fetch(`${API_BASE_URL}/api/ingest/status`);
  return parseJson<IngestionStatus>(response);
}

export async function startIngestion(): Promise<{ job_id: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/ingest/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ families: [], force_refresh: false }),
  });
  return parseJson(response);
}

export async function getRetrievalEvals(): Promise<EvalRow[]> {
  const response = await fetch(`${API_BASE_URL}/api/evals/retrieval`);
  const payload = await parseJson<{ rows: EvalRow[] }>(response);
  return payload.rows;
}

interface StreamHandlers {
  onTrace: (event: TraceEvent) => void;
  onCitation: (citation: Citation) => void;
  onAnswerChunk: (text: string) => void;
  onDone: (payload: ChatDonePayload) => void;
  onError?: (error: { message: string; recoverable: boolean }) => void;
}

/**
 * Open an SSE stream to the backend chat endpoint.
 *
 * Uses fetch + ReadableStream (not EventSource) because we need POST with a JSON body.
 * Manually parses the SSE frame format: "event: <type>\ndata: <json>\n\n".
 * Buffers partial frames across read() calls to handle chunked transfer encoding.
 * 120s timeout auto-aborts the request if the backend hangs.
 */
export async function streamChat(
  question: string,
  history: ChatTurn[],
  handlers: StreamHandlers,
  externalController?: AbortController,
) {
  const controller = externalController ?? new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
    handlers.onError?.({ message: "Request timed out after 120 seconds", recoverable: true });
  }, 120000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: history.map(({ role, content }) => ({ role, content })),
      }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Streaming request failed with ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const lines = frame.split("\n");
        const eventName = lines.find((line) => line.startsWith("event:"))?.replace("event:", "").trim();
        const dataLine = lines.find((line) => line.startsWith("data:"))?.replace("data:", "").trim();
        if (!eventName || !dataLine) {
          continue;
        }
        let payload: unknown;
        try { payload = JSON.parse(dataLine); } catch { continue; }
        if (eventName === "citation") {
          handlers.onCitation(payload as Citation);
        } else if (eventName === "answer_chunk") {
          handlers.onAnswerChunk(String((payload as { text: string }).text ?? ""));
        } else if (eventName === "done") {
          handlers.onDone(payload as ChatDonePayload);
        } else if (eventName === "error") {
          handlers.onError?.(payload as { message: string; recoverable: boolean });
        } else {
          handlers.onTrace({ ...(payload as TraceEvent), type: eventName } as TraceEvent);
        }
      }
    }
  } finally {
    clearTimeout(timeoutId);
  }
}
