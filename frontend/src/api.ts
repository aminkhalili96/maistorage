import type { ChatDonePayload, ChatTurn, EvalRow, IngestionStatus, SourcesResponse, TraceEvent, Citation } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
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
}

export async function streamChat(question: string, history: ChatTurn[], handlers: StreamHandlers) {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
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
      const payload = JSON.parse(dataLine);
      if (eventName === "citation") {
        handlers.onCitation(payload as Citation);
      } else if (eventName === "answer_chunk") {
        handlers.onAnswerChunk(String((payload as { text: string }).text ?? ""));
      } else if (eventName === "done") {
        handlers.onDone(payload as ChatDonePayload);
      } else {
        handlers.onTrace({ ...(payload as TraceEvent), type: eventName } as TraceEvent);
      }
    }
  }
}
