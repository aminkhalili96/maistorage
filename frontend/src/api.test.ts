import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { checkHealth, getIngestionStatus, getSources, streamChat } from "./api";
import type { ChatDonePayload, ChatTurn, Citation, IngestionStatus, SourcesResponse, TraceEvent } from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSSEResponse(...frames: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame + "\n\n"));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function makeJsonResponse<T>(body: T, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function buildDonePayload(overrides: Partial<ChatDonePayload> = {}): ChatDonePayload {
  return {
    assistant_mode: "doc_rag",
    answer: "The H100 uses HBM3 memory.",
    confidence: 0.85,
    used_fallback: false,
    response_mode: "corpus-backed",
    retry_count: 0,
    grounding_passed: true,
    answer_quality_passed: true,
    rejected_chunk_count: 0,
    citation_count: 1,
    query_class: "hardware_topology",
    source_families: ["hardware"],
    model_used: "gpt-5.4",
    ...overrides,
  };
}

function buildCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "chunk-42",
    title: "H100 Datasheet",
    url: "https://docs.nvidia.com/h100/",
    citation_url: "https://docs.nvidia.com/h100/#memory",
    domain: "docs.nvidia.com",
    section_path: "Memory Subsystem",
    snippet: "The H100 features 80 GB of HBM3 memory.",
    source_kind: "corpus",
    ...overrides,
  };
}

function makeStreamHandlers() {
  return {
    onTrace: vi.fn<(event: TraceEvent) => void>(),
    onCitation: vi.fn<(citation: Citation) => void>(),
    onAnswerChunk: vi.fn<(text: string) => void>(),
    onDone: vi.fn<(payload: ChatDonePayload) => void>(),
    onError: vi.fn<(error: { message: string; recoverable: boolean }) => void>(),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("api module", () => {
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = vi.fn();
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // --- checkHealth ---

  describe("checkHealth", () => {
    test("returns true on 200 response", async () => {
      mockFetch.mockResolvedValue(new Response("OK", { status: 200 }));

      const result = await checkHealth();

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/health"),
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });

    test("returns false on network error", async () => {
      mockFetch.mockRejectedValue(new TypeError("Failed to fetch"));

      const result = await checkHealth();

      expect(result).toBe(false);
    });
  });

  // --- getSources ---

  describe("getSources", () => {
    test("parses SourcesResponse correctly", async () => {
      const sourcesPayload: SourcesResponse = {
        sources: [
          {
            id: "nccl",
            title: "NCCL Documentation",
            url: "https://docs.nvidia.com/nccl/",
            doc_family: "networking",
            doc_type: "html",
            crawl_prefix: "https://docs.nvidia.com/nccl/",
            product_tags: ["nccl", "multi-gpu"],
            source_kind: "corpus",
          },
        ],
        families: { networking: 1 },
        indexed_chunks: 350,
        app_mode: "dev",
      };
      mockFetch.mockResolvedValue(makeJsonResponse(sourcesPayload));

      const result = await getSources();

      expect(result).toEqual(sourcesPayload);
      expect(result.sources).toHaveLength(1);
      expect(result.sources[0].id).toBe("nccl");
      expect(result.indexed_chunks).toBe(350);
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("/api/sources"));
    });
  });

  // --- getIngestionStatus ---

  describe("getIngestionStatus", () => {
    test("parses IngestionStatus correctly", async () => {
      const statusPayload: IngestionStatus = {
        active: false,
        last_job_id: "job-abc",
        snapshot_id: "snap-001",
        source_counts: { nccl: 12, "cuda-toolkit": 34 },
        chunk_counts: { nccl: 120, "cuda-toolkit": 340 },
        changed_sources: [],
        errors: [],
        updated_at: "2026-03-20T10:00:00Z",
        loaded_demo_corpus: true,
      };
      mockFetch.mockResolvedValue(makeJsonResponse(statusPayload));

      const result = await getIngestionStatus();

      expect(result).toEqual(statusPayload);
      expect(result.active).toBe(false);
      expect(result.loaded_demo_corpus).toBe(true);
      expect(result.source_counts).toHaveProperty("nccl", 12);
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("/api/ingest/status"));
    });
  });

  // --- streamChat ---

  describe("streamChat", () => {
    const question = "What memory does the H100 use?";
    const history: ChatTurn[] = [];

    test("parses well-formed SSE frames into typed events", async () => {
      const handlers = makeStreamHandlers();
      const citation = buildCitation();
      const donePayload = buildDonePayload();

      mockFetch.mockResolvedValue(
        makeSSEResponse(
          `event: classification\ndata: ${JSON.stringify({ message: "Classified as hardware_topology", payload: { query_class: "hardware_topology" } })}`,
          `event: retrieval\ndata: ${JSON.stringify({ message: "Retrieved 5 chunks", payload: { count: 5 } })}`,
          `event: citation\ndata: ${JSON.stringify(citation)}`,
          `event: answer_chunk\ndata: ${JSON.stringify({ text: "The H100 uses HBM3." })}`,
          `event: done\ndata: ${JSON.stringify(donePayload)}`,
        ),
      );

      await streamChat(question, history, handlers);

      // Trace events: classification + retrieval
      expect(handlers.onTrace).toHaveBeenCalledTimes(2);
      expect(handlers.onTrace).toHaveBeenCalledWith(
        expect.objectContaining({ type: "classification", message: "Classified as hardware_topology" }),
      );
      expect(handlers.onTrace).toHaveBeenCalledWith(
        expect.objectContaining({ type: "retrieval", message: "Retrieved 5 chunks" }),
      );

      // Citation
      expect(handlers.onCitation).toHaveBeenCalledTimes(1);
      expect(handlers.onCitation).toHaveBeenCalledWith(expect.objectContaining({ chunk_id: "chunk-42" }));

      // Answer chunk
      expect(handlers.onAnswerChunk).toHaveBeenCalledTimes(1);
      expect(handlers.onAnswerChunk).toHaveBeenCalledWith("The H100 uses HBM3.");

      // Done
      expect(handlers.onDone).toHaveBeenCalledTimes(1);
      expect(handlers.onDone).toHaveBeenCalledWith(expect.objectContaining({ response_mode: "corpus-backed" }));

      // No errors
      expect(handlers.onError).not.toHaveBeenCalled();
    });

    test("handles done event with full payload", async () => {
      const handlers = makeStreamHandlers();
      const donePayload = buildDonePayload({
        confidence: 0.92,
        citation_count: 3,
        grounding_passed: true,
        answer_quality_passed: true,
        model_used: "gpt-5.4",
        source_families: ["hardware", "networking"],
      });

      mockFetch.mockResolvedValue(
        makeSSEResponse(`event: done\ndata: ${JSON.stringify(donePayload)}`),
      );

      await streamChat(question, history, handlers);

      expect(handlers.onDone).toHaveBeenCalledTimes(1);
      const received = handlers.onDone.mock.calls[0][0];
      expect(received.confidence).toBe(0.92);
      expect(received.citation_count).toBe(3);
      expect(received.grounding_passed).toBe(true);
      expect(received.answer_quality_passed).toBe(true);
      expect(received.model_used).toBe("gpt-5.4");
      expect(received.source_families).toEqual(["hardware", "networking"]);
    });

    test("handles error event and calls onError", async () => {
      const handlers = makeStreamHandlers();
      const errorPayload = { message: "Pipeline timeout on self_reflect node", recoverable: true };

      mockFetch.mockResolvedValue(
        makeSSEResponse(`event: error\ndata: ${JSON.stringify(errorPayload)}`),
      );

      await streamChat(question, history, handlers);

      expect(handlers.onError).toHaveBeenCalledTimes(1);
      expect(handlers.onError).toHaveBeenCalledWith({
        message: "Pipeline timeout on self_reflect node",
        recoverable: true,
      });
      expect(handlers.onDone).not.toHaveBeenCalled();
    });

    test("skips frames with malformed JSON in data line", async () => {
      const handlers = makeStreamHandlers();

      mockFetch.mockResolvedValue(
        makeSSEResponse(
          `event: classification\ndata: {not valid json!!!}`,
          `event: answer_chunk\ndata: ${JSON.stringify({ text: "Valid chunk." })}`,
        ),
      );

      await streamChat(question, history, handlers);

      // Malformed frame skipped, valid frame processed
      expect(handlers.onTrace).not.toHaveBeenCalled();
      expect(handlers.onAnswerChunk).toHaveBeenCalledTimes(1);
      expect(handlers.onAnswerChunk).toHaveBeenCalledWith("Valid chunk.");
    });

    test("skips empty SSE frames gracefully", async () => {
      const handlers = makeStreamHandlers();

      // Empty frames: no event line, no data line, blank frames
      mockFetch.mockResolvedValue(
        makeSSEResponse(
          "",
          "data: {\"text\": \"orphan\"}",
          "event: orphan_event",
          `event: answer_chunk\ndata: ${JSON.stringify({ text: "After empties." })}`,
        ),
      );

      await streamChat(question, history, handlers);

      // Only the well-formed frame should be processed
      expect(handlers.onAnswerChunk).toHaveBeenCalledTimes(1);
      expect(handlers.onAnswerChunk).toHaveBeenCalledWith("After empties.");
      expect(handlers.onTrace).not.toHaveBeenCalled();
      expect(handlers.onCitation).not.toHaveBeenCalled();
    });

    test("throws on HTTP 500 response", async () => {
      const handlers = makeStreamHandlers();

      mockFetch.mockResolvedValue(
        new Response("Internal Server Error", { status: 500 }),
      );

      await expect(streamChat(question, history, handlers)).rejects.toThrow(
        "Streaming request failed with 500",
      );

      expect(handlers.onDone).not.toHaveBeenCalled();
      expect(handlers.onAnswerChunk).not.toHaveBeenCalled();
    });

    test("sends correct request body with history", async () => {
      const handlers = makeStreamHandlers();
      const chatHistory: ChatTurn[] = [
        { role: "user", content: "What is NCCL?" },
        {
          role: "assistant",
          content: "NCCL is NVIDIA Collective Communications Library.",
          citations: [buildCitation()],
          trace: [],
          donePayload: buildDonePayload(),
        },
      ];

      mockFetch.mockResolvedValue(
        makeSSEResponse(`event: done\ndata: ${JSON.stringify(buildDonePayload())}`),
      );

      await streamChat("How does it scale?", chatHistory, handlers);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/chat/stream"),
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
        }),
      );

      // Verify the body strips extra fields (citations, trace, donePayload) from history
      const callArgs = mockFetch.mock.calls[0];
      const body = JSON.parse(callArgs[1].body);
      expect(body.question).toBe("How does it scale?");
      expect(body.history).toHaveLength(2);
      expect(body.history[0]).toEqual({ role: "user", content: "What is NCCL?" });
      expect(body.history[1]).toEqual({ role: "assistant", content: "NCCL is NVIDIA Collective Communications Library." });
      // Extra fields must NOT be sent over the wire
      expect(body.history[1]).not.toHaveProperty("citations");
      expect(body.history[1]).not.toHaveProperty("trace");
      expect(body.history[1]).not.toHaveProperty("donePayload");
    });

    test("clears timeout after successful stream completion", async () => {
      vi.useFakeTimers();
      const handlers = makeStreamHandlers();

      mockFetch.mockResolvedValue(
        makeSSEResponse(`event: done\ndata: ${JSON.stringify(buildDonePayload())}`),
      );

      const promise = streamChat(question, history, handlers);
      await vi.runAllTimersAsync();
      await promise;

      // The 45s timeout should have been cleared; advancing time should not trigger onError
      await vi.advanceTimersByTimeAsync(60_000);
      expect(handlers.onError).not.toHaveBeenCalled();

      vi.useRealTimers();
    });

    test("uses external AbortController when provided", async () => {
      const handlers = makeStreamHandlers();
      const externalController = new AbortController();

      mockFetch.mockResolvedValue(
        makeSSEResponse(`event: done\ndata: ${JSON.stringify(buildDonePayload())}`),
      );

      await streamChat(question, history, handlers, externalController);

      // Verify the signal passed to fetch belongs to the external controller
      const callArgs = mockFetch.mock.calls[0];
      expect(callArgs[1].signal).toBe(externalController.signal);
    });
  });
});
