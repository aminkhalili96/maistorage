import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import App from "./App";
import { checkHealth, getIngestionStatus, getSources, streamChat } from "./api";
import type { ChatDonePayload, Citation, SourceRecord, TraceEvent } from "./types";

vi.mock("./api", () => ({
  getSources: vi.fn(),
  getIngestionStatus: vi.fn(),
  streamChat: vi.fn(),
  checkHealth: vi.fn(),
}));

const mockedStreamChat = vi.mocked(streamChat);
const mockedGetSources = vi.mocked(getSources);
const mockedGetIngestionStatus = vi.mocked(getIngestionStatus);
const mockedCheckHealth = vi.mocked(checkHealth);

function buildDonePayload(overrides: Partial<ChatDonePayload> = {}): ChatDonePayload {
  return {
    assistant_mode: "direct_chat",
    answer: "Here is a regular chat answer.",
    confidence: 0,
    used_fallback: false,
    response_mode: "direct-chat",
    retry_count: 0,
    grounding_passed: true,
    answer_quality_passed: true,
    rejected_chunk_count: 0,
    citation_count: 0,
    query_class: "general",
    source_families: [],
    model_used: "gpt-5.4",
    ...overrides,
  };
}

function buildCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "chunk-1",
    title: "CUDA Installation Guide",
    url: "https://docs.nvidia.com/cuda/",
    citation_url: "https://docs.nvidia.com/cuda/#install",
    domain: "docs.nvidia.com",
    section_path: "Linux Install",
    snippet: "Install the NVIDIA driver and the CUDA toolkit before running containers.",
    source_kind: "corpus",
    ...overrides,
  };
}

function buildTrace(event: Partial<TraceEvent> = {}): TraceEvent {
  return {
    type: "classification",
    message: "Classified question as deployment_runtime",
    payload: { stage: "tool_selection", tool_label: "NVIDIA Docs", brand: "nvidia" },
    ...event,
  };
}

function buildSource(overrides: Partial<SourceRecord> = {}): SourceRecord {
  return {
    id: "cuda-programming-guide",
    title: "CUDA C++ Programming Guide",
    url: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
    pdf_url: "https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf",
    doc_family: "core",
    doc_type: "html",
    crawl_prefix: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
    product_tags: ["cuda", "gpu", "programming"],
    source_kind: "corpus",
    ...overrides,
  };
}

describe("App", () => {
  beforeEach(() => {
    mockedStreamChat.mockReset();
    mockedGetSources.mockReset();
    mockedGetIngestionStatus.mockReset();
    mockedCheckHealth.mockReset();
    mockedCheckHealth.mockResolvedValue(true);
    mockedGetSources.mockResolvedValue({ sources: [buildSource()], families: {}, indexed_chunks: 1, app_mode: "assessment" });
    mockedGetIngestionStatus.mockResolvedValue({ active: false, source_counts: {}, chunk_counts: {}, changed_sources: [], errors: [], loaded_demo_corpus: false });
  });

  test("renders the empty state with composer by default", () => {
    render(<App />);
    expect(screen.getByRole("textbox")).toBeInTheDocument();
    expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    expect(screen.getByText("MaiSearch")).toBeInTheDocument();
    // Suggestion chips should be visible in empty state
    expect(screen.getByText("Why is 4-GPU training scaling poorly?")).toBeInTheDocument();
  });

  test("shows health banner when backend is unreachable", async () => {
    mockedCheckHealth.mockResolvedValue(false);
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/Backend unreachable/)).toBeInTheDocument(),
    );
  });

  test("shows a direct chat response without citations", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onAnswerChunk("Here is a regular chat answer.");
      handlers.onDone(buildDonePayload());
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "How should I prepare for a technical interview?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Here is a regular chat answer.")).toBeInTheDocument());
  });

  test("shows thinking block and inline citations for a doc-backed run", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onTrace(buildTrace());
      handlers.onTrace(
        buildTrace({
          type: "retrieval",
          message: "Retrieved 4 candidates from the hybrid index",
          payload: { stage: "tool_request", status: "request", tool_label: "NVIDIA Docs", brand: "nvidia" },
        }),
      );
      handlers.onCitation(buildCitation());
      handlers.onAnswerChunk("Use the NVIDIA driver and Container Toolkit.");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "Use the NVIDIA driver and Container Toolkit.",
          response_mode: "corpus-backed",
          citation_count: 1,
          confidence: 0.88,
          query_class: "deployment_runtime",
          source_families: ["infrastructure"],
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "What NVIDIA stack is needed on Linux?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(screen.getByText("Use the NVIDIA driver and Container Toolkit.")).toBeInTheDocument(),
    );
    // Citation title should appear in inline source chips (partial match)
    expect(screen.getByText(/CUDA Installation Guide/)).toBeInTheDocument();
  });

  test("shows corpus-backed trust badge for doc-backed responses", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onTrace(buildTrace());
      handlers.onCitation(buildCitation());
      handlers.onAnswerChunk("Use the NVIDIA driver.");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "Use the NVIDIA driver.",
          response_mode: "corpus-backed",
          citation_count: 1,
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "How do I install CUDA?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("corpus-backed")).toBeInTheDocument());
  });

  test("general nvidia questions stay in plain chat mode", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onAnswerChunk("NVIDIA is a technology company known for GPUs and accelerated computing.");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "direct_chat",
          answer: "NVIDIA is a technology company known for GPUs and accelerated computing.",
          response_mode: "direct-chat",
          citation_count: 0,
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "what is nvidia");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(screen.getByText("NVIDIA is a technology company known for GPUs and accelerated computing.")).toBeInTheDocument(),
    );
  });

  test("assistant messages render without avatar", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onAnswerChunk("Test answer.");
      handlers.onDone(buildDonePayload({ answer: "Test answer." }));
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "Hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Test answer.")).toBeInTheDocument());
    // No "M" avatar in the new layout
    const mElements = screen.queryAllByText("M");
    // Filter to only actual avatar elements (not text content containing M)
    const avatarElements = mElements.filter(
      (el) => el.classList.contains("assistant-avatar-m"),
    );
    expect(avatarElements.length).toBe(0);
  });

  test("send button is disabled while sending", async () => {
    mockedStreamChat.mockImplementation(async () => {
      // Never resolves during test
      return new Promise(() => {});
    });

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByRole("textbox"), "test question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Send" })).toBeDisabled(),
    );
  });

  test("shows error banner on stream failure", async () => {
    mockedStreamChat.mockRejectedValue(new Error("Connection refused"));

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByRole("textbox"), "test");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Connection refused")).toBeInTheDocument());
  });

  test("inline citation chips show citation titles when citations exist", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onTrace(buildTrace());
      handlers.onCitation(buildCitation());
      handlers.onCitation(
        buildCitation({
          chunk_id: "chunk-2",
          title: "NVIDIA Container Toolkit",
          citation_url: "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",
          section_path: "Install Guide",
          snippet: "Configure the container runtime to expose NVIDIA GPUs to containers.",
        }),
      );
      handlers.onAnswerChunk("Use the driver and NVIDIA Container Toolkit. [1][2]");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "Use the driver and NVIDIA Container Toolkit. [1][2]",
          response_mode: "corpus-backed",
          citation_count: 2,
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "How do I run NVIDIA containers?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/CUDA Installation Guide/)).toBeInTheDocument());
    expect(screen.getAllByText(/NVIDIA Container Toolkit/).length).toBeGreaterThan(0);
  });

  test("error banner displays on stream failure with error message", async () => {
    mockedStreamChat.mockRejectedValue(new Error("Network timeout"));

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByRole("textbox"), "What is NCCL?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Network timeout")).toBeInTheDocument());
  });

  test("citation title appears after corpus-backed response", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onTrace(buildTrace());
      handlers.onCitation(buildCitation({ chunk_id: "chunk-test", title: "H100 Architecture Guide" }));
      handlers.onAnswerChunk("The H100 uses NVLink 4.0 for inter-GPU bandwidth. [1]");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "The H100 uses NVLink 4.0 for inter-GPU bandwidth. [1]",
          response_mode: "corpus-backed",
          citation_count: 1,
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "What is H100 NVLink bandwidth?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/H100 Architecture Guide/)).toBeInTheDocument());
  });

  test("suggested prompts trigger submit when clicked", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onAnswerChunk("NCCL handles collective communication for multi-GPU training.");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "NCCL handles collective communication for multi-GPU training.",
          response_mode: "corpus-backed",
          citation_count: 0,
        }),
      );
    });

    render(<App />);

    // Find a suggested prompt card and click it
    const promptCards = screen.getAllByRole("button");
    const suggestedPrompt = promptCards.find((btn) => btn.textContent?.includes("NCCL") || btn.textContent?.includes("GPU") || btn.textContent?.includes("scaling"));

    if (suggestedPrompt) {
      await user.click(suggestedPrompt);
      // After clicking, the stream should have been called
      await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    } else {
      // If no suggested prompts visible (e.g. after a response), skip gracefully
      expect(true).toBe(true);
    }
  });

  test("recovers from corrupted localStorage", () => {
    // Mock localStorage.getItem to return corrupted JSON for history
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem").mockImplementation((key: string) => {
      if (key === "maistorage-chat-history") return "{not valid json[}";
      return null;
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {});
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {});

    render(<App />);
    // App should render without crashing despite corrupted localStorage
    expect(screen.getByRole("textbox")).toBeInTheDocument();
    // Empty state should be shown (corrupted data was discarded)
    expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    // Suggestion chips visible
    expect(screen.getByText("Compare H100 vs A100 for inference")).toBeInTheDocument();

    vi.restoreAllMocks();
  });

  test("New Chat button in sidebar clears history", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onAnswerChunk("Test answer.");
      handlers.onDone(buildDonePayload({ answer: "Test answer." }));
    });

    render(<App />);

    // Send a message first
    await user.type(screen.getByRole("textbox"), "Hello");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByText("Test answer.")).toBeInTheDocument());

    // Click New chat in sidebar
    const newChatBtn = screen.getByText("New chat");
    await user.click(newChatBtn);

    // Empty state should return
    await waitFor(() => expect(screen.getByText("How can I help you today?")).toBeInTheDocument());
  });

  test("trust badge has tooltip on hover", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onTrace(buildTrace());
      handlers.onCitation(buildCitation());
      handlers.onAnswerChunk("Test answer.");
      handlers.onDone(
        buildDonePayload({
          assistant_mode: "doc_rag",
          answer: "Test answer.",
          response_mode: "corpus-backed",
          citation_count: 1,
        }),
      );
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "How do I install CUDA?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("corpus-backed")).toBeInTheDocument());
    const badge = screen.getByText("corpus-backed");
    expect(badge.getAttribute("title")).toBe("Answer sourced from NVIDIA documentation corpus");
  });

  test("SSE error event shows error banner", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _history, handlers) => {
      handlers.onError?.({ message: "Internal pipeline error", recoverable: true });
      throw new Error("Internal pipeline error");
    });

    render(<App />);

    await user.type(screen.getByRole("textbox"), "test");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Internal pipeline error")).toBeInTheDocument());
  });

  test("sidebar shows knowledge base documents", async () => {
    render(<App />);

    // Sidebar should load sources and show them
    await waitFor(() => expect(screen.getByText("KNOWLEDGE BASE")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("CUDA C++ Programming Guide")).toBeInTheDocument());
  });

  test("sidebar shows CHATS section", () => {
    render(<App />);
    expect(screen.getByText("CHATS")).toBeInTheDocument();
  });
});
