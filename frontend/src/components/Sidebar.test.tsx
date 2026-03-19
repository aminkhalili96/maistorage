import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { Sidebar } from "./Sidebar";
import type { Conversation, SourceRecord, SourcesResponse } from "../types";

// ---------------------------------------------------------------------------
// Mock API module
// ---------------------------------------------------------------------------

vi.mock("../api", () => ({
  getSources: vi.fn(),
  getIngestionStatus: vi.fn(),
}));

import { getSources } from "../api";

const getSourcesMock = vi.mocked(getSources);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let convIdCounter = 0;

function buildConversation(overrides: Partial<Conversation> = {}): Conversation {
  convIdCounter += 1;
  return {
    id: `conv-${convIdCounter}`,
    title: `Conversation ${convIdCounter}`,
    history: [],
    createdAt: Date.now() - 60_000,
    updatedAt: Date.now(),
    ...overrides,
  };
}

function buildSource(overrides: Partial<SourceRecord> = {}): SourceRecord {
  return {
    id: "nccl",
    title: "NCCL User Guide",
    url: "https://docs.nvidia.com/nccl/",
    doc_family: "networking",
    doc_type: "html",
    crawl_prefix: "https://docs.nvidia.com/nccl/",
    product_tags: ["nccl"],
    ...overrides,
  };
}

function buildSourcesResponse(
  sources: SourceRecord[] = [],
  indexedChunks = 500,
): SourcesResponse {
  return {
    sources,
    families: {},
    indexed_chunks: indexedChunks,
    app_mode: "dev",
  };
}

const defaultProps = {
  conversations: [] as Conversation[],
  activeId: null as string | null,
  onNewChat: vi.fn(),
  onSwitchConversation: vi.fn(),
  onDeleteConversation: vi.fn(),
  chunkCounts: {} as Record<string, number>,
};

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  convIdCounter = 0;
  vi.clearAllMocks();
  // Default: getSources resolves with empty list (tests that need sources override this)
  getSourcesMock.mockResolvedValue(buildSourcesResponse());
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Sidebar", () => {
  test("renders conversation list with titles", async () => {
    const conversations = [
      buildConversation({ title: "NVLink bandwidth question" }),
      buildConversation({ title: "DGX cluster setup" }),
    ];

    render(<Sidebar {...defaultProps} conversations={conversations} />);

    expect(screen.getByText("NVLink bandwidth question")).toBeInTheDocument();
    expect(screen.getByText("DGX cluster setup")).toBeInTheDocument();
  });

  test("active conversation has .active class", () => {
    const conversations = [
      buildConversation({ id: "c1", title: "First chat" }),
      buildConversation({ id: "c2", title: "Second chat" }),
    ];

    render(
      <Sidebar {...defaultProps} conversations={conversations} activeId="c2" />,
    );

    // The active item should have the "active" class
    const activeItem = screen.getByText("Second chat").closest(".sidebar-chat-item");
    expect(activeItem).toHaveClass("active");

    // The inactive item should not
    const inactiveItem = screen.getByText("First chat").closest(".sidebar-chat-item");
    expect(inactiveItem).not.toHaveClass("active");
  });

  test("clicking conversation calls onSwitchConversation", () => {
    const onSwitchConversation = vi.fn();
    const conversations = [
      buildConversation({ id: "abc-123", title: "GPU memory question" }),
    ];

    render(
      <Sidebar
        {...defaultProps}
        conversations={conversations}
        onSwitchConversation={onSwitchConversation}
      />,
    );

    fireEvent.click(screen.getByText("GPU memory question"));
    expect(onSwitchConversation).toHaveBeenCalledTimes(1);
    expect(onSwitchConversation).toHaveBeenCalledWith("abc-123");
  });

  test("delete button calls onDeleteConversation and stops propagation", () => {
    const onDeleteConversation = vi.fn();
    const onSwitchConversation = vi.fn();
    const conversations = [
      buildConversation({ id: "del-1", title: "To be deleted" }),
    ];

    render(
      <Sidebar
        {...defaultProps}
        conversations={conversations}
        onSwitchConversation={onSwitchConversation}
        onDeleteConversation={onDeleteConversation}
      />,
    );

    const deleteBtn = screen.getByRole("button", { name: "Delete conversation" });
    fireEvent.click(deleteBtn);

    expect(onDeleteConversation).toHaveBeenCalledTimes(1);
    expect(onDeleteConversation).toHaveBeenCalledWith("del-1");
    // stopPropagation should prevent onSwitchConversation from firing
    expect(onSwitchConversation).not.toHaveBeenCalled();
  });

  test("new chat button calls onNewChat", () => {
    const onNewChat = vi.fn();

    render(<Sidebar {...defaultProps} onNewChat={onNewChat} />);

    const newChatBtn = screen.getByRole("button", { name: /new chat/i });
    fireEvent.click(newChatBtn);

    expect(onNewChat).toHaveBeenCalledTimes(1);
  });

  test("KB section shows source list from mocked API", async () => {
    const sources = [
      buildSource({ id: "nccl", title: "NCCL User Guide" }),
      buildSource({ id: "cuda-toolkit", title: "CUDA Toolkit Docs" }),
    ];
    getSourcesMock.mockResolvedValue(buildSourcesResponse(sources, 1234));

    render(
      <Sidebar
        {...defaultProps}
        chunkCounts={{ nccl: 42, "cuda-toolkit": 88 }}
      />,
    );

    // Wait for async sources fetch to resolve and re-render
    await waitFor(() => {
      expect(screen.getByText("NCCL User Guide")).toBeInTheDocument();
    });

    expect(screen.getByText("CUDA Toolkit Docs")).toBeInTheDocument();

    // Chunk counts rendered next to source titles
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("88")).toBeInTheDocument();

    // Footer shows document count and indexed chunks
    expect(screen.getByText("2 documents, 1,234 chunks")).toBeInTheDocument();
  });

  test("KB collapse toggle hides and shows source list", async () => {
    const sources = [
      buildSource({ id: "fabric-manager", title: "Fabric Manager Guide" }),
    ];
    getSourcesMock.mockResolvedValue(buildSourcesResponse(sources, 100));

    render(<Sidebar {...defaultProps} />);

    // Wait for sources to load
    await waitFor(() => {
      expect(screen.getByText("Fabric Manager Guide")).toBeInTheDocument();
    });

    // Click the toggle to collapse
    const toggleBtn = screen.getByRole("button", { name: /knowledge base/i });
    fireEvent.click(toggleBtn);

    // Source should be hidden
    expect(screen.queryByText("Fabric Manager Guide")).not.toBeInTheDocument();

    // Click again to expand
    fireEvent.click(toggleBtn);

    // Source should be visible again
    expect(screen.getByText("Fabric Manager Guide")).toBeInTheDocument();
  });

  test("empty state shows 'No conversations yet'", () => {
    render(<Sidebar {...defaultProps} conversations={[]} />);

    expect(screen.getByText("No conversations yet")).toBeInTheDocument();
  });
});
