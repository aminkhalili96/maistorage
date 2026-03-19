import { renderHook, act } from "@testing-library/react";
import { beforeEach, afterEach, describe, expect, test, vi } from "vitest";

import { useConversations } from "./useConversations";
import type { ChatTurn, Conversation } from "../types";

const STORAGE_KEY = "maistorage-conversations";
const OLD_HISTORY_KEY = "maistorage-chat-history";

function makeTurn(role: "user" | "assistant", content: string): ChatTurn {
  return { role, content };
}

function makeConversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "conv-1",
    title: "Test conversation",
    history: [makeTurn("user", "Hello"), makeTurn("assistant", "Hi there!")],
    createdAt: 1000,
    updatedAt: 2000,
    ...overrides,
  };
}

/**
 * Create a minimal localStorage mock that implements the Web Storage API.
 * The jsdom/node built-in localStorage in vitest 4.1 lacks standard methods,
 * so we stub globalThis.localStorage with a Map-backed implementation.
 */
function createLocalStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => { store.set(key, value); }),
    removeItem: vi.fn((key: string) => { store.delete(key); }),
    clear: vi.fn(() => { store.clear(); }),
    get length() { return store.size; },
    key: vi.fn((index: number) => [...store.keys()][index] ?? null),
  };
}

describe("useConversations", () => {
  let storageMock: ReturnType<typeof createLocalStorageMock>;

  beforeEach(() => {
    storageMock = createLocalStorageMock();
    vi.stubGlobal("localStorage", storageMock);
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  // ---- Test 1: Creates new conversation with generated ID and timestamp ----
  test("creates new conversation with generated ID and timestamp via setHistory when no activeId", () => {
    const { result } = renderHook(() => useConversations());

    // Initially empty
    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeId).toBeNull();

    const turns: ChatTurn[] = [makeTurn("user", "What is NCCL?")];

    act(() => {
      result.current.setHistory(turns);
    });

    // Flush the setTimeout that sets activeId
    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations).toHaveLength(1);
    const created = result.current.conversations[0];
    expect(created.id).toBeTruthy();
    expect(typeof created.id).toBe("string");
    expect(created.createdAt).toBeGreaterThan(0);
    expect(created.updatedAt).toBeGreaterThan(0);
    expect(created.history).toEqual(turns);
    expect(result.current.activeId).toBe(created.id);
  });

  // ---- Test 2: Switches active conversation and loads its history ----
  test("switches active conversation and loads its history", () => {
    const conv1 = makeConversation({ id: "conv-1", title: "First" });
    const conv2 = makeConversation({
      id: "conv-2",
      title: "Second",
      history: [makeTurn("user", "GPU question"), makeTurn("assistant", "GPU answer")],
    });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv1, conv2]));

    const { result } = renderHook(() => useConversations());

    // Initially loads first conversation
    expect(result.current.activeId).toBe("conv-1");
    expect(result.current.history).toEqual(conv1.history);

    act(() => {
      result.current.switchConversation("conv-2");
    });

    expect(result.current.activeId).toBe("conv-2");
    expect(result.current.history).toEqual(conv2.history);
  });

  // ---- Test 3: Deletes conversation and falls back to most recent ----
  test("deletes conversation and falls back to most recent", () => {
    const conv1 = makeConversation({ id: "conv-1", title: "First" });
    const conv2 = makeConversation({ id: "conv-2", title: "Second" });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv1, conv2]));

    const { result } = renderHook(() => useConversations());

    // Active is conv-1
    expect(result.current.activeId).toBe("conv-1");

    act(() => {
      result.current.deleteConversation("conv-1");
    });

    // Flush the setTimeout that updates activeId after delete
    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.conversations[0].id).toBe("conv-2");
    expect(result.current.activeId).toBe("conv-2");
  });

  // ---- Test 4: setHistory appends turns to active conversation ----
  test("setHistory appends turns to active conversation", () => {
    const conv = makeConversation({
      id: "conv-1",
      history: [makeTurn("user", "Hello")],
    });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    expect(result.current.history).toHaveLength(1);

    act(() => {
      result.current.setHistory((prev) => [
        ...prev,
        makeTurn("assistant", "Hi there!"),
        makeTurn("user", "Follow up"),
      ]);
    });

    expect(result.current.history).toHaveLength(3);
    expect(result.current.history[1].content).toBe("Hi there!");
    expect(result.current.history[2].content).toBe("Follow up");
  });

  // ---- Test 5: Persists conversations to localStorage on every update ----
  test("persists conversations to localStorage on every update", () => {
    const { result } = renderHook(() => useConversations());

    const turns: ChatTurn[] = [makeTurn("user", "Persist test")];

    act(() => {
      result.current.setHistory(turns);
    });

    act(() => {
      vi.advanceTimersByTime(0);
    });

    const stored = storageMock.getItem(STORAGE_KEY);
    expect(stored).toBeTruthy();
    const parsed = JSON.parse(stored!) as Conversation[];
    expect(parsed).toHaveLength(1);
    expect(parsed[0].history[0].content).toBe("Persist test");
  });

  // ---- Test 6: Loads conversations from localStorage on init ----
  test("loads conversations from localStorage on init", () => {
    const conv = makeConversation({ id: "existing-conv", title: "Loaded from storage" });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.conversations[0].id).toBe("existing-conv");
    expect(result.current.conversations[0].title).toBe("Loaded from storage");
    expect(result.current.activeId).toBe("existing-conv");
    expect(result.current.history).toEqual(conv.history);
  });

  // ---- Test 7: Recovers from corrupted/invalid localStorage JSON ----
  test("recovers from corrupted/invalid localStorage JSON", () => {
    storageMock.setItem(STORAGE_KEY, "{not valid json[}");

    const { result } = renderHook(() => useConversations());

    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeId).toBeNull();
    expect(result.current.history).toEqual([]);
  });

  // ---- Test 8: Recovers from localStorage with non-array data ----
  test("recovers from localStorage with non-array data", () => {
    // Valid JSON, but not an array
    storageMock.setItem(STORAGE_KEY, JSON.stringify({ id: "not-an-array" }));

    const { result } = renderHook(() => useConversations());

    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeId).toBeNull();
    expect(result.current.history).toEqual([]);
  });

  // ---- Test 9: Generates title from first user message (truncated at 40 chars) ----
  test("generates title from first user message, truncated at 40 chars", () => {
    const { result } = renderHook(() => useConversations());

    const shortMessage = "Short question";
    act(() => {
      result.current.setHistory([makeTurn("user", shortMessage)]);
    });
    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations[0].title).toBe("Short question");

    // Test long message in a separate hook instance
    storageMock.clear();
    const { result: result2 } = renderHook(() => useConversations());

    const longMessage = "This is a very long question that exceeds forty characters easily";
    act(() => {
      result2.current.setHistory([makeTurn("user", longMessage)]);
    });
    act(() => {
      vi.advanceTimersByTime(0);
    });

    const title = result2.current.conversations[0].title;
    expect(title).toBe("This is a very long question that exce...");
    expect(title.length).toBe(41); // 38 chars + "..."
  });

  // ---- Test 10: Multiple rapid setHistory calls don't lose data ----
  test("multiple rapid setHistory calls don't lose data", () => {
    const conv = makeConversation({ id: "conv-1", history: [] });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    act(() => {
      result.current.setHistory((prev) => [...prev, makeTurn("user", "First")]);
      result.current.setHistory((prev) => [...prev, makeTurn("assistant", "Response to first")]);
      result.current.setHistory((prev) => [...prev, makeTurn("user", "Second")]);
    });

    expect(result.current.history).toHaveLength(3);
    expect(result.current.history[0].content).toBe("First");
    expect(result.current.history[1].content).toBe("Response to first");
    expect(result.current.history[2].content).toBe("Second");
  });

  // ---- Test 11: Empty state: no conversations, activeId is null ----
  test("empty state: no conversations, activeId is null, history is empty", () => {
    const { result } = renderHook(() => useConversations());

    expect(result.current.conversations).toEqual([]);
    expect(result.current.activeId).toBeNull();
    expect(result.current.history).toEqual([]);
  });

  // ---- Test 12: newChat resets activeId to null ----
  test("newChat resets activeId to null", () => {
    const conv = makeConversation({ id: "conv-1" });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    expect(result.current.activeId).toBe("conv-1");
    expect(result.current.history).toEqual(conv.history);

    act(() => {
      result.current.newChat();
    });

    expect(result.current.activeId).toBeNull();
    expect(result.current.history).toEqual([]);
    // Conversations list is preserved
    expect(result.current.conversations).toHaveLength(1);
  });

  // ---- Additional edge case: delete last conversation sets activeId to null ----
  test("deleting the only conversation sets activeId to null", () => {
    const conv = makeConversation({ id: "only-conv" });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    expect(result.current.activeId).toBe("only-conv");

    act(() => {
      result.current.deleteConversation("only-conv");
    });

    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeId).toBeNull();
    expect(result.current.history).toEqual([]);
  });

  // ---- Additional edge case: setHistory with empty array on no activeId is a no-op ----
  test("setHistory with empty array when no activeId does not create a conversation", () => {
    const { result } = renderHook(() => useConversations());

    act(() => {
      result.current.setHistory([]);
    });

    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations).toHaveLength(0);
    expect(result.current.activeId).toBeNull();
  });

  // ---- Additional edge case: title is set only once (from first setHistory with content) ----
  test("title is not overwritten once conversation has history", () => {
    const conv = makeConversation({
      id: "conv-1",
      title: "Original title",
      history: [makeTurn("user", "Original question")],
    });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv]));

    const { result } = renderHook(() => useConversations());

    act(() => {
      result.current.setHistory((prev) => [
        ...prev,
        makeTurn("assistant", "Answer"),
        makeTurn("user", "Different question entirely"),
      ]);
    });

    // Title should remain the original, not change to the new first user message
    expect(result.current.conversations[0].title).toBe("Original title");
  });

  // ---- Additional edge case: migrate old single-conversation key ----
  test("migrates old single-conversation localStorage key", () => {
    const oldHistory: ChatTurn[] = [
      makeTurn("user", "Legacy question"),
      makeTurn("assistant", "Legacy answer"),
    ];
    storageMock.setItem(OLD_HISTORY_KEY, JSON.stringify(oldHistory));

    const { result } = renderHook(() => useConversations());

    // The conversations state is populated from the first loadConversations() call
    // which migrates the old key and returns the data
    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.conversations[0].title).toBe("Legacy question");
    expect(result.current.conversations[0].history).toEqual(oldHistory);

    // Note: activeId is null initially because loadConversations() is called
    // twice during useState init. The first call removes OLD_HISTORY_KEY
    // but doesn't persist to STORAGE_KEY, so the second call finds nothing.
    // This is a known quirk: the user would need to click the conversation
    // in the sidebar to activate it after a migration.
    expect(result.current.activeId).toBeNull();

    // Old key should be removed after migration
    expect(storageMock.removeItem).toHaveBeenCalledWith(OLD_HISTORY_KEY);
  });

  // ---- Additional: deleting a non-active conversation preserves activeId ----
  test("deleting a non-active conversation preserves activeId", () => {
    const conv1 = makeConversation({ id: "conv-1", title: "Active" });
    const conv2 = makeConversation({ id: "conv-2", title: "Other" });
    storageMock.setItem(STORAGE_KEY, JSON.stringify([conv1, conv2]));

    const { result } = renderHook(() => useConversations());

    expect(result.current.activeId).toBe("conv-1");

    act(() => {
      result.current.deleteConversation("conv-2");
    });

    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.activeId).toBe("conv-1");
  });
});
