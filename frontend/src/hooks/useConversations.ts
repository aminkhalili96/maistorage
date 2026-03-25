import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatTurn, Conversation } from "../types";

const STORAGE_KEY = "maistorage-conversations";
const OLD_HISTORY_KEY = "maistorage-chat-history";

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function titleFromHistory(history: ChatTurn[]): string {
  const firstUser = history.find((t) => t.role === "user");
  if (!firstUser) return "New conversation";
  const text = firstUser.content;
  return text.length > 40 ? text.slice(0, 38) + "..." : text;
}

function loadConversations(): Conversation[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) return parsed as Conversation[];
    }
  } catch { /* ignore */ }

  // Migrate old single-conversation history
  try {
    const old = localStorage.getItem(OLD_HISTORY_KEY);
    if (old) {
      const history = JSON.parse(old);
      if (Array.isArray(history) && history.length > 0) {
        const migrated: Conversation = {
          id: generateId(),
          title: titleFromHistory(history as ChatTurn[]),
          history: history as ChatTurn[],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        localStorage.removeItem(OLD_HISTORY_KEY);
        return [migrated];
      }
    }
  } catch { /* ignore */ }

  return [];
}

function persist(conversations: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch { /* storage unavailable */ }
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(() => {
    const loaded = loadConversations();
    return loaded.length > 0 ? loaded[0].id : null;
  });

  // Persist on every change
  const conversationsRef = useRef(conversations);
  conversationsRef.current = conversations;
  useEffect(() => {
    persist(conversations);
  }, [conversations]);

  // Track activeId via ref so callbacks always read the latest value
  // (prevents stale closure when SSE onDone fires after a new conversation is created)
  const activeIdRef = useRef(activeId);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;
  const history = activeConversation?.history ?? [];

  const setHistory = useCallback(
    (updater: ChatTurn[] | ((prev: ChatTurn[]) => ChatTurn[])) => {
      setConversations((prev) => {
        const curId = activeIdRef.current;
        const next = typeof updater === "function"
          ? updater(prev.find((c) => c.id === curId)?.history ?? [])
          : updater;

        // If no active conversation, create one lazily
        if (!curId || !prev.find((c) => c.id === curId)) {
          if (next.length === 0) return prev;
          const newConv: Conversation = {
            id: generateId(),
            title: titleFromHistory(next),
            history: next,
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          // Immediately update the ref so subsequent setHistory calls see the new ID
          // (prevents SSE callbacks from creating duplicate conversations)
          activeIdRef.current = newConv.id;
          // Also update React state (deferred to avoid state update during render)
          setTimeout(() => setActiveId(newConv.id), 0);
          return [newConv, ...prev];
        }

        return prev.map((c) =>
          c.id === curId
            ? {
                ...c,
                history: next,
                title: c.history.length === 0 ? titleFromHistory(next) : c.title,
                updatedAt: Date.now(),
              }
            : c,
        );
      });
    },
    [],
  );

  const newChat = useCallback(() => {
    setActiveId(null);
  }, []);

  const switchConversation = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const filtered = prev.filter((c) => c.id !== id);
        if (activeIdRef.current === id) {
          setTimeout(() => setActiveId(filtered.length > 0 ? filtered[0].id : null), 0);
        }
        return filtered;
      });
    },
    [],
  );

  return {
    conversations,
    activeId,
    history,
    setHistory,
    newChat,
    switchConversation,
    deleteConversation,
  };
}
