import React, { useCallback, useEffect, useRef, useState } from "react";
import { getSources } from "../api";
import type { ChatTurn, SourceRecord, SourcesResponse } from "../types";

const SIDEBAR_WIDTH_KEY = "maistorage-sidebar-width";
const DEFAULT_WIDTH = 208;
const MIN_WIDTH = 160;
const MAX_WIDTH = 400;

/* ================================================================
   Chat session item for the sidebar
   ================================================================ */

function chatTitle(history: ChatTurn[]): string {
  const firstUserMsg = history.find((t) => t.role === "user");
  if (!firstUserMsg) return "New conversation";
  const text = firstUserMsg.content;
  return text.length > 32 ? text.slice(0, 30) + "..." : text;
}

function relativeTime(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

/* ================================================================
   Knowledge base source row
   ================================================================ */

function KbSourceRow({ source, chunkCount }: { source: SourceRecord; chunkCount?: number }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="sidebar-kb-row"
      title={source.title}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true" className="sidebar-kb-icon">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
      </svg>
      <span className="sidebar-kb-title">{source.title}</span>
      {chunkCount != null && chunkCount > 0 && (
        <span className="sidebar-kb-chunks">{chunkCount}</span>
      )}
    </a>
  );
}

/* ================================================================
   Main sidebar
   ================================================================ */

export const Sidebar = React.memo(function Sidebar({
  history,
  onNewChat,
  chunkCounts,
}: {
  history: ChatTurn[];
  onNewChat: () => void;
  chunkCounts: Record<string, number>;
}) {
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [indexedChunks, setIndexedChunks] = useState(0);
  const [chatTimestamp] = useState(() => Date.now());
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try {
      const stored = localStorage.getItem(SIDEBAR_WIDTH_KEY);
      if (stored) {
        const w = Number(stored);
        if (w >= MIN_WIDTH && w <= MAX_WIDTH) return w;
      }
    } catch { /* ignore */ }
    return DEFAULT_WIDTH;
  });
  const isDragging = useRef(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, ev.clientX));
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      // Persist on release
      setSidebarWidth((w) => {
        try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(w)); } catch { /* ignore */ }
        return w;
      });
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  useEffect(() => {
    getSources()
      .then((r: SourcesResponse) => {
        setSources(r.sources);
        setIndexedChunks(r.indexed_chunks);
      })
      .catch(() => {});
  }, []);

  const hasChat = history.length > 0;

  return (
    <aside className="sidebar" style={{ width: sidebarWidth }}>
      {/* Drag handle */}
      <div className="sidebar-resize-handle" onMouseDown={handleMouseDown} />
      <div className="sidebar-top">
        {/* New chat button */}
        <button
          type="button"
          className="sidebar-new-chat-btn"
          onClick={onNewChat}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New chat
        </button>

        {/* Chat history section */}
        <div className="sidebar-section">
          <div className="sidebar-section-label">CHATS</div>
          <div className="sidebar-chat-list">
            {hasChat ? (
              <div className="sidebar-chat-item active">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true" className="sidebar-chat-icon">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <div className="sidebar-chat-info">
                  <span className="sidebar-chat-title">{chatTitle(history)}</span>
                  <span className="sidebar-chat-time">{relativeTime(chatTimestamp)}</span>
                </div>
              </div>
            ) : (
              <div className="sidebar-chat-empty">No conversations yet</div>
            )}
          </div>
        </div>
      </div>

      {/* Knowledge base section */}
      <div className="sidebar-bottom">
        <div className="sidebar-divider" />
        <div className="sidebar-section">
          <div className="sidebar-section-label">KNOWLEDGE BASE</div>
          <div className="sidebar-kb-list">
            {sources.map((s) => (
              <KbSourceRow
                key={s.id}
                source={s}
                chunkCount={chunkCounts[s.id]}
              />
            ))}
          </div>
        </div>
        {sources.length > 0 && (
          <div className="sidebar-kb-footer">
            {sources.length} documents, {indexedChunks.toLocaleString()} chunks
          </div>
        )}
      </div>
    </aside>
  );
});
