import React, { useCallback, useEffect, useRef, useState } from "react";
import { getSources } from "../api";
import type { Conversation, SourceRecord, SourcesResponse } from "../types";

const SIDEBAR_WIDTH_KEY = "maistorage-sidebar-width";
const DEFAULT_WIDTH = 208;
const MIN_WIDTH = 160;
const MAX_WIDTH = 400;

/* ================================================================
   Relative time helper
   ================================================================ */

function relativeTime(timestamp: number): string {
  const delta = Date.now() - timestamp;
  if (delta < 60_000) return "now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h`;
  return `${Math.floor(delta / 86_400_000)}d`;
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
  conversations,
  activeId,
  onNewChat,
  onSwitchConversation,
  onDeleteConversation,
  chunkCounts,
}: {
  conversations: Conversation[];
  activeId: string | null;
  onNewChat: () => void;
  onSwitchConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  chunkCounts: Record<string, number>;
}) {
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [indexedChunks, setIndexedChunks] = useState(0);
  const [kbCollapsed, setKbCollapsed] = useState(false);
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

  return (
    <aside className="sidebar" style={{ width: sidebarWidth }}>
      {/* Drag handle */}
      <div className="sidebar-resize-handle" onMouseDown={handleMouseDown} />
      <div className="sidebar-top">
        {/* New chat button */}
        <button type="button" className="sidebar-new-chat-btn" onClick={onNewChat}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New chat
        </button>

        {/* Chat history — flat list */}
        <div className="sidebar-section">
          <div className="sidebar-section-label">CHATS</div>
          <div className="sidebar-chat-list">
            {conversations.length > 0 ? (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`sidebar-chat-item${conv.id === activeId ? " active" : ""}`}
                  onClick={() => onSwitchConversation(conv.id)}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true" className="sidebar-chat-icon">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <div className="sidebar-chat-info">
                    <span className="sidebar-chat-title">{conv.title}</span>
                    <span className="sidebar-chat-time">{relativeTime(conv.updatedAt)}</span>
                  </div>
                  <button
                    type="button"
                    className="sidebar-chat-delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    title="Delete conversation"
                    aria-label="Delete conversation"
                  >
                    ×
                  </button>
                </div>
              ))
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
          <button
            type="button"
            className="sidebar-section-toggle"
            onClick={() => setKbCollapsed((v) => !v)}
          >
            <span className="sidebar-section-label">KNOWLEDGE BASE</span>
            <span className={`sidebar-toggle-chevron${kbCollapsed ? " collapsed" : ""}`}>▾</span>
          </button>
          {!kbCollapsed && (
            <div className="sidebar-kb-list">
              {sources.map((s) => (
                <KbSourceRow key={s.id} source={s} chunkCount={chunkCounts[s.id]} />
              ))}
            </div>
          )}
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
