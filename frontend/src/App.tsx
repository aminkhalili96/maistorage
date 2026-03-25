import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { checkHealth, getIngestionStatus, streamChat } from "./api";
import type { ChatDonePayload, ChatTurn, Citation, TraceEvent } from "./types";
import { useConversations } from "./hooks/useConversations";
import { Sidebar } from "./components/Sidebar";
import { ThinkingBlock } from "./components/ThinkingBlock";
import { AnswerContent } from "./components/AnswerContent";
import { CopyButton, FeedbackButtons, RegenerateButton } from "./components/MessageActions";
import { SourceChips } from "./components/SourceChips";


/* ================================================================
   Main App — 2-column layout (sidebar + chat)

   Architecture: App.tsx is the state owner. All 10 child components are
   React.memo'd and receive only the props they need. Key state flows:

   User types question → handleSubmit() → streamChat() SSE →
     onTrace: updates trace[] → ThinkingBlock renders steps in real-time
     onAnswerChunk: appends to draftAnswer → AnswerContent renders incrementally
     onCitation: appends to citations[] → CitationsPanel shows clickable chips
     onDone: receives final payload → MetaBar shows trust badge + quality signals

   Multi-conversation: useConversations hook manages localStorage persistence,
   CRUD, and conversation switching with proper state cleanup.
   ================================================================ */

export default function App() {
  const [question, setQuestion] = useState("");
  const {
    conversations,
    activeId,
    history,
    setHistory,
    newChat,
    switchConversation,
    deleteConversation,
  } = useConversations();
  const [draftAnswer, setDraftAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendHealthy, setBackendHealthy] = useState(true);
  const [thinkingStartTime, setThinkingStartTime] = useState(0);
  const [thinkingSeconds, setThinkingSeconds] = useState(0);
  const [donePayload, setDonePayload] = useState<ChatDonePayload | null>(null);
  const [chunkCounts, setChunkCounts] = useState<Record<string, number>>({});

  // F5: Edit & Resend state
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  const messagesRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const lastQuestionRef = useRef<string>("");
  const abortRef = useRef<AbortController | null>(null); // F3: Stop generation
  // Refs to capture latest accumulated values for onDone closure
  const citationsRef = useRef<Citation[]>([]);
  const traceRef = useRef<TraceEvent[]>([]);
  // Reset run state on explicit conversation switch (called from handlers, not from useEffect)
  const resetForConversationSwitch = useCallback(() => {
    resetRunState();
    setError(null);
    setEditingIndex(null);
    setEditText("");
  }, []);

  // Auto-scroll on new content
  useEffect(() => {
    const node = messagesRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [history, draftAnswer, trace.length, isSending]);

  // Auto-focus textarea on load
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Health check
  useEffect(() => {
    checkHealth().then(setBackendHealthy);
  }, []);

  // Fetch chunk counts for source panel display
  useEffect(() => {
    getIngestionStatus()
      .then((status) => {
        if (status.chunk_counts) setChunkCounts(status.chunk_counts);
      })
      .catch(() => {});
  }, []);

  // Elapsed thinking timer
  useEffect(() => {
    if (!isSending) {
      setThinkingSeconds(Math.round((Date.now() - thinkingStartTime) / 1000));
      return;
    }
    const interval = setInterval(() => {
      setThinkingSeconds(Math.round((Date.now() - thinkingStartTime) / 1000));
    }, 500);
    return () => clearInterval(interval);
  }, [isSending, thinkingStartTime]);

  // F6: Keyboard shortcuts
  useEffect(() => {
    const handler = (e: globalThis.KeyboardEvent) => {
      // Escape: stop generation or cancel edit
      if (e.key === "Escape") {
        if (isSending) {
          handleStop();
          return;
        }
        if (editingIndex !== null) {
          handleCancelEdit();
          return;
        }
      }
      // Ctrl+Shift+N: new chat
      if (e.ctrlKey && e.shiftKey && e.key === "N") {
        e.preventDefault();
        handleNewChat();
        return;
      }
      // Ctrl+Shift+E: export
      if (e.ctrlKey && e.shiftKey && e.key === "E") {
        e.preventDefault();
        if (history.length > 0) handleExport();
        return;
      }
      // "/": focus textarea (if not already in a text field)
      if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const active = document.activeElement;
        if (
          active instanceof HTMLTextAreaElement ||
          active instanceof HTMLInputElement ||
          (active as HTMLElement)?.isContentEditable
        ) {
          return;
        }
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isSending, editingIndex, history.length]);

  function resetRunState() {
    setDraftAnswer("");
    setCitations([]);
    setTrace([]);
    setDonePayload(null);
    setThinkingSeconds(0);
    citationsRef.current = [];
    traceRef.current = [];
  }

  const handleNewChat = useCallback(() => {
    newChat();
    resetForConversationSwitch();
  }, [newChat, resetForConversationSwitch]);

  const handleSwitchConversation = useCallback((id: string) => {
    switchConversation(id);
    resetForConversationSwitch();
  }, [switchConversation, resetForConversationSwitch]);

  const handleSubmit = useCallback(
    async (questionOverride?: string) => {
      const trimmed = (questionOverride ?? question).trim();
      if (!trimmed || isSending) return;

      lastQuestionRef.current = trimmed;
      const userTurn: ChatTurn = { role: "user", content: trimmed };
      const nextHistory = [...history, userTurn];
      setHistory(nextHistory);
      setQuestion("");
      setError(null);
      setIsSending(true);
      setEditingIndex(null);
      setEditText("");
      resetRunState();
      setThinkingStartTime(Date.now());

      // F3: Create abort controller
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat(trimmed, nextHistory, {
          onTrace: (ev) => {
            traceRef.current = [...traceRef.current, ev];
            setTrace(traceRef.current);
          },
          onCitation: (cit) => {
            if (!citationsRef.current.some((c) => c.chunk_id === cit.chunk_id)) {
              citationsRef.current = [...citationsRef.current, cit];
            }
            setCitations(citationsRef.current);
          },
          onAnswerChunk: (text) => setDraftAnswer((cur) => cur + text),
          onDone: (payload: ChatDonePayload) => {
            const answer = String(payload.answer ?? "").trim();
            setDraftAnswer(answer);
            setDonePayload(payload);
            setHistory((cur) => [...cur, {
              role: "assistant",
              content: answer,
              citations: citationsRef.current,
              trace: traceRef.current,
              donePayload: payload,
            }]);
          },
          onError: (err) => setError(err.message),
        }, controller);
      } catch (cause) {
        // F3: Don't show error if user aborted
        if (controller.signal.aborted) {
          // keep partial answer — already committed by handleStop
        } else {
          setError(cause instanceof Error ? cause.message : "Streaming failed.");
        }
      } finally {
        setIsSending(false);
        abortRef.current = null;
      }
    },
    [question, isSending, history, setHistory],
  );

  // F3: Stop generation
  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    // Commit partial answer to history
    setDraftAnswer((current) => {
      if (current.trim()) {
        setHistory((prev) => [...prev, { role: "assistant", content: current.trim() }]);
      }
      return current;
    });
    setIsSending(false);
    abortRef.current = null;
  }, [setHistory]);

  // F1: Regenerate response
  const handleRegenerate = useCallback(() => {
    if (isSending || !lastQuestionRef.current) return;
    // Remove last assistant turn
    setHistory((prev) => {
      const lastAssistantIdx = prev.length - 1;
      if (lastAssistantIdx >= 0 && prev[lastAssistantIdx].role === "assistant") {
        return prev.slice(0, lastAssistantIdx);
      }
      return prev;
    });
    resetRunState();
    // Re-submit with the last question after state update
    setTimeout(() => void handleSubmit(lastQuestionRef.current), 0);
  }, [isSending, handleSubmit, setHistory]);

  // F4: Export conversation as markdown
  const handleExport = useCallback(() => {
    if (history.length === 0) return;
    const date = new Date().toISOString().slice(0, 10);
    const turns = history.length;
    const lines: string[] = [
      `# MaiSearch Conversation`,
      `**Date:** ${date} | **Turns:** ${turns}`,
      "",
    ];
    for (const turn of history) {
      lines.push(turn.role === "user" ? "## User" : "## Assistant");
      lines.push("");
      lines.push(turn.content);
      lines.push("");
      lines.push("---");
      lines.push("");
    }
    const md = lines.join("\n");
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const slug = (history.find((t) => t.role === "user")?.content ?? "chat")
      .slice(0, 30)
      .replace(/[^a-zA-Z0-9]+/g, "-")
      .toLowerCase();
    const a = document.createElement("a");
    a.href = url;
    a.download = `maisearch-${date}-${slug}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [history]);

  // F5: Edit & Resend handlers
  const handleStartEdit = useCallback((index: number, content: string) => {
    setEditingIndex(index);
    setEditText(content);
  }, []);

  const handleCancelEdit = useCallback(() => {
    setEditingIndex(null);
    setEditText("");
  }, []);

  const handleSubmitEdit = useCallback(() => {
    if (editingIndex === null || !editText.trim()) return;
    // Truncate history to the editing index (discard from N onward)
    setHistory((prev) => prev.slice(0, editingIndex));
    setEditingIndex(null);
    setEditText("");
    resetRunState();
    setTimeout(() => void handleSubmit(editText.trim()), 0);
  }, [editingIndex, editText, handleSubmit, setHistory]);

  const handleFormSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      void handleSubmit();
    },
    [handleSubmit],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleTextareaInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, []);

  // Avoid showing the in-progress answer twice in history
  const displayHistory =
    isSending &&
    draftAnswer &&
    history.at(-1)?.role === "assistant" &&
    history.at(-1)?.content === draftAnswer
      ? history.slice(0, -1)
      : history;

  // Compute total pipeline time from trace timestamps
  const totalTimeSec = (() => {
    const timestamps = trace.map((ev) => ev.timestamp).filter((t): t is number => t != null);
    if (timestamps.length < 2) return null;
    return Math.round((timestamps[timestamps.length - 1] - timestamps[0]) * 10) / 10;
  })();

  const hasConversation =
    displayHistory.length > 0 ||
    trace.length > 0 ||
    draftAnswer.length > 0 ||
    isSending;
  const isThinking = isSending && !draftAnswer;
  const isStreaming = isSending && !!draftAnswer;
  const hasInput = question.trim().length > 0;

  return (
    <div className="app">
      {!backendHealthy && (
        <div className="health-banner">
          Backend unreachable — check that the server is running on port 8000
        </div>
      )}
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button
            type="button"
            className="retry-btn"
            onClick={() => {
              setError(null);
              if (lastQuestionRef.current) void handleSubmit(lastQuestionRef.current);
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* -- Header -- */}
      <header className="app-header">
        <span className="header-title">MaiSearch</span>
        {history.length > 0 && (
          <button
            type="button"
            className="header-action-btn"
            onClick={handleExport}
            title="Export conversation as Markdown (Ctrl+Shift+E)"
            aria-label="Export conversation"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export
          </button>
        )}
      </header>

      <div className="main-content">
        {/* -- Left sidebar -- */}
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onNewChat={handleNewChat}
          onSwitchConversation={handleSwitchConversation}
          onDeleteConversation={deleteConversation}
          chunkCounts={chunkCounts}
        />

        {/* -- Conversation area -- */}
        <div className="conversation-column">
          <div className="messages-area" ref={messagesRef}>
            {!hasConversation ? (
              /* Empty state */
              <div className="empty-state">
                <div className="empty-state-title">How can I help you today?</div>
                <div className="suggestion-chips">
                  {[
                    "Why is 4-GPU training scaling poorly?",
                    "Compare H100 vs A100 for inference",
                    "How do I tune NCCL for max bandwidth?",
                  ].map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="suggestion-chip"
                      onClick={() => void handleSubmit(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <AnimatePresence initial={false}>
                  {displayHistory.map((turn, i) => (
                    <motion.div
                      key={`${turn.role}-${i}-${turn.content.slice(0, 32)}`}
                      className="message-wrapper"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.25 }}
                    >
                      {turn.role === "user" ? (
                        <div className="user-message">
                          {editingIndex === i ? (
                            /* F5: Edit mode */
                            <div className="user-edit-container">
                              <textarea
                                className="user-edit-textarea"
                                value={editText}
                                onChange={(e) => setEditText(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmitEdit();
                                  }
                                  if (e.key === "Escape") handleCancelEdit();
                                }}
                                rows={3}
                                autoFocus
                              />
                              <div className="user-edit-actions">
                                <button type="button" className="user-edit-send" onClick={handleSubmitEdit}>
                                  Send
                                </button>
                                <button type="button" className="user-edit-cancel" onClick={handleCancelEdit}>
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            /* F5: Normal user bubble with hover edit icon */
                            <div className="user-bubble-wrapper">
                              <div className="user-bubble">{turn.content}</div>
                              {!isSending && (
                                <button
                                  type="button"
                                  className="user-edit-icon"
                                  onClick={() => handleStartEdit(i, turn.content)}
                                  title="Edit message"
                                  aria-label="Edit message"
                                >
                                  ✎
                                </button>
                              )}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="assistant-message">
                          <div className="assistant-content">
                            {/* Per-message trace block */}
                            {!isSending && (turn.trace ?? []).length > 0 && (
                              <ThinkingBlock
                                trace={turn.trace!}
                                isProcessing={false}
                                thinkingSeconds={thinkingSeconds}
                              />
                            )}
                            <AnswerContent
                              text={turn.content}
                              isStreaming={false}
                              citations={(turn.citations ?? []).length > 0 ? turn.citations : undefined}
                              onCitationClick={undefined}
                            />
                            {/* Per-message source chips */}
                            {!isSending && (turn.citations ?? []).length > 0 && (
                              <SourceChips
                                citations={turn.citations!}
                                donePayload={turn.donePayload ?? null}
                                totalTimeSec={(() => {
                                  const ts = (turn.trace ?? []).map((e) => e.timestamp).filter((t): t is number => t != null);
                                  return ts.length >= 2 ? Math.round((ts[ts.length - 1] - ts[0]) * 10) / 10 : null;
                                })()}
                                chunkCounts={chunkCounts}
                              />
                            )}
                            {/* Trust badge fallback when no citations but has payload */}
                            {!isSending && turn.donePayload && (turn.citations ?? []).length === 0 && (
                              <SourceChips
                                citations={[]}
                                donePayload={turn.donePayload}
                                totalTimeSec={null}
                                chunkCounts={chunkCounts}
                              />
                            )}
                            <div className="answer-actions">
                              <CopyButton text={turn.content} />
                              <FeedbackButtons messageIndex={i} conversationId={activeId ?? undefined} />
                              {/* F1: Regenerate on last assistant message */}
                              {!isSending && i === displayHistory.length - 1 && (
                                <RegenerateButton onClick={handleRegenerate} disabled={isSending} />
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {/* -- Live response -- */}
                {(isThinking || isStreaming || (draftAnswer && isSending)) && (
                  <motion.div
                    className="message-wrapper"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <div className="assistant-message">
                      <div className="assistant-content">
                        {(trace.length > 0 || isThinking) && (
                          <ThinkingBlock
                            trace={trace}
                            isProcessing={isThinking}
                            thinkingSeconds={thinkingSeconds}
                          />
                        )}
                        {draftAnswer && (
                          <AnswerContent
                            text={draftAnswer}
                            isStreaming={isStreaming}
                            citations={citations}
                            onCitationClick={undefined}
                          />
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </div>

          {/* -- Composer -- */}
          <div className="composer-area">
            <div className="composer-inner">
              <form className="composer-box" onSubmit={handleFormSubmit}>
                <textarea
                  ref={textareaRef}
                  className="composer-textarea"
                  value={question}
                  onChange={(e) => {
                    setQuestion(e.target.value);
                    handleTextareaInput();
                  }}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={isSending}
                />
                {isSending ? (
                  /* F3: Stop button while generating */
                  <button
                    type="button"
                    className="stop-button"
                    onClick={handleStop}
                    aria-label="Stop generation"
                    title="Stop generation (Escape)"
                  >
                    ■
                  </button>
                ) : (
                  <button
                    type="submit"
                    className={`send-button${hasInput && !isSending ? " active" : ""}`}
                    disabled={isSending || !hasInput}
                    aria-label="Send"
                  >
                    {"\u2192"}
                  </button>
                )}
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
