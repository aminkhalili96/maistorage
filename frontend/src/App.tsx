import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { checkHealth, getIngestionStatus, streamChat } from "./api";
import type { ChatDonePayload, ChatTurn, Citation, TraceEvent } from "./types";
import { Sidebar } from "./components/Sidebar";
import { ThinkingBlock } from "./components/ThinkingBlock";
import { AnswerContent } from "./components/AnswerContent";
import { CopyButton, FeedbackButtons } from "./components/MessageActions";
import { SourceChips } from "./components/SourceChips";

const HISTORY_STORAGE_KEY = "maistorage-chat-history";


/* ================================================================
   Main App — 2-column layout (sidebar + chat)
   ================================================================ */

export default function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>(() => {
    try {
      const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!stored) return [];
      const parsed = JSON.parse(stored);
      if (!Array.isArray(parsed)) return [];
      return parsed as ChatTurn[];
    } catch {
      try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch { /* ignore */ }
      return [];
    }
  });
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

  const messagesRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const lastQuestionRef = useRef<string>("");

  // Persist history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // storage unavailable
    }
  }, [history]);

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

  function resetRunState() {
    setDraftAnswer("");
    setCitations([]);
    setTrace([]);
    setDonePayload(null);
    setThinkingSeconds(0);
  }

  const handleNewChat = useCallback(() => {
    setHistory([]);
    resetRunState();
    setError(null);
    try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch { /* ignore */ }
  }, []);

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
      resetRunState();
      setThinkingStartTime(Date.now());

      try {
        await streamChat(trimmed, nextHistory, {
          onTrace: (ev) => setTrace((cur) => [...cur, ev]),
          onCitation: (cit) =>
            setCitations((cur) =>
              cur.some((c) => c.chunk_id === cit.chunk_id) ? cur : [...cur, cit],
            ),
          onAnswerChunk: (text) => setDraftAnswer((cur) => cur + text),
          onDone: (payload: ChatDonePayload) => {
            const answer = String(payload.answer ?? "").trim();
            setDraftAnswer(answer);
            setDonePayload(payload);
            setHistory((cur) => [...cur, { role: "assistant", content: answer }]);
          },
          onError: (err) => setError(err.message),
        });
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Streaming failed.");
      } finally {
        setIsSending(false);
      }
    },
    [question, isSending, history],
  );

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
      </header>

      <div className="main-content">
        {/* -- Left sidebar -- */}
        <Sidebar history={history} onNewChat={handleNewChat} chunkCounts={chunkCounts} />

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
                          <div className="user-bubble">{turn.content}</div>
                        </div>
                      ) : (
                        <div className="assistant-message">
                          <div className="assistant-content">
                            {/* Persist trace block above the last completed answer */}
                            {!isSending && i === displayHistory.length - 1 && trace.length > 0 && (
                              <ThinkingBlock
                                trace={trace}
                                isProcessing={false}
                                thinkingSeconds={thinkingSeconds}
                              />
                            )}
                            <AnswerContent
                              text={turn.content}
                              isStreaming={false}
                              citations={i === displayHistory.length - 1 ? citations : undefined}
                              onCitationClick={undefined}
                            />
                            {/* Inline citations for completed messages */}
                            {!isSending && i === displayHistory.length - 1 && citations.length > 0 && (
                              <SourceChips
                                citations={citations}
                                donePayload={donePayload}
                                totalTimeSec={totalTimeSec}
                                chunkCounts={chunkCounts}
                              />
                            )}
                            {/* Trust badge fallback when no citations but has payload */}
                            {!isSending && donePayload && i === displayHistory.length - 1 && citations.length === 0 && (
                              <SourceChips
                                citations={[]}
                                donePayload={donePayload}
                                totalTimeSec={totalTimeSec}
                                chunkCounts={chunkCounts}
                              />
                            )}
                            <div className="answer-actions">
                              <CopyButton text={turn.content} />
                              <FeedbackButtons messageIndex={i} />
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
                <button
                  type="submit"
                  className={`send-button${hasInput && !isSending ? " active" : ""}`}
                  disabled={isSending || !hasInput}
                  aria-label="Send"
                >
                  {"\u2192"}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
