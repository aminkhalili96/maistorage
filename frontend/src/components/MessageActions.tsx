import React, { useState } from "react";

const FEEDBACK_STORAGE_KEY = "maistorage-feedback";

export const CopyButton = React.memo(function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable
    }
  };

  return (
    <button
      type="button"
      className="copy-btn"
      onClick={handleCopy}
      title="Copy to clipboard"
      aria-label="Copy answer"
    >
      {copied ? "✓ Copied" : "⎘ Copy"}
    </button>
  );
});

export const RegenerateButton = React.memo(function RegenerateButton({
  onClick,
  disabled,
}: {
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      className="copy-btn regenerate-btn"
      onClick={onClick}
      disabled={disabled}
      title="Regenerate response"
      aria-label="Regenerate response"
    >
      ↻ Regenerate
    </button>
  );
});

export const FeedbackButtons = React.memo(function FeedbackButtons({
  messageIndex,
  conversationId,
}: {
  messageIndex: number;
  conversationId?: string;
}) {
  const storageKey = conversationId
    ? `maistorage-feedback-${conversationId}-${messageIndex}`
    : `${FEEDBACK_STORAGE_KEY}-${messageIndex}`;
  const [feedback, setFeedback] = useState<"up" | "down" | null>(() => {
    try {
      return (localStorage.getItem(storageKey) as "up" | "down" | null) ?? null;
    } catch {
      return null;
    }
  });

  const handleFeedback = (value: "up" | "down") => {
    const next = feedback === value ? null : value;
    setFeedback(next);
    try {
      if (next) localStorage.setItem(storageKey, next);
      else localStorage.removeItem(storageKey);
    } catch {
      // storage unavailable
    }
  };

  return (
    <div className="feedback-buttons">
      <button
        type="button"
        className={`feedback-btn${feedback === "up" ? " active-up" : ""}`}
        onClick={() => handleFeedback("up")}
        title="Helpful"
        aria-label="Mark as helpful"
      >
        👍
      </button>
      <button
        type="button"
        className={`feedback-btn${feedback === "down" ? " active-down" : ""}`}
        onClick={() => handleFeedback("down")}
        title="Not helpful"
        aria-label="Mark as not helpful"
      >
        👎
      </button>
    </div>
  );
});
