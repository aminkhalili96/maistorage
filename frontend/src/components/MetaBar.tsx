import React from "react";
import type { ChatDonePayload } from "../types";

/* ================================================================
   MetaBar — simplified, trust badge is now in SourceChips.
   Shows only grounding/quality/degraded status when relevant.
   ================================================================ */

function trustBadgeClass(mode: string): string {
  if (mode === "knowledge-base-backed") return "trust-badge knowledge-base";
  if (mode === "web-backed") return "trust-badge web";
  if (mode === "insufficient-evidence") return "trust-badge insufficient";
  if (mode === "llm-knowledge") return "trust-badge llm";
  return "";
}

function trustBadgeLabel(mode: string): string {
  if (mode === "knowledge-base-backed") return "Knowledge-base-backed";
  if (mode === "web-backed") return "Web-backed";
  if (mode === "insufficient-evidence") return "Insufficient evidence";
  if (mode === "llm-knowledge") return "LLM knowledge";
  if (mode === "direct-chat") return "Direct chat";
  return "";
}

function trustBadgeTooltip(mode: string): string {
  if (mode === "knowledge-base-backed") return "Answer sourced from NVIDIA knowledge base";
  if (mode === "web-backed") return "Answer sourced from live web search";
  if (mode === "llm-knowledge") return "Answer from AI general knowledge (not grounded in docs)";
  if (mode === "insufficient-evidence") return "Could not find sufficient evidence to answer";
  if (mode === "direct-chat") return "Conversational response (no retrieval needed)";
  return "";
}

export const MetaBar = React.memo(function MetaBar({ payload }: { payload: ChatDonePayload | null }) {
  if (!payload || payload.response_mode === "direct-chat") return null;
  const badgeClass = trustBadgeClass(payload.response_mode);
  const badgeLabel = trustBadgeLabel(payload.response_mode);
  const showConfidence =
    payload.confidence > 0 &&
    (payload.response_mode === "knowledge-base-backed" || payload.response_mode === "web-backed");
  return (
    <div className="meta-bar">
      {badgeClass && <span className={badgeClass} title={trustBadgeTooltip(payload.response_mode)}>{badgeLabel}</span>}
      {showConfidence && (
        <span className="meta-chip confidence-chip">
          {Math.round(payload.confidence * 100)}% confidence
        </span>
      )}
      {payload.grounding_passed != null && (
        <span className={`meta-chip ${payload.grounding_passed ? "meta-chip-ok" : "meta-chip-fail"}`}>
          {payload.grounding_passed ? "\u2713" : "\u2717"} grounding
        </span>
      )}
      {payload.answer_quality_passed != null && (
        <span
          className={`meta-chip ${payload.answer_quality_passed ? "meta-chip-ok" : "meta-chip-warn"}`}
        >
          {payload.answer_quality_passed ? "\u2713" : "\u26A0"} quality
        </span>
      )}
      {payload.rejected_chunk_count > 0 && (
        <span className="meta-chip meta-chip-warn">
          {payload.rejected_chunk_count} rejected
        </span>
      )}
      {payload.generation_degraded && (
        <span className="meta-chip meta-chip-warn" title="LLM unavailable; answer from keyword fallback">
          \u26A0 degraded
        </span>
      )}
    </div>
  );
});
