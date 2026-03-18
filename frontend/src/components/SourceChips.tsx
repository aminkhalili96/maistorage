import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { ChatDonePayload, Citation } from "../types";

/* ================================================================
   Trust badge helpers
   ================================================================ */

function trustBadgeClass(mode: string): string {
  if (mode === "corpus-backed") return "source-trust-pill corpus";
  if (mode === "web-backed") return "source-trust-pill web";
  if (mode === "insufficient-evidence") return "source-trust-pill insufficient";
  if (mode === "llm-knowledge") return "source-trust-pill llm";
  return "";
}

function trustBadgeLabel(mode: string): string {
  if (mode === "corpus-backed") return "corpus-backed";
  if (mode === "web-backed") return "web-backed";
  if (mode === "insufficient-evidence") return "insufficient evidence";
  if (mode === "llm-knowledge") return "llm knowledge";
  if (mode === "direct-chat") return "direct chat";
  return "";
}

function trustBadgeTooltip(mode: string): string {
  if (mode === "corpus-backed") return "Answer sourced from NVIDIA documentation corpus";
  if (mode === "web-backed") return "Answer sourced from live web search";
  if (mode === "llm-knowledge") return "Answer from AI general knowledge (not grounded in docs)";
  if (mode === "insufficient-evidence") return "Could not find sufficient evidence to answer";
  if (mode === "direct-chat") return "Conversational response (no retrieval needed)";
  return "";
}

/* ================================================================
   Snippet with keyword highlighting
   ================================================================ */

const HIGHLIGHT_TERMS = [
  "NVLink", "NVSwitch", "InfiniBand", "GPUDirect", "RDMA", "PCIe",
  "SXM5", "SXM4", "HBM3", "HBM3e", "HBM2e", "DGX", "HGX", "SuperPOD", "BasePOD",
  "ConnectX-7", "ConnectX-6", "NDR", "HDR",
  "TFLOPS", "TOPS",
];

function HighlightedSnippet({ text }: { text: string }) {
  // Build a regex that matches any of the highlight terms or numeric specs
  const termPattern = HIGHLIGHT_TERMS.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const regex = new RegExp(
    `(\\b\\d+x\\s+\\S+\\s+\\S+\\s+GPUs?\\b|\\b\\d+\\s*(?:GB\\/s|Gb\\/s|TB\\/s|GiB|GB|MHz|GHz)\\b|${termPattern})`,
    "gi"
  );

  const parts = text.split(regex);
  return (
    <p className="source-panel-snippet">
      {parts.map((part, i) =>
        regex.test(part) ? (
          <span key={i} className="source-highlight">{part}</span>
        ) : (
          <React.Fragment key={i}>{part}</React.Fragment>
        )
      )}
    </p>
  );
}

/* ================================================================
   SourceChips — inline citations below answers
   ================================================================ */

export const SourceChips = React.memo(function SourceChips({
  citations,
  donePayload,
  totalTimeSec,
  chunkCounts,
}: {
  citations: Citation[];
  donePayload: ChatDonePayload | null;
  totalTimeSec?: number | null;
  chunkCounts?: Record<string, number>;
}) {
  const [expandedChipId, setExpandedChipId] = useState<string | null>(null);

  if (citations.length === 0 && !donePayload) return null;
  if (donePayload && donePayload.response_mode === "direct-chat") return null;

  const handleChipClick = (chunkId: string) => {
    setExpandedChipId((cur) => (cur === chunkId ? null : chunkId));
  };

  const expandedCitation = expandedChipId
    ? citations.find((c) => c.chunk_id === expandedChipId) ?? null
    : null;

  const expandedIndex = expandedCitation
    ? citations.findIndex((c) => c.chunk_id === expandedCitation.chunk_id)
    : -1;

  function chipLabel(cit: Citation, idx: number): string {
    const title = cit.title.length > 24 ? cit.title.slice(0, 22) + "..." : cit.title;
    const chunkMatch = cit.chunk_id.match(/chunk[_-]?(\d+)/i);
    const chunkNum = chunkMatch ? chunkMatch[1] : String(idx + 1);
    return `${title}, chunk ${chunkNum}`;
  }

  return (
    <div className="source-chips-container">
      {/* Source chips row */}
      {citations.length > 0 && (
        <div className="source-chips-row">
          <span className="source-chips-label">Sources</span>
          <div className="source-chips-list">
            {citations.map((cit, idx) => {
              const isActive = expandedChipId === cit.chunk_id;
              return (
                <button
                  key={cit.chunk_id}
                  type="button"
                  className={`source-chip${isActive ? " active" : ""}`}
                  onClick={() => handleChipClick(cit.chunk_id)}
                  title={cit.title}
                >
                  <span className="source-chip-title">{chipLabel(cit, idx)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Expanded source panel */}
      <AnimatePresence>
        {expandedCitation && (
          <motion.div
            key={expandedCitation.chunk_id}
            className="source-panel"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            <div className="source-panel-inner">
              <div className="source-panel-header">
                <div className="source-panel-header-left">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
                    <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
                  </svg>
                  <span className="source-panel-filename">{expandedCitation.title}</span>
                </div>
                <span className="source-panel-chunk-label">
                  {(() => {
                    const chunkMatch = expandedCitation.chunk_id.match(/chunk[_-]?(\d+)/i);
                    const chunkNum = chunkMatch ? chunkMatch[1] : String(expandedIndex + 1);
                    const total = expandedCitation.source_id && chunkCounts?.[expandedCitation.source_id];
                    return total ? `chunk ${chunkNum} of ${total}` : `chunk ${chunkNum}`;
                  })()}
                </span>
              </div>
              <div className="source-panel-body">
                <HighlightedSnippet text={expandedCitation.snippet} />
              </div>
              <div className="source-panel-footer">
                {expandedCitation.score != null && (
                  <span className="source-panel-meta">
                    Similarity: {expandedCitation.score.toFixed(2)}
                  </span>
                )}
                {expandedCitation.char_count != null && (
                  <span className="source-panel-meta">
                    {expandedCitation.char_count.toLocaleString()} chars
                  </span>
                )}
                <span className="source-panel-meta">
                  {expandedCitation.page != null ? `Page ${expandedCitation.page}` : expandedCitation.domain}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Trust badge row */}
      {donePayload && donePayload.response_mode !== "direct-chat" && (
        <div className="source-trust-row">
          {trustBadgeClass(donePayload.response_mode) && (
            <span
              className={trustBadgeClass(donePayload.response_mode)}
              title={trustBadgeTooltip(donePayload.response_mode)}
            >
              {trustBadgeLabel(donePayload.response_mode)}
            </span>
          )}
          <span className="source-trust-meta">
            {[
              citations.length > 0 ? `${citations.length} source${citations.length !== 1 ? "s" : ""}` : null,
              totalTimeSec != null ? `${totalTimeSec}s` : null,
            ].filter(Boolean).join(", ")}
          </span>
        </div>
      )}
    </div>
  );
});
