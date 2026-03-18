import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Citation } from "../types";

const CitationCard = React.memo(function CitationCard({
  citation,
  index,
  selected,
  onClick,
}: {
  citation: Citation;
  index: number;
  selected: boolean;
  onClick: () => void;
}) {
  const isWeb = citation.source_kind === "web";
  return (
    <button
      type="button"
      className={`citation-card${selected ? " selected" : ""}`}
      onClick={onClick}
    >
      <div className="citation-card-top">
        <div className="citation-num">{index + 1}</div>
        <span className={`citation-kind-badge ${isWeb ? "web" : "corpus"}`}>
          {isWeb ? "Web" : "Docs"}
        </span>
      </div>
      <div className="citation-card-title">{citation.title}</div>
      <div className="citation-card-snippet">
        {citation.snippet.length > 140
          ? citation.snippet.slice(0, 140) + "…"
          : citation.snippet}
      </div>
      <div className="citation-card-source">
        {citation.domain || "NVIDIA Docs"}
      </div>
    </button>
  );
});

export const CitationsPanel = React.memo(function CitationsPanel({
  citations,
  selectedCitationId,
  onSelectCitation,
}: {
  citations: Citation[];
  selectedCitationId: string | null;
  onSelectCitation: (chunkId: string) => void;
}) {
  if (citations.length === 0) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="sources-sidebar"
        initial={{ x: 60, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 60, opacity: 0 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
      >
        <div className="sources-header">
          <span className="sources-title">Sources</span>
          <span className="sources-count">{citations.length}</span>
        </div>
        <div className="sources-list">
          {citations.map((cit, idx) => (
            <CitationCard
              key={cit.chunk_id}
              citation={cit}
              index={idx}
              selected={selectedCitationId === cit.chunk_id}
              onClick={() => onSelectCitation(cit.chunk_id)}
            />
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
});
