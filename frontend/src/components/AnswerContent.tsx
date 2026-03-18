import React, { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import type { Citation } from "../types";

function transformCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, '<span class="citation-ref" data-idx="$1">$1</span>');
}

export const AnswerContent = React.memo(function AnswerContent({
  text,
  isStreaming,
  citations,
  onCitationClick,
}: {
  text: string;
  isStreaming: boolean;
  citations?: Citation[];
  onCitationClick?: (chunkId: string) => void;
}) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!citations?.length || !onCitationClick) return;
    const container = contentRef.current;
    if (!container) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains("citation-ref")) {
        const idx = parseInt(target.dataset.idx ?? "0", 10) - 1;
        const cit = citations[idx];
        if (cit) {
          onCitationClick(cit.chunk_id);
          setTimeout(() => {
            document.querySelector(".citation-card.selected")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 50);
        }
      }
    };
    container.addEventListener("click", handler);
    return () => container.removeEventListener("click", handler);
  }, [citations, onCitationClick]);

  return (
    <div className="answer-content" ref={contentRef}>
      <ReactMarkdown rehypePlugins={[rehypeHighlight, rehypeRaw]}>
        {transformCitations(text)}
      </ReactMarkdown>
      {isStreaming && <span className="streaming-cursor" />}
    </div>
  );
});
