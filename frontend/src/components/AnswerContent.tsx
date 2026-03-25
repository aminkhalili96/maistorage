import React, { ReactNode, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import type { Citation } from "../types";

function transformCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, '<span class="citation-ref" data-idx="$1">$1</span>');
}

/** Recursively extract text content from React element tree */
function extractText(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (typeof node === "object" && "props" in node) {
    return extractText(node.props.children);
  }
  return "";
}

const CodeBlockCopyButton = React.memo(function CodeBlockCopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable */ }
  };
  return (
    <button
      type="button"
      className="code-copy-btn"
      onClick={handleCopy}
      aria-label="Copy code"
    >
      {copied ? "✓ Copied" : "Copy"}
    </button>
  );
});

function CodeBlockWrapper({ children, ...props }: React.HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  const code = extractText(children);
  return (
    <div className="code-block-wrapper">
      <pre {...props}>{children}</pre>
      <CodeBlockCopyButton code={code} />
    </div>
  );
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
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        components={{ pre: CodeBlockWrapper }}
      >
        {transformCitations(text)}
      </ReactMarkdown>
      {isStreaming && <span className="streaming-cursor" />}
    </div>
  );
});
