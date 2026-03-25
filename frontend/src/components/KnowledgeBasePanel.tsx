import { useEffect, useState } from "react";
import { getSources } from "../api";
import type { SourceRecord } from "../types";

/* ── Topic display config ── */
const FAMILY_LABELS: Record<string, string> = {
  core:                    "Core CUDA",
  distributed:             "Distributed Training",
  infrastructure:          "Infrastructure",
  advanced:                "Advanced Libraries",
  hardware:                "Hardware",
  "inference-optimization":"Inference Optimization",
  "inference-serving":     "Inference Serving",
  networking:              "Networking",
  "enterprise-platform":   "Enterprise Platform",
  "architecture-whitepaper": "Architecture",
};

const FAMILY_ORDER = [
  "hardware",
  "core",
  "advanced",
  "distributed",
  "networking",
  "infrastructure",
  "inference-optimization",
  "inference-serving",
  "enterprise-platform",
  "architecture-whitepaper",
];

function familyLabel(family: string): string {
  return FAMILY_LABELS[family] ?? family.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ── Inline PDF viewer modal ── */
function PdfModal({ source, onClose }: { source: SourceRecord; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="pdf-modal-overlay" onClick={onClose}>
      <div className="pdf-modal" onClick={(e) => e.stopPropagation()}>
        <div className="pdf-modal-header">
          <span className="pdf-modal-title">{source.title}</span>
          <div className="pdf-modal-actions">
            <a
              href={source.pdf_url!}
              target="_blank"
              rel="noopener noreferrer"
              className="pdf-modal-open-btn"
            >
              ↗ Open in new tab
            </a>
            <button className="pdf-modal-close" onClick={onClose} type="button">✕</button>
          </div>
        </div>
        <iframe
          src={source.pdf_url!}
          title={source.title}
          className="pdf-modal-iframe"
        />
      </div>
    </div>
  );
}

/* ── Source row ── */
function SourceRow({ source, onPdf }: { source: SourceRecord; onPdf: (s: SourceRecord) => void }) {
  const hasPdf = !!source.pdf_url;
  const docTypeLabel =
    source.doc_type === "pdf" ? "PDF"
    : source.doc_type === "html" ? "HTML"
    : source.doc_type === "markdown" ? "MD"
    : source.doc_type === "product" ? "Page"
    : source.doc_type.toUpperCase();

  return (
    <div className="kb-source-row">
      <div className="kb-source-main">
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="kb-source-title"
          title={source.title}
        >
          {source.title}
        </a>
        <span className="kb-doc-type">{docTypeLabel}</span>
      </div>
      {hasPdf && (
        <button
          className="kb-pdf-btn"
          type="button"
          onClick={() => onPdf(source)}
          title="View PDF"
        >
          <PdfIcon />
          <span>PDF</span>
        </button>
      )}
    </div>
  );
}

/* ── Topic group (collapsible) ── */
function FamilyGroup({
  family,
  sources,
  onPdf,
}: {
  family: string;
  sources: SourceRecord[];
  onPdf: (s: SourceRecord) => void;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="kb-family-group">
      <button
        className="kb-family-header"
        type="button"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="kb-family-chevron" style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}>›</span>
        <span className="kb-family-label">{familyLabel(family)}</span>
        <span className="kb-family-count">{sources.length}</span>
      </button>
      {open && (
        <div className="kb-family-sources">
          {sources.map((s) => (
            <SourceRow key={s.id} source={s} onPdf={onPdf} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Main panel ── */
export function KnowledgeBasePanel({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [pdfSource, setPdfSource] = useState<SourceRecord | null>(null);

  useEffect(() => {
    getSources()
      .then((r) => setSources(r.sources))
      .catch(() => {});
  }, []);

  /* Group by doc_family */
  const byFamily = sources.reduce<Record<string, SourceRecord[]>>((acc, s) => {
    (acc[s.doc_family] ??= []).push(s);
    return acc;
  }, {});

  const orderedFamilies = [
    ...FAMILY_ORDER.filter((f) => byFamily[f]),
    ...Object.keys(byFamily).filter((f) => !FAMILY_ORDER.includes(f)),
  ];

  return (
    <>
      {/* Toggle tab — always visible */}
      <button
        className={`kb-panel-tab${open ? " open" : ""}`}
        type="button"
        onClick={onToggle}
        title={open ? "Close knowledge base panel" : "Browse knowledge base"}
      >
        <DocsIcon />
        {!open && <span className="kb-tab-label">Knowledge Base</span>}
      </button>

      {/* Side panel */}
      <div className={`kb-panel${open ? " open" : ""}`}>
        <div className="kb-panel-header">
          <span className="kb-panel-title">Knowledge Base</span>
          <span className="kb-panel-subtitle">{sources.length} sources</span>
        </div>
        <div className="kb-panel-scroll">
          {orderedFamilies.map((family) => (
            <FamilyGroup
              key={family}
              family={family}
              sources={byFamily[family]}
              onPdf={setPdfSource}
            />
          ))}
        </div>
      </div>

      {pdfSource && (
        <PdfModal source={pdfSource} onClose={() => setPdfSource(null)} />
      )}
    </>
  );
}

/* ── Icons ── */
function DocsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
    </svg>
  );
}

function PdfIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
      <path d="M14 2v6h6" />
    </svg>
  );
}
