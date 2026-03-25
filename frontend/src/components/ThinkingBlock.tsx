import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { TraceEvent } from "../types";

/* ================================================================
   Helpers — map trace events to 4-step display
   ================================================================ */

type TraceStep = {
  id: string;
  label: string;
  detail: string | React.ReactNode;
  color: string;
  durationMs: number | null;
};

const STEP_COLORS = {
  retrieve: "#378ADD",
  grade: "#BA7517",
  generate: "#7F77DD",
  hallucination: "#BA7517",
} as const;

function GradePills({ grades, keptCount, totalCount }: {
  grades: string[];
  keptCount: number;
  totalCount: number;
}) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
      {grades.map((g, i) => (
        <span key={i} className={`grade-pill ${g === "pass" ? "pass" : "fail"}`}>
          {g === "pass" ? "PASS" : "FAIL"}
        </span>
      ))}
      <span style={{ marginLeft: 4, color: "var(--text-tertiary)", fontSize: 12 }}>
        {keptCount}/{totalCount} relevant
      </span>
    </span>
  );
}

function GroundedBadge({ passed }: { passed: boolean }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span className={passed ? "grounded-pill" : "not-grounded-pill"}>
        {passed ? "GROUNDED" : "NOT GROUNDED"}
      </span>
      <span style={{ color: "var(--text-tertiary)", fontSize: 12 }}>
        {passed ? "Answer supported by sources" : "Answer not supported by sources"}
      </span>
    </span>
  );
}

function buildSteps(trace: TraceEvent[]): TraceStep[] {
  const steps: TraceStep[] = [];

  for (let i = 0; i < trace.length; i++) {
    const ev = trace[i];
    const prevTs = i > 0 ? trace[i - 1].timestamp : undefined;
    const curTs = ev.timestamp;
    let durationMs: number | null = null;
    if (prevTs != null && curTs != null) {
      durationMs = Math.round((curTs - prevTs) * 1000);
    }

    switch (ev.type) {
      case "retrieve":
      case "retrieval": {
        const count =
          (ev as any).result_count ??
          (ev as any).chunks_retrieved ??
          (ev.payload as any)?.result_count ??
          (ev.payload as any)?.chunks_retrieved ??
          null;
        const confidence =
          (ev as any).confidence ??
          (ev.payload as any)?.confidence ??
          null;
        const confStr = confidence != null ? ` (confidence ${Number(confidence).toFixed(2)})` : "";
        const detail = count != null
          ? `Knowledge base query, ${count} chunks returned${confStr}`
          : "Searching NVIDIA documentation";
        steps.push({
          id: "retrieve",
          label: "Retrieve",
          detail,
          color: STEP_COLORS.retrieve,
          durationMs,
        });
        break;
      }
      case "rerank": {
        const count = (ev.payload as any)?.retrieved_total ?? null;
        const confidence = (ev.payload as any)?.confidence ?? null;
        const confStr = confidence != null ? ` (confidence ${Number(confidence).toFixed(2)})` : "";
        const detail = count != null
          ? `Knowledge base query, ${count} chunks returned${confStr}`
          : "Reranked results";
        steps.push({
          id: "retrieve",   // same id → dedup keeps this last one
          label: "Retrieve",
          detail,
          color: STEP_COLORS.retrieve,
          durationMs,
        });
        break;
      }
      case "document_grading": {
        const totalCount =
          (ev as any).total_count ?? (ev.payload as any)?.total_count ?? null;
        const keptCount =
          (ev as any).kept_count ?? (ev.payload as any)?.kept_count ?? null;
        const grades: string[] =
          (ev as any).grades ?? (ev.payload as any)?.grades ?? [];
        let detail: string | React.ReactNode;
        if (totalCount != null && keptCount != null && grades.length > 0) {
          detail = <GradePills grades={grades} keptCount={keptCount} totalCount={totalCount} />;
        } else if (totalCount != null && keptCount != null) {
          detail = `${keptCount}/${totalCount} relevant`;
        } else {
          detail = "Grading retrieved documents";
        }
        steps.push({
          id: "grade",
          label: "Grade documents",
          detail,
          color: STEP_COLORS.grade,
          durationMs,
        });
        break;
      }
      case "generation": {
        const model =
          (ev as any).model ?? (ev.payload as any)?.model ?? null;
        const chunks =
          (ev as any).context_chunks ?? (ev.payload as any)?.context_chunks ?? null;
        const parts: string[] = [];
        if (model) parts.push(model);
        if (chunks != null) parts.push(`${chunks} chunks in context`);
        const detail = parts.length > 0 ? parts.join(", ") : "Synthesizing answer with citations";
        steps.push({
          id: "generate",
          label: "Generate",
          detail,
          color: STEP_COLORS.generate,
          durationMs,
        });
        break;
      }
      case "grounding_check": {
        const passed = (ev as any).passed ?? (ev.payload as any)?.passed;
        steps.push({
          id: "hallucination",
          label: "Hallucination check",
          detail: <GroundedBadge passed={passed !== false} />,
          color: STEP_COLORS.hallucination,
          durationMs,
        });
        break;
      }
      default:
        break;
    }
  }

  // Deduplicate: keep last occurrence of each step id (most complete data)
  const deduped: TraceStep[] = [];
  const lastIndexById = new Map<string, number>();
  steps.forEach((s, idx) => lastIndexById.set(s.id, idx));
  steps.forEach((s, idx) => {
    if (lastIndexById.get(s.id) === idx) deduped.push(s);
  });
  return deduped;
}

function computeTotalTime(trace: TraceEvent[]): number | null {
  const timestamps = trace
    .map((ev) => ev.timestamp)
    .filter((t): t is number => t != null);
  if (timestamps.length < 2) return null;
  return Math.round((timestamps[timestamps.length - 1] - timestamps[0]) * 10) / 10;
}

/* ================================================================
   BurstSpinner
   ================================================================ */

export function BurstSpinner({ spinning, size = 22 }: { spinning: boolean; size?: number }) {
  return (
    <motion.svg
      viewBox="0 0 100 100"
      style={{ width: size, height: size, color: "var(--accent)", flexShrink: 0 }}
      animate={spinning ? { rotate: 360 } : { rotate: 0 }}
      transition={
        spinning
          ? { duration: 2.8, ease: "linear", repeat: Infinity }
          : { duration: 0.4, ease: "easeOut" }
      }
    >
      <g transform="translate(50 50)" fill="currentColor">
        {Array.from({ length: 12 }, (_, i) => (
          <rect
            key={i}
            x="-2.8"
            y="-34"
            width="5.6"
            height="18"
            rx="2.8"
            transform={`rotate(${i * 30})`}
          />
        ))}
        <circle r="8" />
      </g>
    </motion.svg>
  );
}

/* ================================================================
   ThinkingBlock — Claude-inspired trace block
   ================================================================ */

export const ThinkingBlock = React.memo(function ThinkingBlock({
  trace,
  isProcessing,
  thinkingSeconds,
}: {
  trace: TraceEvent[];
  isProcessing: boolean;
  thinkingSeconds: number;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const steps = buildSteps(trace);
  const totalTimeSec = computeTotalTime(trace);


  if (steps.length === 0 && !isProcessing) return null;

  const headerText = isProcessing
    ? "Searching NVIDIA AI knowledge base"
    : "Searched NVIDIA AI knowledge base";

  const timeDisplay = !isProcessing && totalTimeSec != null
    ? `${totalTimeSec}s`
    : isProcessing
      ? `${thinkingSeconds}s`
      : null;

  return (
    <motion.div
      className="trace-block"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28 }}
    >
      <button
        className="trace-toggle"
        onClick={() => setCollapsed((c) => !c)}
        type="button"
      >
        <span className={`trace-caret${collapsed ? "" : " expanded"}`}>
          {"\u25B6"}
        </span>
        <span className="trace-header-text">{headerText}</span>
        {timeDisplay && (
          <span className="trace-header-time">{timeDisplay}</span>
        )}
      </button>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="trace-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            <div className="trace-steps">
              <AnimatePresence initial={false}>
                {steps.map((step) => (
                  <motion.div
                    key={step.id}
                    className="trace-step"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.22, ease: "easeOut" }}
                  >
                    <span
                      className="trace-step-dot"
                      style={{ backgroundColor: step.color }}
                    />
                    <div className="trace-step-content">
                      <div className="trace-step-header">
                        <span className="trace-step-name">{step.label}</span>
                        {step.durationMs != null && step.durationMs > 0 && (
                          <span className="trace-step-time">{step.durationMs}ms</span>
                        )}
                      </div>
                      <div className="trace-step-detail">{step.detail}</div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {isProcessing && steps.length === 0 && (
                <div className="trace-step" style={{ color: "var(--text-tertiary)" }}>
                  <span className="trace-step-dot" style={{ backgroundColor: STEP_COLORS.retrieve }} />
                  <div className="trace-step-content">
                    <div className="trace-step-header">
                      <span className="trace-step-name">Processing</span>
                    </div>
                    <div className="trace-step-detail">
                      Analyzing your question
                      <span className="dot-1">.</span>
                      <span className="dot-2">.</span>
                      <span className="dot-3">.</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
});
