import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import type { TraceEvent } from "../types";

// ---------------------------------------------------------------------------
// Mock framer-motion so we don't need a real animation runtime
// ---------------------------------------------------------------------------

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: any) => {
      const { initial, animate, exit, transition, ...rest } = props;
      return <div {...rest}>{children}</div>;
    },
    svg: ({ children, ...props }: any) => {
      const { initial, animate, exit, transition, ...rest } = props;
      return <svg {...rest}>{children}</svg>;
    },
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

import { ThinkingBlock } from "./ThinkingBlock";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildTraceEvent(overrides: Partial<TraceEvent> = {}): TraceEvent {
  return {
    type: "retrieval",
    message: "",
    payload: {},
    timestamp: Date.now() / 1000,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ThinkingBlock", () => {
  test("renders trace steps with correct labels", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now, payload: { result_count: 10 } }),
      buildTraceEvent({ type: "document_grading", timestamp: now + 0.5, payload: { kept_count: 5, total_count: 10 } }),
      buildTraceEvent({ type: "generation", timestamp: now + 1.0, payload: { model: "gpt-5.4" } }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={3} />);

    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Grade documents")).toBeInTheDocument();
    expect(screen.getByText("Generate")).toBeInTheDocument();
  });

  test("shows Processing text with animated dots when isProcessing and no steps", () => {
    render(<ThinkingBlock trace={[]} isProcessing={true} thinkingSeconds={2} />);

    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("Searching NVIDIA AI infrastructure corpus")).toBeInTheDocument();
  });

  test("hides processing indicator when isProcessing is false with trace events", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now, payload: { result_count: 5 } }),
      buildTraceEvent({ type: "generation", timestamp: now + 1.0, payload: {} }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={0} />);

    expect(screen.queryByText("Processing")).not.toBeInTheDocument();
    expect(screen.getByText("Searched NVIDIA AI infrastructure corpus")).toBeInTheDocument();
  });

  test("collapse/expand toggle hides and shows trace steps", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now, payload: { result_count: 8 } }),
      buildTraceEvent({ type: "generation", timestamp: now + 1.0, payload: {} }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={3} />);

    // Steps visible initially
    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Generate")).toBeInTheDocument();

    // Click the toggle button to collapse
    const toggleBtn = screen.getByRole("button");
    fireEvent.click(toggleBtn);

    // Steps should be hidden
    expect(screen.queryByText("Retrieve")).not.toBeInTheDocument();
    expect(screen.queryByText("Generate")).not.toBeInTheDocument();

    // Click again to expand
    fireEvent.click(toggleBtn);

    // Steps visible again
    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Generate")).toBeInTheDocument();
  });

  test("empty trace with isProcessing false renders nothing", () => {
    const { container } = render(
      <ThinkingBlock trace={[]} isProcessing={false} thinkingSeconds={0} />,
    );

    // Component returns null
    expect(container.querySelector(".trace-block")).not.toBeInTheDocument();
  });

  test("displays thinkingSeconds while processing", () => {
    render(<ThinkingBlock trace={[]} isProcessing={true} thinkingSeconds={5} />);

    expect(screen.getByText("5s")).toBeInTheDocument();
  });

  test("displays totalTimeSec when done processing", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now }),
      buildTraceEvent({ type: "generation", timestamp: now + 2.3 }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={0} />);

    expect(screen.getByText("2.3s")).toBeInTheDocument();
  });

  test("handles grounding_check trace event with passed=true", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now, payload: { result_count: 5 } }),
      buildTraceEvent({ type: "grounding_check", timestamp: now + 1.0, payload: { passed: true } }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={0} />);

    expect(screen.getByText("Hallucination check")).toBeInTheDocument();
    expect(screen.getByText("GROUNDED")).toBeInTheDocument();
    expect(screen.getByText("Answer supported by sources")).toBeInTheDocument();
  });

  test("handles grounding_check trace event with passed=false", () => {
    const now = Date.now() / 1000;
    const trace: TraceEvent[] = [
      buildTraceEvent({ type: "retrieval", timestamp: now, payload: {} }),
      buildTraceEvent({ type: "grounding_check", timestamp: now + 1.0, payload: { passed: false } }),
    ];

    render(<ThinkingBlock trace={trace} isProcessing={false} thinkingSeconds={0} />);

    expect(screen.getByText("Hallucination check")).toBeInTheDocument();
    expect(screen.getByText("NOT GROUNDED")).toBeInTheDocument();
    expect(screen.getByText("Answer not supported by sources")).toBeInTheDocument();
  });
});
