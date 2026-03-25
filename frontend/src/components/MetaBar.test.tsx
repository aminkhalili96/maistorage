import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { MetaBar } from "./MetaBar";
import type { ChatDonePayload } from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildPayload(overrides: Partial<ChatDonePayload> = {}): ChatDonePayload {
  return {
    answer: "Test answer",
    assistant_mode: "doc_rag",
    confidence: 0.85,
    used_fallback: false,
    response_mode: "knowledge-base-backed",
    retry_count: 0,
    grounding_passed: true,
    answer_quality_passed: true,
    rejected_chunk_count: 0,
    citation_count: 3,
    query_class: "factoid",
    source_families: ["cuda"],
    model_used: "gpt-5.4",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MetaBar", () => {
  test("null payload renders nothing", () => {
    const { container } = render(<MetaBar payload={null} />);
    expect(container.querySelector(".meta-bar")).not.toBeInTheDocument();
  });

  test("direct-chat response_mode renders nothing", () => {
    const { container } = render(
      <MetaBar payload={buildPayload({ response_mode: "direct-chat" })} />,
    );
    expect(container.querySelector(".meta-bar")).not.toBeInTheDocument();
  });

  test("knowledge-base-backed shows Knowledge-base-backed badge", () => {
    render(<MetaBar payload={buildPayload({ response_mode: "knowledge-base-backed" })} />);
    expect(screen.getByText("Knowledge-base-backed")).toBeInTheDocument();
  });

  test("web-backed shows Web-backed badge", () => {
    render(<MetaBar payload={buildPayload({ response_mode: "web-backed" })} />);
    expect(screen.getByText("Web-backed")).toBeInTheDocument();
  });

  test("llm-knowledge shows LLM knowledge badge", () => {
    render(
      <MetaBar payload={buildPayload({ response_mode: "llm-knowledge", confidence: 0 })} />,
    );
    expect(screen.getByText("LLM knowledge")).toBeInTheDocument();
  });

  test("insufficient-evidence shows badge", () => {
    render(
      <MetaBar
        payload={buildPayload({ response_mode: "insufficient-evidence", confidence: 0 })}
      />,
    );
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
  });

  test("confidence display for knowledge-base-backed", () => {
    render(<MetaBar payload={buildPayload({ confidence: 0.857 })} />);
    // Math.round(0.857 * 100) = 86
    expect(screen.getByText("86% confidence")).toBeInTheDocument();
  });

  test("confidence hidden when confidence is 0", () => {
    render(<MetaBar payload={buildPayload({ confidence: 0 })} />);
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });

  test("confidence hidden for llm-knowledge mode", () => {
    render(
      <MetaBar
        payload={buildPayload({ response_mode: "llm-knowledge", confidence: 0.9 })}
      />,
    );
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });

  test("grounding passed shows checkmark", () => {
    render(<MetaBar payload={buildPayload({ grounding_passed: true })} />);
    expect(screen.getByText("\u2713 grounding")).toBeInTheDocument();
  });

  test("grounding failed shows X mark", () => {
    render(<MetaBar payload={buildPayload({ grounding_passed: false })} />);
    expect(screen.getByText("\u2717 grounding")).toBeInTheDocument();
  });

  test("quality passed shows checkmark", () => {
    render(<MetaBar payload={buildPayload({ answer_quality_passed: true })} />);
    expect(screen.getByText("\u2713 quality")).toBeInTheDocument();
  });

  test("quality failed shows warning", () => {
    render(<MetaBar payload={buildPayload({ answer_quality_passed: false })} />);
    expect(screen.getByText("\u26A0 quality")).toBeInTheDocument();
  });

  test("generation_degraded shows degraded warning", () => {
    render(<MetaBar payload={buildPayload({ generation_degraded: true })} />);
    // In JSX text (no curly braces), \u26A0 is literal characters, not a unicode escape
    expect(screen.getByText(/degraded/)).toBeInTheDocument();
  });

  test("generation_degraded false does not show degraded warning", () => {
    render(<MetaBar payload={buildPayload({ generation_degraded: false })} />);
    expect(screen.queryByText(/degraded/)).not.toBeInTheDocument();
  });

  test("rejected chunk count shows when greater than 0", () => {
    render(<MetaBar payload={buildPayload({ rejected_chunk_count: 4 })} />);
    expect(screen.getByText("4 rejected")).toBeInTheDocument();
  });

  test("rejected chunk count hidden when 0", () => {
    render(<MetaBar payload={buildPayload({ rejected_chunk_count: 0 })} />);
    expect(screen.queryByText(/rejected/)).not.toBeInTheDocument();
  });

  test("trust badge has correct CSS class for knowledge-base-backed", () => {
    render(<MetaBar payload={buildPayload({ response_mode: "knowledge-base-backed" })} />);
    const badge = screen.getByText("Knowledge-base-backed");
    expect(badge).toHaveClass("trust-badge", "knowledge-base");
  });

  test("trust badge has correct CSS class for web-backed", () => {
    render(<MetaBar payload={buildPayload({ response_mode: "web-backed" })} />);
    const badge = screen.getByText("Web-backed");
    expect(badge).toHaveClass("trust-badge", "web");
  });
});
