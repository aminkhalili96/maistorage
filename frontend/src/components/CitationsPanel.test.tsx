import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    li: ({ children, ...props }: any) => <li {...props}>{children}</li>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

import { CitationsPanel } from "./CitationsPanel";
import type { Citation } from "../types";

function buildCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "test-chunk-1",
    title: "NCCL Developer Guide",
    url: "https://docs.nvidia.com/nccl",
    citation_url: "https://docs.nvidia.com/nccl#section",
    domain: "docs.nvidia.com",
    section_path: "NCCL > Getting Started",
    snippet:
      "NCCL provides multi-GPU and multi-node collective communication primitives.",
    source_kind: "corpus",
    source_id: "nccl",
    score: 0.85,
    char_count: 500,
    ...overrides,
  };
}

describe("CitationsPanel", () => {
  it("empty citations renders nothing", () => {
    const { container } = render(
      <CitationsPanel
        citations={[]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("single citation renders card", () => {
    render(
      <CitationsPanel
        citations={[buildCitation()]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(screen.getByText("NCCL Developer Guide")).toBeInTheDocument();
  });

  it("citation title displayed", () => {
    render(
      <CitationsPanel
        citations={[buildCitation({ title: "Fabric Manager User Guide" })]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(screen.getByText("Fabric Manager User Guide")).toBeInTheDocument();
  });

  it("citation snippet displayed", () => {
    render(
      <CitationsPanel
        citations={[
          buildCitation({
            snippet: "Short snippet text for testing.",
          }),
        ]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Short snippet text for testing."),
    ).toBeInTheDocument();
  });

  it("citation snippet is truncated when longer than 140 characters", () => {
    const longSnippet = "A".repeat(200);
    render(
      <CitationsPanel
        citations={[buildCitation({ snippet: longSnippet })]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    // Should show first 140 chars + ellipsis
    const expected = "A".repeat(140) + "\u2026";
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it("citation domain displayed", () => {
    render(
      <CitationsPanel
        citations={[buildCitation({ domain: "developer.nvidia.com" })]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(screen.getByText("developer.nvidia.com")).toBeInTheDocument();
  });

  it("source kind badge shows Docs for corpus citations", () => {
    render(
      <CitationsPanel
        citations={[buildCitation({ source_kind: "corpus" })]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    const badge = screen.getByText("Docs");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("corpus");
  });

  it("source kind badge shows Web for web citations", () => {
    render(
      <CitationsPanel
        citations={[buildCitation({ source_kind: "web" })]}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    const badge = screen.getByText("Web");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("web");
  });

  it("click selects citation by calling onSelectCitation with chunk_id", () => {
    const onSelectCitation = vi.fn();
    render(
      <CitationsPanel
        citations={[buildCitation({ chunk_id: "chunk-abc" })]}
        selectedCitationId={null}
        onSelectCitation={onSelectCitation}
      />,
    );
    fireEvent.click(screen.getByText("NCCL Developer Guide"));
    expect(onSelectCitation).toHaveBeenCalledTimes(1);
    expect(onSelectCitation).toHaveBeenCalledWith("chunk-abc");
  });

  it("click on already-selected citation fires onSelectCitation (toggle)", () => {
    const onSelectCitation = vi.fn();
    render(
      <CitationsPanel
        citations={[buildCitation({ chunk_id: "chunk-abc" })]}
        selectedCitationId="chunk-abc"
        onSelectCitation={onSelectCitation}
      />,
    );
    // The card should have the selected class
    const card = screen.getByRole("button");
    expect(card).toHaveClass("selected");

    fireEvent.click(card);
    expect(onSelectCitation).toHaveBeenCalledTimes(1);
    expect(onSelectCitation).toHaveBeenCalledWith("chunk-abc");
  });

  it("multiple citations render all cards", () => {
    const citations = [
      buildCitation({ chunk_id: "c1", title: "NCCL Guide" }),
      buildCitation({ chunk_id: "c2", title: "Fabric Manager Guide" }),
      buildCitation({ chunk_id: "c3", title: "GPU Operator Guide" }),
    ];
    render(
      <CitationsPanel
        citations={citations}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    expect(screen.getByText("NCCL Guide")).toBeInTheDocument();
    expect(screen.getByText("Fabric Manager Guide")).toBeInTheDocument();
    expect(screen.getByText("GPU Operator Guide")).toBeInTheDocument();
  });

  it("citation count badge shows correct number", () => {
    const citations = [
      buildCitation({ chunk_id: "c1", title: "Guide A" }),
      buildCitation({ chunk_id: "c2", title: "Guide B" }),
      buildCitation({ chunk_id: "c3", title: "Guide C" }),
    ];
    render(
      <CitationsPanel
        citations={citations}
        selectedCitationId={null}
        onSelectCitation={vi.fn()}
      />,
    );
    const countBadges = screen.getAllByText("3");
    const countBadge = countBadges.find((el) =>
      el.classList.contains("sources-count"),
    );
    expect(countBadge).toBeDefined();
    expect(countBadge).toHaveClass("sources-count");
  });
});
