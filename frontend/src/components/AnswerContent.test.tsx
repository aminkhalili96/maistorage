import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { AnswerContent } from "./AnswerContent";
import type { Citation } from "../types";

function buildCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "chunk-1",
    title: "CUDA Installation Guide",
    url: "https://docs.nvidia.com/cuda/",
    citation_url: "https://docs.nvidia.com/cuda/#install",
    domain: "docs.nvidia.com",
    section_path: "Linux Install",
    snippet: "Install the NVIDIA driver and the CUDA toolkit before running containers.",
    source_kind: "knowledge_base",
    ...overrides,
  };
}

describe("AnswerContent", () => {
  test("renders plain markdown text", () => {
    render(<AnswerContent text="Hello **world**" isStreaming={false} />);
    expect(screen.getByText("world")).toBeInTheDocument();
    // "world" should be bold
    const bold = screen.getByText("world");
    expect(bold.tagName).toBe("STRONG");
  });

  test("renders citation refs as spans with class citation-ref", () => {
    render(
      <AnswerContent
        text="The H100 uses NVLink 4.0 [1] for interconnect."
        isStreaming={false}
        citations={[buildCitation()]}
      />,
    );
    const citationSpan = screen.getByText("1");
    expect(citationSpan).toBeInTheDocument();
    expect(citationSpan).toHaveClass("citation-ref");
    expect(citationSpan).toHaveAttribute("data-idx", "1");
  });

  test("renders GFM tables as HTML table elements", () => {
    const tableMarkdown = `| GPU | Memory |\n| --- | --- |\n| H100 | 80GB |\n| A100 | 40GB |`;
    const { container } = render(
      <AnswerContent text={tableMarkdown} isStreaming={false} />,
    );
    const table = container.querySelector("table");
    expect(table).toBeInTheDocument();
    expect(screen.getByText("H100")).toBeInTheDocument();
    expect(screen.getByText("80GB")).toBeInTheDocument();
  });

  test("renders code blocks with code-block-wrapper div", () => {
    const codeMarkdown = "```python\nprint('hello')\n```";
    const { container } = render(
      <AnswerContent text={codeMarkdown} isStreaming={false} />,
    );
    const wrapper = container.querySelector(".code-block-wrapper");
    expect(wrapper).toBeInTheDocument();
    const pre = wrapper?.querySelector("pre");
    expect(pre).toBeInTheDocument();
  });

  test("shows streaming cursor when isStreaming is true", () => {
    const { container } = render(
      <AnswerContent text="Loading answer..." isStreaming={true} />,
    );
    const cursor = container.querySelector(".streaming-cursor");
    expect(cursor).toBeInTheDocument();
  });

  test("hides streaming cursor when isStreaming is false", () => {
    const { container } = render(
      <AnswerContent text="Complete answer." isStreaming={false} />,
    );
    const cursor = container.querySelector(".streaming-cursor");
    expect(cursor).not.toBeInTheDocument();
  });

  test("citation click calls onCitationClick with correct chunk_id", async () => {
    const onCitationClick = vi.fn();
    const citations = [
      buildCitation({ chunk_id: "chunk-abc" }),
      buildCitation({ chunk_id: "chunk-def", title: "NCCL Guide" }),
    ];

    render(
      <AnswerContent
        text="First point [1] and second point [2]."
        isStreaming={false}
        citations={citations}
        onCitationClick={onCitationClick}
      />,
    );

    // Click citation [2]
    const citationSpans = screen.getAllByText("2").filter(
      (el) => el.classList.contains("citation-ref"),
    );
    expect(citationSpans.length).toBe(1);
    fireEvent.click(citationSpans[0]);

    expect(onCitationClick).toHaveBeenCalledTimes(1);
    expect(onCitationClick).toHaveBeenCalledWith("chunk-def");
  });

  test("handles empty text gracefully", () => {
    const { container } = render(
      <AnswerContent text="" isStreaming={false} />,
    );
    const answerDiv = container.querySelector(".answer-content");
    expect(answerDiv).toBeInTheDocument();
    // No crash, no streaming cursor
    expect(container.querySelector(".streaming-cursor")).not.toBeInTheDocument();
  });

  test("handles malformed citation ref [999] without crash", () => {
    const onCitationClick = vi.fn();
    const citations = [buildCitation({ chunk_id: "chunk-only" })];

    render(
      <AnswerContent
        text="Some fact [999] is not backed."
        isStreaming={false}
        citations={citations}
        onCitationClick={onCitationClick}
      />,
    );

    // The span should still render
    const citationSpan = screen.getByText("999");
    expect(citationSpan).toHaveClass("citation-ref");
    expect(citationSpan).toHaveAttribute("data-idx", "999");

    // Clicking it should NOT call onCitationClick because citations[998] is undefined
    fireEvent.click(citationSpan);
    expect(onCitationClick).not.toHaveBeenCalled();
  });

  test("code blocks contain a copy button with aria-label Copy code", () => {
    const codeMarkdown = "```bash\nnvidia-smi\n```";
    render(<AnswerContent text={codeMarkdown} isStreaming={false} />);
    const copyBtn = screen.getByRole("button", { name: "Copy code" });
    expect(copyBtn).toBeInTheDocument();
    expect(copyBtn).toHaveTextContent("Copy");
  });
});
