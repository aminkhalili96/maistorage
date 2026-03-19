import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { CopyButton, FeedbackButtons, RegenerateButton } from "./MessageActions";

const writeTextMock = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined);

// In-memory localStorage mock (jsdom may lack localStorage.clear)
const store: Record<string, string> = {};
vi.spyOn(Storage.prototype, "getItem").mockImplementation(
  (key: string) => store[key] ?? null,
);
vi.spyOn(Storage.prototype, "setItem").mockImplementation(
  (key: string, value: string) => { store[key] = value; },
);
vi.spyOn(Storage.prototype, "removeItem").mockImplementation(
  (key: string) => { delete store[key]; },
);

beforeEach(() => {
  writeTextMock.mockClear();
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText: writeTextMock },
    writable: true,
    configurable: true,
  });
  for (const key of Object.keys(store)) {
    delete store[key];
  }
});

afterEach(() => {
  vi.useRealTimers();
});

describe("CopyButton", () => {
  test("renders with copy text", () => {
    render(<CopyButton text="hello" />);
    const btn = screen.getByRole("button", { name: "Copy answer" });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveTextContent("⎘ Copy");
  });

  test("calls navigator.clipboard.writeText on click", async () => {
    render(<CopyButton text="some answer text" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy answer" }));
    });

    expect(writeTextMock).toHaveBeenCalledWith("some answer text");
  });

  test("shows copied text after successful copy", async () => {
    render(<CopyButton text="test" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy answer" }));
    });

    expect(screen.getByRole("button", { name: "Copy answer" })).toHaveTextContent("✓ Copied");
  });

  test("resets to copy text after timeout", async () => {
    vi.useFakeTimers();

    render(<CopyButton text="test" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy answer" }));
    });

    expect(screen.getByRole("button", { name: "Copy answer" })).toHaveTextContent("✓ Copied");

    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByRole("button", { name: "Copy answer" })).toHaveTextContent("⎘ Copy");
  });
});

describe("FeedbackButtons", () => {
  test("renders thumbs up and down buttons", () => {
    render(<FeedbackButtons messageIndex={0} />);

    expect(screen.getByRole("button", { name: "Mark as helpful" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mark as not helpful" })).toBeInTheDocument();
  });

  test("toggles active-up class on thumbs up click", () => {
    render(<FeedbackButtons messageIndex={0} />);

    const upBtn = screen.getByRole("button", { name: "Mark as helpful" });
    expect(upBtn).not.toHaveClass("active-up");

    fireEvent.click(upBtn);
    expect(upBtn).toHaveClass("active-up");

    // Click again to toggle off
    fireEvent.click(upBtn);
    expect(upBtn).not.toHaveClass("active-up");
  });

  test("only allows one active at a time", () => {
    render(<FeedbackButtons messageIndex={0} />);

    const upBtn = screen.getByRole("button", { name: "Mark as helpful" });
    const downBtn = screen.getByRole("button", { name: "Mark as not helpful" });

    // Click thumbs up
    fireEvent.click(upBtn);
    expect(upBtn).toHaveClass("active-up");
    expect(downBtn).not.toHaveClass("active-down");

    // Click thumbs down — should replace thumbs up
    fireEvent.click(downBtn);
    expect(downBtn).toHaveClass("active-down");
    expect(upBtn).not.toHaveClass("active-up");
  });
});

describe("RegenerateButton", () => {
  test("calls onClick and respects disabled prop", () => {
    const handleClick = vi.fn();

    const { rerender } = render(<RegenerateButton onClick={handleClick} disabled={false} />);

    const btn = screen.getByRole("button", { name: "Regenerate response" });
    expect(btn).toBeEnabled();

    fireEvent.click(btn);
    expect(handleClick).toHaveBeenCalledTimes(1);

    // Re-render as disabled
    rerender(<RegenerateButton onClick={handleClick} disabled={true} />);
    expect(btn).toBeDisabled();

    // Click while disabled should not fire handler
    fireEvent.click(btn);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
