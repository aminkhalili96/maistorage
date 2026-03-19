import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { ErrorBoundary } from "./ErrorBoundary";

let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

// A child component that conditionally throws during render
function ThrowingComponent({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error("Test render error");
  }
  return <p>Child rendered successfully</p>;
}

beforeEach(() => {
  // Suppress console.error noise from React's error boundary logging
  consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  consoleErrorSpy.mockRestore();
});

describe("ErrorBoundary", () => {
  test("renders children when no error occurs", () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={false} />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Child rendered successfully")).toBeInTheDocument();
  });

  test("shows fallback UI when a child throws during render", () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.queryByText("Child rendered successfully")).not.toBeInTheDocument();
  });

  test("fallback UI shows the error description text", () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>,
    );

    expect(screen.getByText("An unexpected error occurred in the application.")).toBeInTheDocument();
  });

  test("Try again button clears error state and re-renders children", async () => {
    const user = userEvent.setup();

    // Use a mutable ref to control whether the child throws.
    // Start with throwing, then after "Try again" the child should render normally.
    let shouldThrow = true;

    function ConditionalChild() {
      if (shouldThrow) {
        throw new Error("Test render error");
      }
      return <p>Child rendered successfully</p>;
    }

    render(
      <ErrorBoundary>
        <ConditionalChild />
      </ErrorBoundary>,
    );

    // Verify fallback is shown
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();

    // Stop throwing before clicking retry
    shouldThrow = false;

    await user.click(screen.getByRole("button", { name: "Try again" }));

    // After retry, the child should render successfully
    expect(screen.getByText("Child rendered successfully")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });
});
