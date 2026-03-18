import React, { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          gap: "1rem",
          fontFamily: "system-ui, sans-serif",
          color: "#e0e0e0",
          background: "#1a1a1a",
        }}>
          <h1 style={{ fontSize: "1.4rem", margin: 0 }}>Something went wrong</h1>
          <p style={{ color: "#999", margin: 0 }}>An unexpected error occurred in the application.</p>
          <button
            onClick={() => this.setState({ hasError: false })}
            style={{
              padding: "0.5rem 1.2rem",
              borderRadius: "6px",
              border: "1px solid #444",
              background: "#2a2a2a",
              color: "#e0e0e0",
              cursor: "pointer",
              fontSize: "0.9rem",
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
