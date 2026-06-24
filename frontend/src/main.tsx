import { Component, StrictMode } from "react";
import type { ReactNode, ErrorInfo } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };

  static getDerivedStateFromError(e: Error) {
    return { error: e };
  }

  componentDidCatch(e: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", e, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <div style={{
          fontFamily: "monospace", padding: "2rem", background: "#fff1f0",
          color: "#c00", minHeight: "100vh", whiteSpace: "pre-wrap",
        }}>
          <strong style={{ fontSize: "1.1rem" }}>런타임 오류 (ErrorBoundary)</strong>
          <hr style={{ margin: "1rem 0", borderColor: "#fca5a5" }} />
          <div>{String(error)}</div>
          <div style={{ marginTop: "1rem", color: "#888", fontSize: "0.85rem" }}>
            {(error as Error).stack}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
