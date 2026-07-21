import type { AppMode } from "../types";

interface TopNavProps {
  mode: AppMode;
  onModeChange: (m: AppMode) => void;
  inputFolder?: string;
}

const ITEMS = [
  { id: "dashboard"  as AppMode, label: "Overview", dot: "◈" },
  { id: "inference"  as AppMode, label: "추론 결과", dot: "▶" },
  { id: "test"       as AppMode, label: "테스트 셋", dot: "⊞" },
  { id: "benchmark"  as AppMode, label: "벤치마크", dot: "◆" },
];

export default function TopNav({ mode, onModeChange, inputFolder }: TopNavProps) {
  return (
    <nav className="top-nav">
      <div className="nav-brand">
        <div className="nav-logo">⬡</div>
        <span className="nav-wordmark">AI Scheduling<span> Dashboard</span></span>
      </div>

      <div className="nav-items">
        {ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item${mode === item.id ? " active" : ""}`}
            onClick={() => onModeChange(item.id)}
          >
            <span className="nav-dot" style={{ opacity: mode === item.id ? 1 : 0.4 }} />
            {item.label}
          </button>
        ))}
      </div>

      <div className="nav-right">
        <div className="nav-status">
          <div className="nav-pulse" />
          LIVE
        </div>
        {inputFolder && (
          <code className="nav-folder">{inputFolder}</code>
        )}
      </div>
    </nav>
  );
}
