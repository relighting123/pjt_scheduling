import type { AppMode } from "../types";

interface TopNavProps {
  mode: AppMode;
  onModeChange: (m: AppMode) => void;
  inputFolder?: string;
}

const ITEMS = [
  { id: "dashboard"  as AppMode, label: "개요" },
  { id: "inference"  as AppMode, label: "추론 결과" },
  { id: "test"       as AppMode, label: "테스트 셋" },
  { id: "benchmark"  as AppMode, label: "벤치마크" },
  { id: "dataset"    as AppMode, label: "데이터셋" },
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
        {inputFolder && (
          <button
            type="button"
            className="nav-folder"
            onClick={() => onModeChange("dataset")}
            title="데이터셋 변경"
          >
            {inputFolder}
          </button>
        )}
      </div>
    </nav>
  );
}
