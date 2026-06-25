import type { AppConfig, AppMode } from "../types";

interface SidebarProps {
  mode: AppMode;
  config: AppConfig | null;
  onModeChange: (mode: AppMode) => void;
  open: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  mode,
  config,
  onModeChange,
  open,
  onToggle,
}: SidebarProps) {
  return (
    <aside className={`sidebar${open ? "" : " is-collapsed"}`}>
      <div className="sidebar-header">
        <div>
          <h1 className="sidebar-title">AI Scheduling Dashboard</h1>
          <p className="sidebar-subtitle">반도체 설비 AI 스케줄링</p>
        </div>
        <button
          type="button"
          className="sidebar-toggle-btn"
          onClick={onToggle}
          aria-label={open ? "사이드바 숨기기" : "사이드바 보이기"}
          title={open ? "사이드바 숨기기" : "사이드바 보이기"}
        >
          {open ? "◀" : "▶"}
        </button>
      </div>

      <hr />

      <fieldset className="mode-group">
        <legend>워크플로</legend>
        <div className="mode-pills">
          <label className={`mode-pill${mode === "train" ? " active" : ""}`}>
            <input
              type="radio"
              name="mode"
              checked={mode === "train"}
              onChange={() => onModeChange("train")}
            />
            학습 (Train)
          </label>
          <label className={`mode-pill${mode === "test" ? " active" : ""}`}>
            <input
              type="radio"
              name="mode"
              checked={mode === "test"}
              onChange={() => onModeChange("test")}
            />
            테스트 (Test)
          </label>
          <label className={`mode-pill${mode === "inference" ? " active" : ""}`}>
            <input
              type="radio"
              name="mode"
              checked={mode === "inference"}
              onChange={() => onModeChange("inference")}
            />
            추론 (Inference)
          </label>
        </div>
      </fieldset>

      <hr />

      <div className="sidebar-section">
        <button
          type="button"
          className={`btn btn-secondary sidebar-dataset-link${mode === "dataset" ? " active" : ""}`}
          onClick={() => onModeChange("dataset")}
        >
          데이터셋 조회
        </button>
      </div>

      {config && (
        <>
          <hr />
          <div className="sidebar-meta">
            <p>모델: <code>{config.model_dir}</code></p>
            <p>입력: <code>{config.input_dir}</code></p>
            <p>출력: <code>{config.output_dir}</code></p>
            {config.sql_dir && <p>SQL: <code>{config.sql_dir}</code></p>}
          </div>
        </>
      )}
    </aside>
  );
}
