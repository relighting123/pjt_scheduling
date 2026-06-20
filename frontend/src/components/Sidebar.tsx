import type { AppConfig, AppMode } from "../types";

interface SidebarProps {
  mode: AppMode;
  config: AppConfig | null;
  onModeChange: (mode: AppMode) => void;
  onInputFolderChange: (folder: string) => void;
  folderLoading: boolean;
  open: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  mode,
  config,
  onModeChange,
  onInputFolderChange,
  folderLoading,
  open,
  onToggle,
}: SidebarProps) {
  const folders = config?.input_folders?.length
    ? config.input_folders
    : config
      ? [config.input_folder]
      : ["FAC001/train"];

  const handleFolderSelect = (value: string) => {
    if (value && value !== config?.input_folder) {
      onInputFolderChange(value);
    }
  };

  return (
    <aside className={`sidebar${open ? "" : " is-collapsed"}`}>
      <div className="sidebar-header">
        <div>
          <h1 className="sidebar-title">Post-Scheduling RL</h1>
          <p className="sidebar-subtitle">반도체 설비 스케줄링 최적화</p>
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
        <h3>데이터셋 선택</h3>
        <label className="field-label" htmlFor="input-folder">
          dataset 경로
        </label>
        <select
          id="input-folder"
          className="input-select"
          value={config?.input_folder ?? folders[0]}
          onChange={(e) => handleFolderSelect(e.target.value)}
          disabled={!config || folderLoading}
        >
          {folders.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <p className="sidebar-hint">
          {mode === "test"
            ? "Test 탭은 FAC 하위 test 기간 전체를 벤치마크합니다."
            : mode === "train"
              ? "단일 선택 또는 학습 탭에서 기간·복수 스냅샷을 지정할 수 있습니다."
              : "추론에 사용할 입력 경로입니다."}
        </p>
      </div>

      <hr />

      <div className="sidebar-section">
        <button
          type="button"
          className={`btn btn-secondary sidebar-dataset-link${mode === "dataset" ? " active" : ""}`}
          onClick={() => onModeChange("dataset")}
        >
          데이터셋 생성
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
