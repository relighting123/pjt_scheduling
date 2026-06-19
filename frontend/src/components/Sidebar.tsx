import { useEffect, useState } from "react";
import type { AppConfig, AppMode, SampleScenario } from "../types";
import { api } from "../lib/api";

interface SidebarProps {
  mode: AppMode;
  config: AppConfig | null;
  onModeChange: (mode: AppMode) => void;
  onCreateSample: (inputFolder: string, scenario: string) => void;
  onInputFolderChange: (folder: string) => void;
  sampleLoading: boolean;
  folderLoading: boolean;
}

export default function Sidebar({
  mode,
  config,
  onModeChange,
  onCreateSample,
  onInputFolderChange,
  sampleLoading,
  folderLoading,
}: SidebarProps) {
  const [newFolder, setNewFolder] = useState("");
  const [scenario, setScenario] = useState("default");
  const [scenarios, setScenarios] = useState<SampleScenario[]>([]);

  useEffect(() => {
    api.getSampleScenarios().then((r) => setScenarios(r.scenarios)).catch(() => {});
  }, []);

  const selectedScenario = scenarios.find((s) => s.id === scenario);

  const folders = config?.input_folders?.length
    ? config.input_folders
    : config
      ? [config.input_folder]
      : ["input"];

  const handleFolderSelect = (value: string) => {
    if (value && value !== config?.input_folder) {
      onInputFolderChange(value);
    }
  };

  const handleCreateInNewFolder = () => {
    const name = newFolder.trim();
    if (!name) return;
    onCreateSample(name, scenario);
    setNewFolder("");
  };

  const handleCreateCurrent = () => {
    const folder = config?.input_folder ?? selectedScenario?.default_folder ?? "input";
    onCreateSample(folder, scenario);
  };

  const handleCreateScenarioDefault = () => {
    const folder = selectedScenario?.default_folder ?? "input";
    onCreateSample(folder, scenario);
  };

  return (
    <aside className="sidebar">
      <h1 className="sidebar-title">Post-Scheduling RL</h1>
      <p className="sidebar-subtitle">반도체 설비 스케줄링 최적화</p>

      <hr />

      <fieldset className="mode-group">
        <legend>모드 선택</legend>
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
        <h3>데이터셋</h3>
        <label className="field-label" htmlFor="input-folder">
          입력 폴더
        </label>
        <select
          id="input-folder"
          className="input-select"
          value={config?.input_folder ?? "input"}
          onChange={(e) => handleFolderSelect(e.target.value)}
          disabled={!config || folderLoading}
        >
          {folders.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <label className="field-label" htmlFor="sample-scenario">
          샘플 시나리오
        </label>
        <select
          id="sample-scenario"
          className="input-select"
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          disabled={sampleLoading}
        >
          {(scenarios.length ? scenarios : [
            { id: "default", name: "기본 (3제품)", description: "", default_folder: "input" },
            { id: "single_heavy_wip", name: "단일제품 ST½ 재공다량", description: "", default_folder: "single_heavy_wip" },
          ]).map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
        {selectedScenario?.description && (
          <p className="scenario-hint">{selectedScenario.description}</p>
        )}
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          onClick={handleCreateScenarioDefault}
          disabled={sampleLoading}
        >
          {sampleLoading ? "생성 중..." : `시나리오 폴더에 생성 (${selectedScenario?.default_folder ?? ""})`}
        </button>
        <div className="input-folder-new">
          <input
            type="text"
            className="input-text"
            placeholder="새 폴더명 (예: case_a)"
            value={newFolder}
            onChange={(e) => setNewFolder(e.target.value)}
            disabled={sampleLoading}
          />
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleCreateInNewFolder}
            disabled={sampleLoading || !newFolder.trim()}
          >
            새 폴더에 샘플 생성
          </button>
        </div>
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          onClick={handleCreateCurrent}
          disabled={sampleLoading}
        >
          {sampleLoading ? "생성 중..." : "현재 폴더에 샘플 생성"}
        </button>
      </div>

      {config && (
        <>
          <hr />
          <div className="sidebar-meta">
            <p>모델: <code>{config.model_dir}</code></p>
            <p>입력: <code>{config.input_dir}</code></p>
            <p>출력: <code>{config.output_dir}</code></p>
          </div>
        </>
      )}
    </aside>
  );
}
