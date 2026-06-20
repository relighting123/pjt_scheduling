import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import DatasetEmptyPanel, { type CreateSampleOpts } from "./components/DatasetEmptyPanel";
import DatasetPage from "./pages/DatasetPage";
import InferencePage from "./pages/InferencePage";
import TestPage from "./pages/TestPage";
import TrainPage from "./pages/TrainPage";
import { api } from "./lib/api";
import type { AppConfig, AppMode, DataSummary, GeneratorConfig, SampleScenario } from "./types";
import "./App.css";

const SIDEBAR_KEY = "pjt_sidebar_open";

export default function App() {
  const [mode, setMode] = useState<AppMode>("train");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [modelExists, setModelExists] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) !== "false",
  );

  const [facId, setFacId] = useState("FAC001");
  const [scenario, setScenario] = useState("random");
  const [scenarios, setScenarios] = useState<SampleScenario[]>([]);
  const [genConfig, setGenConfig] = useState<GeneratorConfig | null>(null);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_KEY, String(sidebarOpen));
  }, [sidebarOpen]);

  useEffect(() => {
    Promise.all([api.getSampleScenarios(), api.getGeneratorConfigDefaults()])
      .then(([sc, defs]) => {
        setScenarios(sc.scenarios);
        setGenConfig(defs.defaults);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (config?.fac_id) setFacId(config.fac_id);
  }, [config?.fac_id]);

  const sampleOpts = useCallback(
    (): CreateSampleOpts => ({
      fac_id: facId,
      scenario,
      generator_config: genConfig ?? undefined,
    }),
    [facId, scenario, genConfig],
  );

  const refreshData = useCallback(async () => {
    setLoadError(null);
    try {
      const [cfg, sum, status] = await Promise.all([
        api.getConfig(),
        api.getDataSummary(),
        api.getModelStatus(),
      ]);
      setConfig(cfg);
      setSummary(sum);
      setModelExists(status.exists);
    } catch (e) {
      setSummary(null);
      setLoadError(e instanceof Error ? e.message : "데이터 로드 실패");
      try {
        const cfg = await api.getConfig();
        setConfig(cfg);
        const status = await api.getModelStatus();
        setModelExists(status.exists);
      } catch {
        /* ignore */
      }
    }
  }, []);

  useEffect(() => {
    refreshData();
  }, [refreshData]);

  const handleCreateSample = async (opts: CreateSampleOpts) => {
    setSampleLoading(true);
    setLoadError(null);
    try {
      await api.createSample(opts);
      await refreshData();
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "샘플 생성 실패");
    } finally {
      setSampleLoading(false);
    }
  };

  const handleInputFolderChange = async (folder: string) => {
    setFolderLoading(true);
    setLoadError(null);
    try {
      await api.setInputFolder(folder);
      await refreshData();
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "입력 폴더 변경 실패");
    } finally {
      setFolderLoading(false);
    }
  };

  const hasData = summary !== null;
  const datasetProps = {
    config,
    summary,
    facId,
    onFacIdChange: setFacId,
    scenario,
    onScenarioChange: setScenario,
    scenarios,
    genConfig,
    onGenConfigChange: (patch: Partial<GeneratorConfig>) =>
      setGenConfig((prev) => (prev ? { ...prev, ...patch } : prev)),
    sampleLoading,
    loadError,
    onCreateSample: handleCreateSample,
  };

  return (
    <div className={`app-layout${sidebarOpen ? "" : " sidebar-collapsed"}`}>
      <Sidebar
        mode={mode}
        config={config}
        onModeChange={setMode}
        onInputFolderChange={handleInputFolderChange}
        folderLoading={folderLoading}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
      />
      {!sidebarOpen && (
        <button
          type="button"
          className="sidebar-fab"
          onClick={() => setSidebarOpen(true)}
          aria-label="사이드바 보이기"
        >
          ▶ 메뉴
        </button>
      )}
      <main className="main-content">
        {loadError && hasData && mode !== "dataset" && mode !== "test" && (
          <div className="banner banner-warn">{loadError}</div>
        )}
        <div key={mode} className="page-enter">
          {mode === "dataset" && <DatasetPage {...datasetProps} />}
          {mode === "test" && config && (
            <TestPage config={config} modelExists={modelExists} />
          )}
          {mode !== "dataset" && mode !== "test" && !hasData && (
            <DatasetEmptyPanel
              facId={facId}
              scenario={scenario}
              scenarios={scenarios}
              sampleLoading={sampleLoading}
              loadError={loadError}
              onCreateSample={handleCreateSample}
              sampleOpts={sampleOpts}
              onGoToDataset={() => setMode("dataset")}
            />
          )}
          {mode === "train" && hasData && config && (
            <TrainPage
              config={config}
              summary={summary}
              modelExists={modelExists}
              onTrained={() => api.getModelStatus().then((s) => setModelExists(s.exists))}
              onRefresh={refreshData}
            />
          )}
          {mode === "inference" && hasData && config && summary && (
            <InferencePage
              modelExists={modelExists}
              config={config}
              summary={summary}
              folderLoading={folderLoading}
            />
          )}
        </div>
      </main>
    </div>
  );
}
