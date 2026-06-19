import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import InferencePage from "./pages/InferencePage";
import TrainPage from "./pages/TrainPage";
import { api } from "./lib/api";
import type { AppConfig, AppMode, DataSummary } from "./types";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState<AppMode>("train");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [modelExists, setModelExists] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);

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

  const handleCreateSample = async (inputFolder: string, scenario: string = "default") => {
    setSampleLoading(true);
    try {
      await api.createSample(inputFolder, scenario);
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

  return (
    <div className="app-layout">
      <Sidebar
        mode={mode}
        config={config}
        onModeChange={setMode}
        onCreateSample={handleCreateSample}
        onInputFolderChange={handleInputFolderChange}
        sampleLoading={sampleLoading}
        folderLoading={folderLoading}
      />
      <main className="main-content">
        {loadError && (
          <div className="banner banner-warn">
            {loadError}
            <span className="banner-hint"> — 사이드바에서 샘플 데이터를 생성하세요.</span>
          </div>
        )}
        <div key={mode} className="page-enter">
          {config && mode === "train" && (
            <TrainPage
              config={config}
              summary={summary}
              modelExists={modelExists}
              onTrained={() => api.getModelStatus().then((s) => setModelExists(s.exists))}
              onRefresh={refreshData}
            />
          )}
          {mode === "inference" && config && (
            <InferencePage
              modelExists={modelExists}
              config={config}
              summary={summary}
              onInputFolderChange={handleInputFolderChange}
              folderLoading={folderLoading}
            />
          )}
        </div>
      </main>
    </div>
  );
}
