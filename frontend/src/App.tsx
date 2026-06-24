import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import DatasetEmptyPanel from "./components/DatasetEmptyPanel";
import DatasetPage from "./pages/DatasetPage";
import InferencePage from "./pages/InferencePage";
import TestPage from "./pages/TestPage";
import TrainPage from "./pages/TrainPage";
import { api } from "./lib/api";
import type { AppConfig, AppMode, DataSummary } from "./types";
import "./App.css";

const SIDEBAR_KEY = "pjt_sidebar_open";

export default function App() {
  const [mode, setMode] = useState<AppMode>("train");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [modelExists, setModelExists] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [folderLoading, setFolderLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) !== "false",
  );

  useEffect(() => {
    localStorage.setItem(SIDEBAR_KEY, String(sidebarOpen));
  }, [sidebarOpen]);

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

  return (
    <div className={`app-layout${sidebarOpen ? "" : " sidebar-collapsed"}`}>
      <Sidebar
        mode={mode}
        config={config}
        onModeChange={setMode}
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
        {loadError && (hasData || mode === "inference") && mode !== "dataset" && mode !== "test" && (
          <div className="banner banner-warn">{loadError}</div>
        )}
        <div key={mode} className="page-enter">
          {mode === "dataset" && (
            <DatasetPage
              config={config}
              summary={summary}
              folderLoading={folderLoading}
              loadError={loadError}
              onInputFolderChange={handleInputFolderChange}
            />
          )}
          {mode === "test" && config && (
            <TestPage config={config} modelExists={modelExists} />
          )}
          {mode === "train" && !hasData && (
            <DatasetEmptyPanel
              loadError={loadError}
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
          {mode === "inference" && (
            <InferencePage
              modelExists={modelExists}
              config={config}
              summary={summary}
              folderLoading={folderLoading}
              onInputFolderChange={handleInputFolderChange}
            />
          )}
        </div>
      </main>
    </div>
  );
}
