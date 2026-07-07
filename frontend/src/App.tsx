import { useCallback, useEffect, useState } from "react";
import TopNav from "./components/TopNav";
import ErrorLogPanel from "./components/ErrorLogPanel";
import DashboardPage from "./pages/DashboardPage";
import InferencePage from "./pages/InferencePage";
import TestPage from "./pages/TestPage";
import TrainPage from "./pages/TrainPage";
import DatasetPage from "./pages/DatasetPage";
import DatasetEmptyPanel from "./components/DatasetEmptyPanel";
import { api } from "./lib/api";
import type { AppConfig, AppMode, DataSummary } from "./types";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState<AppMode>("dashboard");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [modelExists, setModelExists] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
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
      } catch { /* ignore */ }
    }
  }, []);

  useEffect(() => { void refreshData(); }, [refreshData]);

  const handleInputFolderChange = useCallback(async (folder: string) => {
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
  }, [refreshData]);

  const hasData = (summary?.eqp_count ?? 0) > 0;

  return (
    <div className="app-shell">
      <TopNav
        mode={mode}
        onModeChange={setMode}
        inputFolder={config?.input_folder}
      />

      <main className="app-main">
        {/* 데이터 경고/오류: 축약형 ErrorLogPanel만 표시 */}
        <ErrorLogPanel
          errors={loadError ? [loadError] : []}
          warnings={summary?.warnings ?? []}
        />

        <div key={mode} className="page-enter">
          {mode === "dashboard" && (
            <DashboardPage onNavigate={setMode} />
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

          {mode === "test" && (
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
              onTrained={() => api.getModelStatus().then((s) => setModelExists(s.exists)).catch(() => {})}
              onRefresh={refreshData}
            />
          )}

          {mode === "dataset" && (
            <DatasetPage
              config={config}
              summary={summary}
              folderLoading={folderLoading}
              loadError={loadError}
              onInputFolderChange={handleInputFolderChange}
              onRefresh={refreshData}
            />
          )}
        </div>
      </main>
    </div>
  );
}
