import { useCallback, useEffect, useMemo, useState } from "react";
import BatchInfoTable from "../components/BatchInfoTable";
import type { AppConfig, DataSummary } from "../types";

interface DatasetPageProps {
  config: AppConfig | null;
  summary: DataSummary | null;
  folderLoading: boolean;
  loadError: string | null;
  onInputFolderChange: (folder: string) => void | Promise<void>;
}

function folderPeriodLabel(folder: string): string {
  const parts = folder.split("/");
  return parts[parts.length - 1] ?? folder;
}

export default function DatasetPage({
  config,
  summary,
  folderLoading,
  loadError,
  onInputFolderChange,
}: DatasetPageProps) {
  const datasetFolders = useMemo(
    () =>
      (config?.input_folders?.length ? config.input_folders : config ? [config.input_folder] : []),
    [config],
  );

  const [selectedFolder, setSelectedFolder] = useState("");

  useEffect(() => {
    if (!config || datasetFolders.length === 0) return;
    setSelectedFolder((prev) => {
      if (prev && datasetFolders.includes(prev)) return prev;
      return config.input_folder;
    });
  }, [config, datasetFolders]);

  const handleDatasetChange = useCallback(
    async (folder: string) => {
      if (!folder || folder === selectedFolder) return;
      setSelectedFolder(folder);
      await onInputFolderChange(folder);
    },
    [onInputFolderChange, selectedFolder],
  );

  return (
    <div className="page">
      <h2>데이터셋 조회</h2>
      <p className="hint page-lead">
        현재 로드된 입력 JSON을 확인합니다. 경로를 바꾸면 학습·추론에 사용하는 데이터도 함께 변경됩니다.
      </p>

      {loadError && <div className="banner banner-warn">{loadError}</div>}

      <div className="card-stagger">
        {config && (
          <section className="card inference-dataset-card">
            <h3>데이터셋 경로</h3>
            <p className="hint">조회할 입력 JSON 경로를 선택합니다.</p>
            <label className="field-label" htmlFor="dataset-folder">
              dataset 경로
            </label>
            <select
              id="dataset-folder"
              className="input-select inference-dataset-select"
              value={selectedFolder}
              onChange={(e) => void handleDatasetChange(e.target.value)}
              disabled={folderLoading || datasetFolders.length === 0}
            >
              {datasetFolders.map((f) => (
                <option key={f} value={f}>
                  {f} ({folderPeriodLabel(f)})
                </option>
              ))}
            </select>
            {folderLoading && <p className="hint">데이터 로드 중…</p>}
            <p className="hint dataset-path-hint">
              실제 폴더: <code>{config.input_dir}</code>
            </p>
          </section>
        )}

        {config && (
          <section className="card">
            <h3>데이터 요약</h3>
            {summary ? (
              <>
                <div className="metrics-row">
                  <div className="metric">
                    <span className="metric-label">EQP</span>
                    <span className="metric-value">{summary.eqp_count}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">LOT</span>
                    <span className="metric-value">{summary.lot_count}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">제품</span>
                    <span className="metric-value">{summary.prod_count}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">공정</span>
                    <span className="metric-value">{summary.oper_count}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Batch 레시피</span>
                    <span className="metric-value">{summary.batch_info_count}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">시뮬 종료(분)</span>
                    <span className="metric-value">{summary.sim_end_minutes}</span>
                  </div>
                </div>
                <p className="hint">
                  시뮬 기준 시각: <code>{summary.sim_base_time}</code>
                </p>
              </>
            ) : (
              <p className="hint">
                선택한 경로에 데이터가 없습니다. dataset 폴더에 JSON을 준비한 뒤 새로고침하세요.
              </p>
            )}
          </section>
        )}

        {summary && (summary.batch_info?.length ?? 0) > 0 && (
          <section className="card">
            <h3>Batch Info (PPK × OPER → LOT_CD / TEMP)</h3>
            <p className="hint page-lead">
              conversion·tool cap 판단에 사용되는 배치 레시피입니다.
            </p>
            <BatchInfoTable rows={summary.batch_info ?? []} />
          </section>
        )}
      </div>
    </div>
  );
}
