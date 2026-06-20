import { useCallback, useEffect, useMemo, useState } from "react";
import PlotChart from "../components/PlotChart";
import ChartSettingsPanel from "../components/ChartSettingsPanel";
import { api } from "../lib/api";
import {
  ALGO_CHART_COLORS,
  benchmarkRowsFromResponse,
  buildAlgorithmAchievementComparison,
  buildAlgorithmGanttComparison,
  buildAlgorithmKpiComparison,
  buildTestMetricChart,
  resultScheduleStats,
  TEST_METRICS,
  type AlgoCompareEntry,
} from "../lib/charts";
import type {
  AlgorithmId,
  AlgorithmInfo,
  AppConfig,
  TestBenchmarkDataset,
  TestBenchmarkResponse,
  TestDatasetInfo,
} from "../types";

interface TestPageProps {
  config: AppConfig | null;
  modelExists: boolean;
}

const FALLBACK_ALGOS: AlgorithmInfo[] = [
  { id: "rl", name: "PPO (강화학습)", description: "", requires_model: true },
  { id: "minprogress", name: "Min-Progress (휴리스틱)", description: "", requires_model: false },
  { id: "earliest_st", name: "Earliest-ST (휴리스틱)", description: "", requires_model: false },
];

function formatSavedTime(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString("ko-KR");
  } catch {
    return iso;
  }
}

export default function TestPage({ config, modelExists }: TestPageProps) {
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [compareAlgos, setCompareAlgos] = useState<Set<AlgorithmId>>(new Set());
  const [benchmark, setBenchmark] = useState<TestBenchmarkResponse | null>(null);
  const [testFolders, setTestFolders] = useState<TestDatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingSaved, setLoadingSaved] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [runProgress, setRunProgress] = useState<{ current: number; total: number; label: string } | null>(null);

  const [ganttTimeFixed, setGanttTimeFixed] = useState(false);
  const [ganttTimeStart, setGanttTimeStart] = useState(0);
  const [ganttTimeEnd, setGanttTimeEnd] = useState(1440);

  const facId = config?.fac_id ?? "FAC001";
  const algoList = algorithms.length ? algorithms : FALLBACK_ALGOS;

  const availableAlgos = useMemo(
    () => algoList.filter((a) => !a.requires_model || modelExists),
    [algoList, modelExists],
  );

  const algoLabels = useMemo(() => {
    const map: Record<string, string> = {};
    algoList.forEach((a) => { map[a.id] = a.name; });
    return map;
  }, [algoList]);

  const runAlgos = useMemo(
    () => [...compareAlgos].filter((id) => availableAlgos.some((a) => a.id === id)),
    [compareAlgos, availableAlgos],
  );

  const displayAlgos = useMemo((): AlgorithmId[] => {
    if (benchmark?.algorithms?.length) {
      return benchmark.algorithms.filter((id) => algoLabels[id]);
    }
    return runAlgos;
  }, [benchmark?.algorithms, runAlgos, algoLabels]);

  useEffect(() => {
    api.getAlgorithms().then((res) => setAlgorithms(res.algorithms)).catch(() => {});
  }, []);

  useEffect(() => {
    setCompareAlgos(new Set(availableAlgos.map((a) => a.id)));
  }, [availableAlgos]);

  const refreshFolders = useCallback(() => {
    api.getTestDatasets(facId)
      .then((res) => setTestFolders(res.datasets))
      .catch(() => setTestFolders([]));
  }, [facId]);

  useEffect(() => {
    refreshFolders();
  }, [refreshFolders]);

  const loadSaved = useCallback(async () => {
    setLoadingSaved(true);
    try {
      const saved = await api.getSavedTestBenchmark(facId);
      if (saved.datasets?.length) {
        setBenchmark(saved);
        const firstOk = saved.datasets.find((d) => d.results.length > 0);
        if (firstOk) setSelectedFolder(firstOk.input_folder);
      } else {
        setBenchmark(null);
      }
    } catch {
      setBenchmark(null);
    } finally {
      setLoadingSaved(false);
    }
  }, [facId]);

  useEffect(() => {
    loadSaved();
  }, [loadSaved]);

  const chartRows = useMemo(
    () => (benchmark ? benchmarkRowsFromResponse(benchmark.datasets, algoLabels) : []),
    [benchmark, algoLabels],
  );

  const selectedDataset = useMemo(
    () => benchmark?.datasets.find((d) => d.input_folder === selectedFolder) ?? null,
    [benchmark, selectedFolder],
  );

  const selectedEntries = useMemo((): AlgoCompareEntry[] => {
    if (!selectedDataset) return [];
    return selectedDataset.results.map((r) => ({
      algorithm: r.algorithm ?? "rl",
      label: algoLabels[r.algorithm ?? "rl"] ?? (r.algorithm ?? "rl"),
      result: r,
    }));
  }, [selectedDataset, algoLabels]);

  const dataTimeEnd = selectedDataset?.sim_end_minutes ?? 0;

  useEffect(() => {
    if (dataTimeEnd > 0 && !ganttTimeFixed) {
      setGanttTimeEnd(dataTimeEnd);
    }
  }, [dataTimeEnd, ganttTimeFixed]);

  const ganttAxis = useMemo(
    () => ({
      eqpIds: selectedDataset?.eqp_ids ?? [],
      timeStartMinutes: ganttTimeFixed ? ganttTimeStart : 0,
      timeEndMinutes: ganttTimeFixed ? ganttTimeEnd : dataTimeEnd,
      fixedRange: ganttTimeFixed,
    }),
    [selectedDataset, ganttTimeFixed, ganttTimeStart, ganttTimeEnd, dataTimeEnd],
  );

  const toggleAlgo = (id: AlgorithmId) => {
    setCompareAlgos((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const runBenchmark = useCallback(async () => {
    if (!runAlgos.length) {
      setError("비교할 알고리즘을 하나 이상 선택하세요.");
      return;
    }
    if (!testFolders.length) {
      setError("test 데이터셋이 없습니다.");
      return;
    }

    setLoading(true);
    setError(null);
    setRunProgress({ current: 0, total: testFolders.length, label: "" });

    try {
      await api.initTestBenchmark(runAlgos, facId);
      setBenchmark({
        fac_id: facId,
        algorithms: runAlgos,
        status: "running",
        progress: { current: 0, total: testFolders.length, label: "" },
        datasets: [],
      });

      for (let i = 0; i < testFolders.length; i++) {
        const ds = testFolders[i];
        setRunProgress({ current: i + 1, total: testFolders.length, label: ds.label });

        const res = await api.runTestBenchmarkOne({
          algorithms: runAlgos,
          input_folder: ds.input_folder,
          fac_id: facId,
          progress_current: i + 1,
          progress_total: testFolders.length,
          done: i === testFolders.length - 1,
        });

        setBenchmark(res);
        const lastOk = [...res.datasets].reverse().find((d) => d.results.length > 0);
        if (lastOk) setSelectedFolder(lastOk.input_folder);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "테스트 벤치마크 실패");
    } finally {
      setLoading(false);
      setRunProgress(null);
    }
  }, [runAlgos, facId, testFolders]);

  const clearSaved = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.clearSavedTestBenchmark(facId);
      setBenchmark(null);
      setSelectedFolder(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "저장 결과 삭제 실패");
    } finally {
      setLoading(false);
    }
  };

  const selectByIndex = (index: number) => {
    const row = chartRows[index];
    if (row) setSelectedFolder(row.input_folder);
  };

  const selectedLabel = selectedDataset?.label;
  const hasResults = (benchmark?.datasets.length ?? 0) > 0;
  const progressPct = runProgress && runProgress.total > 0
    ? Math.round((runProgress.current / runProgress.total) * 100)
    : 0;

  const datasetByFolder = useMemo(() => {
    const map = new Map<string, TestBenchmarkDataset>();
    benchmark?.datasets.forEach((d) => map.set(d.input_folder, d));
    return map;
  }, [benchmark]);

  return (
    <div className="page">
      <h2>테스트 (Test)</h2>
      <p className="hint page-lead">
        FAC <strong>{facId}</strong> test 데이터셋({testFolders.length}개)을 <strong>하나씩</strong> 실행하며
        결과를 즉시 표시합니다. 완료 후 <code>test_benchmark.json</code>에 저장되어 다음 접속 시 그대로 불러옵니다.
      </p>

      {benchmark?.updated_at && !loading && (
        <p className="hint test-saved-meta">
          저장된 결과 · {formatSavedTime(benchmark.updated_at)}
          {benchmark.status === "complete" ? " · 완료" : benchmark.status === "running" ? " · 실행 중" : ""}
        </p>
      )}

      <section className="card">
        <h3>비교 알고리즘</h3>
        <div className="algo-check-group">
          {algoList.map((algo) => {
            const disabled = algo.requires_model && !modelExists;
            return (
              <label key={algo.id} className={`algo-check${disabled ? " disabled" : ""}`}>
                <input
                  type="checkbox"
                  checked={compareAlgos.has(algo.id)}
                  disabled={disabled || loading}
                  onChange={() => toggleAlgo(algo.id)}
                />
                <span className="algo-name">{algo.name}</span>
                {disabled && <span className="algo-desc"> (모델 없음)</span>}
              </label>
            );
          })}
        </div>
        <div className="btn-row">
          <button
            type="button"
            className={`btn btn-primary${loading ? " is-loading" : ""}`}
            onClick={runBenchmark}
            disabled={loading || runAlgos.length === 0 || testFolders.length === 0}
          >
            {loading
              ? `실행 중 ${runProgress?.current ?? 0}/${runProgress?.total ?? 0}…`
              : `test ${testFolders.length}개 순차 실행`}
          </button>
          {hasResults && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={clearSaved}
              disabled={loading}
            >
              저장 결과 삭제
            </button>
          )}
        </div>
        {loading && runProgress && (
          <div className="test-progress">
            <div className="test-progress-bar" style={{ width: `${progressPct}%` }} />
            <span className="test-progress-label">
              {runProgress.current}/{runProgress.total} · {runProgress.label}
            </span>
          </div>
        )}
        {testFolders.length === 0 && (
          <p className="status-warn">test 데이터셋이 없습니다. 데이터셋 페이지에서 test 기간을 생성하세요.</p>
        )}
      </section>

      {error && <p className="error">{error}</p>}

      {loadingSaved && !benchmark && (
        <p className="hint">저장된 테스트 결과 불러오는 중…</p>
      )}

      {hasResults && (
        <div className="test-results card-stagger">
          {chartRows.length > 0 && (
            <section className="card">
              <h3>{chartRows.length >= 2 ? "지표별 성능 추이" : "지표별 알고리즘 비교"}</h3>
              <p className="hint">
                {chartRows.length >= 2
                  ? "X축: 데이터셋 · Y축: 지표 · 선: 알고리즘 (차트 클릭으로 선택)"
                  : "데이터셋 1개 — 알고리즘별 막대 차트"}
              </p>
              <div className="test-metric-grid">
                {TEST_METRICS.map((metric) => (
                  <div key={metric.key} className="test-metric-chart">
                    <PlotChart
                      key={`${metric.key}-${chartRows.map((r) => r.input_folder).join("|")}`}
                      {...buildTestMetricChart(
                        metric,
                        chartRows,
                        displayAlgos,
                        algoLabels,
                        selectedLabel,
                      )}
                      onPointClick={chartRows.length >= 2 ? selectByIndex : undefined}
                    />
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="card">
            <h3>데이터셋 목록</h3>
            <div className="table-wrap">
              <table className="compare-table test-dataset-table">
                <thead>
                  <tr>
                    <th>상태</th>
                    <th>데이터셋</th>
                    <th>경로</th>
                    {displayAlgos.map((id) => (
                      <th key={id}>{algoLabels[id] ?? id}</th>
                    ))}
                    <th>평균 달성률</th>
                  </tr>
                </thead>
                <tbody>
                  {testFolders.map((tf, idx) => {
                    const ds = datasetByFolder.get(tf.input_folder);
                    const isRunning = loading && runProgress?.label === tf.label;
                    const isPending = loading && !ds && (runProgress?.current ?? 0) <= idx;
                    if (ds) {
                      return (
                        <TestDatasetRow
                          key={tf.input_folder}
                          dataset={ds}
                          runAlgos={displayAlgos}
                          algoLabels={algoLabels}
                          selected={selectedFolder === tf.input_folder}
                          onSelect={() => setSelectedFolder(tf.input_folder)}
                          statusLabel={isRunning ? "실행 중" : "완료"}
                        />
                      );
                    }
                    return (
                      <tr key={tf.input_folder} className="test-dataset-row test-dataset-pending">
                        <td>{isRunning ? "실행 중" : isPending ? "대기" : "—"}</td>
                        <td>{tf.label}</td>
                        <td><code className="test-folder-code">{tf.input_folder}</code></td>
                        <td colSpan={displayAlgos.length + 1} className="hint">—</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {selectedDataset && selectedEntries.length > 0 && (
            <div className="test-detail card-stagger">
              <section className="card test-detail-header">
                <h3>상세: {selectedDataset.label}</h3>
                <p className="hint">
                  <code>{selectedDataset.input_folder}</code>
                  {" · "}시뮬 종료 {selectedDataset.sim_end_minutes}분
                  {" · "}EQP {selectedDataset.eqp_ids.length} · 제품 {selectedDataset.prod_keys.length}
                </p>
                {selectedDataset.errors.length > 0 && (
                  <div className="banner banner-warn">
                    {selectedDataset.errors.map((e) => (
                      <span key={e.algorithm}> {e.algorithm}: {e.message}</span>
                    ))}
                  </div>
                )}
              </section>

              <ChartSettingsPanel
                dataTimeEnd={dataTimeEnd}
                ganttTimeFixed={ganttTimeFixed}
                ganttTimeStart={ganttTimeStart}
                ganttTimeEnd={ganttTimeEnd}
                onGanttFixedChange={(fixed) => {
                  setGanttTimeFixed(fixed);
                  if (fixed && dataTimeEnd > 0) {
                    setGanttTimeStart(0);
                    setGanttTimeEnd(dataTimeEnd);
                  }
                }}
                onGanttStartChange={setGanttTimeStart}
                onGanttEndChange={setGanttTimeEnd}
              />

              <div className="grid-2">
                <section className="card">
                  <h3>KPI 비교</h3>
                  <PlotChart {...buildAlgorithmKpiComparison(selectedEntries)} />
                </section>
                <section className="card">
                  <h3>달성률 비교</h3>
                  <PlotChart {...buildAlgorithmAchievementComparison(selectedEntries)} />
                </section>
              </div>

              <section className="card">
                <h3>알고리즘별 간트</h3>
                <PlotChart {...buildAlgorithmGanttComparison(selectedEntries, ganttAxis)} />
              </section>

              <section className="card">
                <h3>KPI 수치</h3>
                <div className="table-wrap">
                  <table className="compare-table">
                    <thead>
                      <tr>
                        <th>알고리즘</th>
                        <th>Makespan</th>
                        <th>Idle</th>
                        <th>공정 전환</th>
                        <th>제품 전환</th>
                        <th>평균 달성률</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedEntries.map((e) => {
                        const s = resultScheduleStats(e.result);
                        const ach = Object.values(s.achievement);
                        const avg = ach.length
                          ? Math.round((ach.reduce((a, b) => a + b, 0) / ach.length) * 10) / 10
                          : 0;
                        return (
                          <tr key={e.algorithm}>
                            <td>
                              <span
                                className="algo-color-dot"
                                style={{ background: ALGO_CHART_COLORS[e.algorithm] ?? "#888" }}
                              />
                              {e.label}
                            </td>
                            <td>{s.makespan}</td>
                            <td>{s.idle_total}</td>
                            <td>{s.oper_switches}</td>
                            <td>{s.prod_switches}</td>
                            <td>{avg}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          )}
        </div>
      )}

      {!hasResults && !loading && !loadingSaved && (
        <p className="hint">알고리즘을 선택하고 순차 실행을 시작하세요.</p>
      )}
    </div>
  );
}

function TestDatasetRow({
  dataset,
  runAlgos,
  algoLabels,
  selected,
  onSelect,
  statusLabel = "완료",
}: {
  dataset: TestBenchmarkDataset;
  runAlgos: AlgorithmId[];
  algoLabels: Record<string, string>;
  selected: boolean;
  onSelect: () => void;
  statusLabel?: string;
}) {
  if (dataset.error && !dataset.results.length) {
    return (
      <tr className={`test-dataset-row${selected ? " selected" : ""}`} onClick={onSelect}>
        <td>{statusLabel}</td>
        <td>{dataset.label}</td>
        <td colSpan={runAlgos.length + 2} className="test-dataset-error">{dataset.error}</td>
      </tr>
    );
  }

  const avgByAlgo = runAlgos.map((id) => {
    const r = dataset.results.find((x) => x.algorithm === id);
    if (!r) return "—";
    return resultScheduleStats(r).makespan;
  });

  const bestMakespan = Math.min(
    ...dataset.results.map((r) => resultScheduleStats(r).makespan).filter((v) => v > 0),
  );

  const allAch = dataset.results.flatMap((r) => Object.values(resultScheduleStats(r).achievement));
  const avgAch = allAch.length
    ? Math.round((allAch.reduce((a, b) => a + b, 0) / allAch.length) * 10) / 10
    : 0;

  return (
    <tr
      className={`test-dataset-row${selected ? " selected" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter") onSelect(); }}
    >
      <td>{statusLabel}</td>
      <td><strong>{dataset.label}</strong></td>
      <td><code className="test-folder-code">{dataset.input_folder}</code></td>
      {runAlgos.map((id, i) => {
        const val = avgByAlgo[i];
        const isBest = val !== "—" && val === bestMakespan;
        return (
          <td key={id} className={isBest ? "test-cell-best" : ""} title={`${algoLabels[id]} Makespan`}>
            {val}
          </td>
        );
      })}
      <td>{avgAch}%</td>
    </tr>
  );
}
