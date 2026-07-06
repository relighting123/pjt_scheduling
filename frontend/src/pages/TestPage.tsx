import { useCallback, useEffect, useMemo, useState } from "react";
import PlotChart from "../components/PlotChart";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import FullscreenPanel from "../components/FullscreenPanel";
import { api } from "../lib/api";
import { downloadExcel } from "../lib/exportExcel";
import { ruleTimekeyFromFolder, simBaseTimeFromRuleTimekey } from "../lib/ganttTime";
import {
  ALGO_CHART_COLORS,
  benchmarkRowsFromResponse,
  buildAlgorithmAchievementComparison,
  buildAlgorithmGanttComparison,
  buildMetricSummaryChart,
  buildMetricSummaryRows,
  buildTestMetricChart,
  TEST_METRICS,
  type AlgoCompareEntry,
} from "../lib/charts";
import { computeInferenceKpi } from "../lib/metrics";
import type {
  AlgorithmId, AlgorithmInfo, AppConfig,
  TestBenchmarkResponse, TestDatasetInfo,
} from "../types";

interface Props { config: AppConfig | null; modelExists: boolean; }

type TestTab = "summary" | "gantt" | "detail";

export default function TestPage({ config, modelExists }: Props) {
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [compareAlgos, setCompareAlgos] = useState<Set<AlgorithmId>>(new Set());
  const [benchmark, setBenchmark] = useState<TestBenchmarkResponse | null>(null);
  const [testFolders, setTestFolders] = useState<TestDatasetInfo[]>([]);
  const [loading, setLoading]   = useState(false);
  const [savedLoading, setSaved] = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ current: number; total: number; label: string } | null>(null);
  const [tab, setTab]           = useState<TestTab>("summary");
  const [compareView, setCompareView] = useState<"summary" | "period">("summary");

  const [ganttFixed, setGanttFixed] = useState(false);
  const [ganttStart, setGanttStart] = useState(0);
  const [ganttEnd, setGanttEnd]     = useState(1440);

  const [facIdOverride, setFacIdOverride] = useState("");

  const algoList  = algorithms;
  const available = useMemo(() => algoList.filter(a => !a.requires_model || modelExists), [algoList, modelExists]);
  const algoLabels = useMemo(() => { const m: Record<string,string>={}; algoList.forEach(a=>m[a.id]=a.name); return m; }, [algoList]);
  const facId = facIdOverride.trim() || config?.fac_id || "FAC001";

  const displayAlgos = useMemo((): AlgorithmId[] => {
    if (benchmark?.algorithms?.length) return benchmark.algorithms;
    const seen = new Set<AlgorithmId>();
    benchmark?.datasets?.forEach(d => d.results.forEach(r => { if (r.algorithm) seen.add(r.algorithm as AlgorithmId); }));
    return seen.size ? [...seen] : available.map(a => a.id);
  }, [benchmark, available]);

  useEffect(() => { api.getAlgorithms().then(r => setAlgorithms(r.algorithms)).catch(() => {}); }, []);
  useEffect(() => { setCompareAlgos(new Set(available.map(a => a.id))); }, [available]);

  useEffect(() => {
    setSaved(true);
    const fac = facIdOverride.trim() || config?.fac_id;
    Promise.all([
      api.getSavedTestBenchmark(fac).catch(() => null),
      api.getTestDatasets(fac).catch(() => null),
    ]).then(([saved, folders]) => {
      if (saved) setBenchmark(saved);
      if (folders?.datasets) setTestFolders(folders.datasets);
    }).finally(() => setSaved(false));
  }, [config?.fac_id, facIdOverride]);

  const chartRows = useMemo(() => benchmark?.datasets?.length ? benchmarkRowsFromResponse(benchmark.datasets, algoLabels) : [], [benchmark, algoLabels]);

  const summaryRows = useMemo(() => buildMetricSummaryRows(chartRows, displayAlgos), [chartRows, displayAlgos]);

  const selectedDataset = useMemo(() => selected ? benchmark?.datasets?.find(d => d.input_folder === selected) ?? null : null, [selected, benchmark]);

  const detailEntries = useMemo((): AlgoCompareEntry[] =>
    (selectedDataset?.results ?? []).map(r => ({
      algorithm: r.algorithm ?? "rl",
      label: algoLabels[r.algorithm ?? "rl"] ?? (r.algorithm ?? ""),
      result: r,
    })),
  [selectedDataset, algoLabels]);

  const simBaseTime = useMemo(() => {
    const folder = selectedDataset?.input_folder;
    if (!folder) return undefined;
    const rtk = ruleTimekeyFromFolder(folder);
    return rtk ? simBaseTimeFromRuleTimekey(rtk) : undefined;
  }, [selectedDataset]);

  const ganttAxis = useMemo(() => ({
    eqpIds: selectedDataset?.eqp_ids ?? benchmark?.datasets?.[0]?.eqp_ids ?? [],
    timeStartMinutes: ganttFixed ? ganttStart : 0,
    timeEndMinutes:   ganttFixed ? ganttEnd   : (selectedDataset?.sim_end_minutes ?? 1440),
    fixedRange: ganttFixed,
    simBaseTime,
  }), [selectedDataset, benchmark, ganttFixed, ganttStart, ganttEnd, simBaseTime]);

  const run = useCallback(async () => {
    const ids = [...compareAlgos].filter(id => available.some(a => a.id === id));
    if (!ids.length) { setError("알고리즘을 선택하세요."); return; }
    setLoading(true); setError(null); setProgress(null);
    try {
      const datasets = testFolders.map(f => f.input_folder);
      if (!datasets.length) {
        setBenchmark(await api.runTestBenchmark(ids as AlgorithmId[], { fac_id: facId }));
        return;
      }
      let bm = await api.initTestBenchmark(ids as AlgorithmId[], facId);
      for (let i = 0; i < datasets.length; i++) {
        setProgress({ current: i+1, total: datasets.length, label: datasets[i] });
        bm = await api.runTestBenchmarkOne({ algorithms: ids as AlgorithmId[], input_folder: datasets[i], fac_id: facId, progress_current: i+1, progress_total: datasets.length, done: i === datasets.length-1 });
        setBenchmark(bm);
      }
    } catch(e) { setError(e instanceof Error ? e.message : "테스트 실패"); }
    finally { setLoading(false); setProgress(null); }
  }, [compareAlgos, available, facId, testFolders]);

  const clear = useCallback(async () => {
    try { await api.clearSavedTestBenchmark(facId); setBenchmark(null); } catch { /* ignore */ }
  }, [facId]);

  const hasData = !!(benchmark?.datasets?.length);

  return (
    <div className="detail-page">
      <div className="detail-page-title">
        테스트 셋 결과
        <span className="page-badge badge badge-info">Test Bench</span>
      </div>

      {/* ── Control panel ── */}
      <aside className="ctrl-panel">
        <div className="card">
          <div className="card-title">설정</div>
          <label className="field-label" htmlFor="test-fac-id">FAC_ID</label>
          <input
            id="test-fac-id"
            className="input"
            type="text"
            placeholder={config?.fac_id ?? "FAC001"}
            value={facIdOverride}
            onChange={e => setFacIdOverride(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="card">
          <div className="card-title">알고리즘</div>
          <div className="algo-list mb-2">
            {algoList.map(a => {
              const dis = a.requires_model && !modelExists;
              return (
                <label key={a.id} className={`algo-option${compareAlgos.has(a.id) ? " selected" : ""}`}>
                  <input type="checkbox" disabled={dis || loading} checked={compareAlgos.has(a.id)}
                    onChange={() => setCompareAlgos(prev => { const n = new Set(prev); n.has(a.id)?n.delete(a.id):n.add(a.id); return n; })} />
                  <span className="algo-dot" style={{ background: ALGO_CHART_COLORS[a.id] ?? "#555" }} />
                  <span className={`algo-name${dis ? " algo-name-dim" : ""}`}>{a.name}{dis?" (모델없음)":""}</span>
                </label>
              );
            })}
          </div>
          <div className="gap-row">
            <button type="button" className={`btn btn-primary${loading ? " loading" : ""}`} onClick={run} disabled={loading || compareAlgos.size === 0}>
              {loading ? "" : "테스트 실행"}
            </button>
            {hasData && (
              <button type="button" className="btn btn-ghost btn-sm" onClick={clear} disabled={loading}>초기화</button>
            )}
          </div>
          {loading && progress && (
            <div className="mt-2">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.round((progress.current/progress.total)*100)}%` }} />
              </div>
              <p className="hint mt-1">{progress.current}/{progress.total} · {progress.label}</p>
            </div>
          )}
        </div>

        {hasData && (
          <div className="card">
            <div className="card-title">데이터셋 선택</div>
            <div className="dataset-list">
              {benchmark?.datasets?.map(d => {
                const makespans = displayAlgos.map(a => {
                  const r = d.results.find(res => res.algorithm === a);
                  return r?.schedule?.length ? Math.max(...r.schedule.map(s => s.END_TM)) : null;
                });
                const minMs = Math.min(...makespans.filter((v): v is number => v !== null));
                return (
                  <div key={d.input_folder}
                    className={`dataset-row${selected === d.input_folder ? " selected" : ""}${!d.results.length ? " opacity-50" : ""}`}
                    onClick={() => setSelected(selected === d.input_folder ? null : d.input_folder)}
                  >
                    <div>
                      <div className="dataset-label">{d.label}</div>
                      <div className="dataset-folder">{d.input_folder}</div>
                    </div>
                    <div className="dataset-vals">
                      {displayAlgos.map((a, i) => (
                        <span key={a} className={`dataset-val${makespans[i] === minMs && makespans[i] !== null ? " best" : ""}`}>
                          {makespans[i] !== null ? `${makespans[i]}분` : "—"}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
            {benchmark?.updated_at && (
              <p className="hint mt-1">업데이트: {new Date(benchmark.updated_at).toLocaleString("ko-KR")}</p>
            )}
          </div>
        )}
      </aside>

      {/* ── Main content ── */}
      <div className="content-area">
        {error && <ExpandableErrorBanner message={error} />}
        {savedLoading && <div className="hint">저장 결과 불러오는 중...</div>}

        {!savedLoading && !hasData && !loading && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>테스트를 실행하면 알고리즘별 비교 결과가 여기에 표시됩니다.</p>
          </div>
        )}

        {hasData && (
          <>
            <div className="tabs">
              <button type="button" className={`tab-btn${tab === "summary" ? " active" : ""}`} onClick={() => setTab("summary")}>성능 비교</button>
              <button type="button" className={`tab-btn${tab === "gantt" ? " active" : ""}`} onClick={() => setTab("gantt")} disabled={!selectedDataset}>간트 비교</button>
              <button type="button" className={`tab-btn${tab === "detail" ? " active" : ""}`} onClick={() => setTab("detail")} disabled={!selectedDataset}>달성률</button>
            </div>

            {tab === "summary" && (
              <div className="tab-panel">
                <div className="tabs mb-2">
                  <button type="button" className={`tab-btn${compareView === "summary" ? " active" : ""}`} onClick={() => setCompareView("summary")}>요약</button>
                  <button type="button" className={`tab-btn${compareView === "period" ? " active" : ""}`} onClick={() => setCompareView("period")}>기간별</button>
                </div>

                {compareView === "summary" && (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem" }}>
                    {summaryRows.map(r => (
                      <FullscreenPanel key={r.key} title={r.label} className="card chart-wrap">
                        <PlotChart {...buildMetricSummaryChart(r, displayAlgos, algoLabels)} />
                      </FullscreenPanel>
                    ))}
                  </div>
                )}

                {compareView === "period" && (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem" }}>
                    {TEST_METRICS.map(m => (
                      <FullscreenPanel key={m.key} title={m.label} className="card chart-wrap">
                        <PlotChart {...buildTestMetricChart(m, chartRows, displayAlgos, algoLabels, selected ?? undefined)} />
                      </FullscreenPanel>
                    ))}
                  </div>
                )}
              </div>
            )}

            {tab === "gantt" && selectedDataset && detailEntries.length > 0 && (
              <div className="tab-panel gantt-workspace">
                <div className="gantt-workspace-head">
                  <span className="gantt-algo-badge">{selectedDataset.label}</span>
                </div>
                <div className="gantt-toolbar">
                  <div className="gantt-toolbar-group time-range-group" style={{ marginLeft: "auto" }}>
                    <label className="check-label">
                      <input type="checkbox" checked={ganttFixed} onChange={e => { setGanttFixed(e.target.checked); if(e.target.checked){setGanttStart(0);setGanttEnd(selectedDataset.sim_end_minutes);} }} />
                      X축 고정
                    </label>
                    {ganttFixed && <>
                      <label className="field-label gantt-time-field">시작<input type="number" className="time-input" min={0} value={ganttStart} onChange={e=>setGanttStart(Math.max(0, Number(e.target.value)))} /></label>
                      <label className="field-label gantt-time-field">종료<input type="number" className="time-input" value={ganttEnd} onChange={e=>setGanttEnd(Number(e.target.value))} /></label>
                    </>}
                  </div>
                </div>
                <FullscreenPanel title={selectedDataset.label} className="chart-wrap gantt-chart-panel">
                  <PlotChart {...buildAlgorithmGanttComparison(detailEntries, ganttAxis)} scrollable />
                </FullscreenPanel>
              </div>
            )}

            {tab === "detail" && selectedDataset && detailEntries.length > 0 && (
              <div className="tab-panel">
                <FullscreenPanel
                  title={`KPI 비교 — ${selectedDataset.label}`}
                  className="card mb-2"
                  actions={
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => {
                      const H = ["알고리즘","Makespan(분)","가동률(%)","유휴율(%)","공정전환","제품전환","Tool전환","계획달성률(%)","타겟달성률(%)"];
                      const rows = detailEntries.map(e => {
                        const k = computeInferenceKpi(e.result);
                        return [e.label, k.makespan, k.avgUtilPct, k.avgIdlePct, k.operSwitches, k.prodSwitches, k.toolSwitches, k.avgAchPct, k.avgTargetAchPct];
                      });
                      downloadExcel(`kpi_compare_${selectedDataset.label}.xls`, H, rows);
                    }}>
                      엑셀 다운로드
                    </button>
                  }
                >
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>알고리즘</th><th>Makespan</th><th>가동률</th><th>유휴율</th><th>공정전환</th><th>제품전환</th><th>Tool전환</th><th>계획달성률</th><th>타겟달성률</th></tr></thead>
                      <tbody>
                        {detailEntries.map(e => {
                          const k = computeInferenceKpi(e.result);
                          return (
                            <tr key={e.algorithm}>
                              <td style={{ fontFamily:"var(--font)", fontWeight:700, color: ALGO_CHART_COLORS[e.algorithm] }}>{e.label}</td>
                              <td>{k.makespan}분</td><td>{k.avgUtilPct}%</td><td>{k.avgIdlePct}%</td>
                              <td>{k.operSwitches}회</td><td>{k.prodSwitches}회</td><td>{k.toolSwitches}회</td>
                              <td style={{ color: k.avgAchPct>=90?"var(--ok)":k.avgAchPct>=70?"var(--warn)":"var(--err)" }}>{k.avgAchPct}%</td>
                              <td style={{ color: k.avgTargetAchPct>=90?"var(--ok)":k.avgTargetAchPct>=70?"var(--warn)":"var(--err)" }}>{k.avgTargetAchPct}%</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </FullscreenPanel>
                {(() => {
                  const ach = buildAlgorithmAchievementComparison(detailEntries);
                  return ach ? (
                    <FullscreenPanel title="달성률 비교" className="chart-wrap">
                      <PlotChart {...ach} />
                    </FullscreenPanel>
                  ) : null;
                })()}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
