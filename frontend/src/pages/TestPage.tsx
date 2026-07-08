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
  buildAlgorithmGantt,
  buildMetricSummaryChart,
  buildMetricSummaryRows,
  buildTestMetricChart,
  TEST_METRICS,
  type AlgoCompareEntry,
  type TestChartType,
  type TestMetricKey,
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
  const [visibleMetrics, setVisibleMetrics] = useState<Set<TestMetricKey>>(() => new Set(TEST_METRICS.map(m => m.key)));
  const [periodChartType, setPeriodChartType] = useState<TestChartType>("line");
  const [rangeN, setRangeN] = useState("");

  const [ganttFixed, setGanttFixed] = useState(false);
  const [ganttStart, setGanttStart] = useState(0);
  const [ganttEnd, setGanttEnd]     = useState(1440);

  const [facIdOverride, setFacIdOverride] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate]     = useState("");
  const [prevcntStr, setPrevcntStr] = useState("");

  const algoList  = algorithms;
  const available = useMemo(() => algoList.filter(a => !a.requires_model || modelExists), [algoList, modelExists]);
  const algoLabels = useMemo(() => { const m: Record<string,string>={}; algoList.forEach(a=>m[a.id]=a.name); return m; }, [algoList]);
  const facId = facIdOverride.trim() || config?.fac_id || "FAC001";
  const prevcnt = prevcntStr.trim() ? Number(prevcntStr) : undefined;
  const hasRange = Boolean(fromDate.trim() && toDate.trim());
  const hasPrevcnt = prevcnt != null && !Number.isNaN(prevcnt);

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
    const filter = hasRange
      ? { from_date: fromDate.trim(), to_date: toDate.trim() }
      : hasPrevcnt
      ? { prevcnt }
      : undefined;
    Promise.all([
      api.getSavedTestBenchmark(fac).catch(() => null),
      api.getTestDatasets(fac, filter).catch(() => null),
    ]).then(([saved, folders]) => {
      if (saved) setBenchmark(saved);
      if (folders?.datasets) setTestFolders(folders.datasets);
    }).finally(() => setSaved(false));
  }, [config?.fac_id, facIdOverride, hasRange, fromDate, toDate, hasPrevcnt, prevcnt]);

  const chartRows = useMemo(() => benchmark?.datasets?.length ? benchmarkRowsFromResponse(benchmark.datasets, algoLabels) : [], [benchmark, algoLabels]);

  const rangedRows = useMemo(() => {
    const n = parseInt(rangeN, 10);
    if (!rangeN.trim() || Number.isNaN(n) || n <= 0) return chartRows;
    return chartRows.slice(-n);
  }, [chartRows, rangeN]);

  const summaryRows = useMemo(() => buildMetricSummaryRows(rangedRows, displayAlgos), [rangedRows, displayAlgos]);

  const visibleTestMetrics = useMemo(() => TEST_METRICS.filter(m => visibleMetrics.has(m.key)), [visibleMetrics]);

  const toggleMetric = useCallback((key: TestMetricKey) => {
    setVisibleMetrics(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const selectedDataset = useMemo(() => selected ? benchmark?.datasets?.find(d => d.input_folder === selected) ?? null : null, [selected, benchmark]);

  const detailEntries = useMemo((): AlgoCompareEntry[] =>
    (selectedDataset?.results ?? []).map(r => ({
      algorithm: r.algorithm ?? "scheduling_rl",
      label: algoLabels[r.algorithm ?? "scheduling_rl"] ?? (r.algorithm ?? ""),
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
          <p className="hint mb-2">CLI <code>test</code> 와 동일한 옵션입니다.</p>
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

          <label className="field-label mt-2" htmlFor="test-from">기간 (RULE_TIMEKEY)</label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              id="test-from"
              className="input"
              type="text"
              placeholder="시작"
              value={fromDate}
              onChange={e => setFromDate(e.target.value)}
              disabled={loading || hasPrevcnt}
            />
            <input
              id="test-to"
              className="input"
              type="text"
              placeholder="종료"
              value={toDate}
              onChange={e => setToDate(e.target.value)}
              disabled={loading || hasPrevcnt}
            />
          </div>

          <label className="field-label mt-2" htmlFor="test-prevcnt">PREVCNT (최근 N개)</label>
          <input
            id="test-prevcnt"
            className="input-number"
            type="number"
            min={1}
            placeholder="미지정 시 test 전체"
            value={prevcntStr}
            onChange={e => setPrevcntStr(e.target.value)}
            disabled={loading || hasRange}
          />
          <p className="hint mt-1">{testFolders.length}개 test 폴더 선택됨</p>
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

                <div className="card chart-controls mb-2">
                  <div className="chart-controls-row">
                    <span className="field-label">볼 지표</span>
                    <div className="chart-controls-metrics">
                      {TEST_METRICS.map(m => (
                        <label key={m.key} className="check-label chart-controls-metric">
                          <input type="checkbox" checked={visibleMetrics.has(m.key)} onChange={() => toggleMetric(m.key)} />
                          {m.label}
                        </label>
                      ))}
                    </div>
                  </div>
                  <div className="chart-controls-row">
                    {compareView === "period" && (
                      <div className="chart-controls-group">
                        <span className="field-label">차트 유형</span>
                        <div className="label-mode-group">
                          <label className={`label-pill${periodChartType === "line" ? " active" : ""}`}>
                            <input type="radio" name="period-chart-type" checked={periodChartType === "line"} onChange={() => setPeriodChartType("line")} />
                            선
                          </label>
                          <label className={`label-pill${periodChartType === "bar" ? " active" : ""}`}>
                            <input type="radio" name="period-chart-type" checked={periodChartType === "bar"} onChange={() => setPeriodChartType("bar")} />
                            막대
                          </label>
                        </div>
                      </div>
                    )}
                    <label className="field-label chart-controls-range">
                      범위 (최근 N개 기간, 비우면 전체)
                      <input
                        type="number" min={1} className="input-number" placeholder="전체"
                        value={rangeN} onChange={e => setRangeN(e.target.value)}
                      />
                    </label>
                  </div>
                </div>

                {compareView === "summary" && (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem" }}>
                    {summaryRows.filter(r => visibleMetrics.has(r.key)).map(r => {
                      const chart = buildMetricSummaryChart(r, displayAlgos, algoLabels);
                      return (
                        <FullscreenPanel key={r.key} title={r.label} className="card chart-wrap">
                          {chart ? <PlotChart {...chart} /> : <p className="hint chart-empty-hint">표시할 데이터가 없습니다.</p>}
                        </FullscreenPanel>
                      );
                    })}
                  </div>
                )}

                {compareView === "period" && (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem" }}>
                    {visibleTestMetrics.map(m => {
                      const chart = buildTestMetricChart(m, rangedRows, displayAlgos, algoLabels, selected ?? undefined, periodChartType);
                      return (
                        <FullscreenPanel key={m.key} title={m.label} className="card chart-wrap">
                          {chart ? <PlotChart {...chart} /> : <p className="hint chart-empty-hint">표시할 데이터가 없습니다.</p>}
                        </FullscreenPanel>
                      );
                    })}
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
                  <div className="gantt-compare-stack">
                    {detailEntries.map((entry) => {
                      const algoColor = ALGO_CHART_COLORS[entry.algorithm] ?? "var(--accent)";
                      return (
                        <section key={entry.algorithm} className="gantt-compare-section">
                          <div className="gantt-compare-section-head">
                            <span
                              className="gantt-algo-badge gantt-algo-badge--compare"
                              style={{
                                color: algoColor,
                                borderColor: `${algoColor}55`,
                                background: `${algoColor}18`,
                              }}
                            >
                              {entry.label}
                            </span>
                          </div>
                          <PlotChart {...buildAlgorithmGantt(entry, ganttAxis)} scrollable />
                        </section>
                      );
                    })}
                  </div>
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
                    <table className="data-table">
                      <thead><tr><th>알고리즘</th><th className="num">Makespan</th><th className="num">가동률</th><th className="num">유휴율</th><th className="num">공정전환</th><th className="num">제품전환</th><th className="num">Tool전환</th><th className="num">계획달성률</th><th className="num">타겟달성률</th></tr></thead>
                      <tbody>
                        {detailEntries.map(e => {
                          const k = computeInferenceKpi(e.result);
                          return (
                            <tr key={e.algorithm}>
                              <td style={{ fontWeight:700, color: ALGO_CHART_COLORS[e.algorithm] }}>{e.label}</td>
                              <td className="num">{k.makespan}분</td><td className="num">{k.avgUtilPct}%</td><td className="num">{k.avgIdlePct}%</td>
                              <td className="num">{k.operSwitches}회</td><td className="num">{k.prodSwitches}회</td><td className="num">{k.toolSwitches}회</td>
                              <td className="num" style={{ color: k.avgAchPct>=90?"var(--ok)":k.avgAchPct>=70?"var(--warn)":"var(--err)" }}>{k.avgAchPct}%</td>
                              <td className="num" style={{ color: k.avgTargetAchPct>=90?"var(--ok)":k.avgTargetAchPct>=70?"var(--warn)":"var(--err)" }}>{k.avgTargetAchPct}%</td>
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
