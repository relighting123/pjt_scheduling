import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlotChart from "../components/PlotChart";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import GanttKpiPanel from "../components/GanttKpiPanel";
import GanttLegendPanel from "../components/GanttLegendPanel";
import GanttSummaryPanel from "../components/GanttSummaryPanel";
import { EventTimeline } from "../components/EventTimeline";
import { api } from "../lib/api";
import { loadResultFromFile } from "../lib/resultFile";
import {
  buildEnhancedGantt,
  buildGanttLegendItems,
  buildProductProductionCharts,
  buildInferenceWipChart,
  buildAlgorithmKpiComparison,
  buildAlgorithmAchievementComparison,
  buildAlgorithmGanttComparison,
  buildCompareGanttAxis,
  type GanttBarLabel,
  type AlgoCompareEntry,
  ALGO_CHART_COLORS,
} from "../lib/charts";
import { buildEqpModelMap } from "../lib/metrics";
import { ruleTimekeyFromFolder, simBaseTimeFromRuleTimekey } from "../lib/ganttTime";
import type {
  AlgorithmCompareResponse, AlgorithmId, AlgorithmInfo,
  AppConfig, DataSummary, InferenceResult,
} from "../types";

interface Props {
  modelExists: boolean;
  config: AppConfig | null;
  summary: DataSummary | null;
  folderLoading: boolean;
  onInputFolderChange: (f: string) => void | Promise<void>;
}

const FALLBACK_ALGOS: AlgorithmInfo[] = [
  { id: "rl",          name: "PPO (강화학습)",        description: "", requires_model: true },
  { id: "minprogress", name: "Min-Progress (휴리스틱)", description: "", requires_model: false },
  { id: "earliest_st", name: "Earliest-ST (휴리스틱)", description: "", requires_model: false },
];

type MainTab = "gantt" | "events" | "table" | "compare";
const ROWS = 200;

function VirtualTable({ rows }: { rows: InferenceResult["schedule"] }) {
  const [page, setPage] = useState(0);
  const total = rows.length;
  const pages = Math.ceil(total / ROWS);
  const visible = rows.slice(page * ROWS, (page + 1) * ROWS);
  const cols = ["EQP_ID","LOT_ID","CARRIER_ID","PLAN_PROD_KEY","OPER_ID","ST","START_TM","END_TM","PROC_TIME","WF_QTY"] as const;

  return (
    <>
      <div className="vtable-header">
        <span className="vtable-count">총 {total.toLocaleString()}건</span>
        {pages > 1 && (
          <div className="vtable-pagination">
            <button type="button" className="btn btn-ghost btn-xs" disabled={page === 0} onClick={() => setPage(0)}>«</button>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page === 0} onClick={() => setPage(p => p - 1)}>‹</button>
            <span className="page-info">{page + 1} / {pages}</span>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page >= pages-1} onClick={() => setPage(p => p + 1)}>›</button>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page >= pages-1} onClick={() => setPage(pages-1)}>»</button>
          </div>
        )}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {visible.map((r, i) => (
              <tr key={`${r.EQP_ID}-${r.LOT_ID}-${r.START_TM}-${i}`}>
                <td>{r.EQP_ID}</td><td>{r.LOT_ID}</td><td>{r.CARRIER_ID ?? ""}</td>
                <td>{r.PLAN_PROD_KEY}</td><td>{r.OPER_ID ?? ""}</td><td>{r.ST ?? ""}</td>
                <td>{r.START_TM}</td><td>{r.END_TM}</td><td>{r.PROC_TIME ?? ""}</td><td>{r.WF_QTY ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function csv(sched: InferenceResult["schedule"], name: string) {
  const H = ["EQP_ID","LOT_ID","CARRIER_ID","PLAN_PROD_KEY","OPER_ID","ST","START_TM","END_TM","PROC_TIME","WF_QTY"];
  const body = sched.map(r => H.map(h => String(r[h as keyof typeof r] ?? "")).join(",")).join("\n");
  const blob = new Blob(["﻿"+H.join(",")+"\n"+body], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
  a.click(); URL.revokeObjectURL(a.href);
}

export default function InferencePage({ modelExists, config, summary, folderLoading, onInputFolderChange }: Props) {
  const [result, setResult]           = useState<InferenceResult | null>(null);
  const [compareData, setCompareData] = useState<AlgorithmCompareResponse | null>(null);
  const [algorithm, setAlgorithm]     = useState<AlgorithmId>("rl");
  const [algorithms, setAlgorithms]   = useState<AlgorithmInfo[]>([]);
  const [compareAlgos, setCompareAlgos] = useState<Set<AlgorithmId>>(new Set());
  const [loading, setLoading]         = useState(false);
  const [compareLoading, setCmpLoad]  = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [tab, setTab]                 = useState<MainTab>("gantt");
  const [fileSource, setFileSource]   = useState<string | null>(null);
  const fileRef                       = useRef<HTMLInputElement>(null);

  const [selectedFolder, setSelectedFolder] = useState("");
  const [decisionLog, setDecisionLog]       = useState(false);
  const [wipInflow, setWipInflow]           = useState(false);

  const [labelMode, setLabelMode]         = useState<GanttBarLabel>("lot");
  const [ganttFixed, setGanttFixed]       = useState(false);
  const [ganttStart, setGanttStart]       = useState(0);
  const [ganttEnd, setGanttEnd]           = useState(1440);
  const [compareShowGantt, setCompareShowGantt] = useState(true);
  const [hiddenLegendKeys, setHiddenLegendKeys] = useState<Set<string>>(new Set());
  const [showConversionBars, setShowConversionBars] = useState(true);

  const folders = useMemo(() =>
    config?.input_folders?.length ? config.input_folders : [],
  [config]);

  useEffect(() => {
    if (!folders.length) {
      setSelectedFolder("");
      return;
    }
    setSelectedFolder(prev => {
      if (prev && folders.includes(prev)) return prev;
      return folders.find(f => f.endsWith("/infer")) ?? folders[0];
    });
  }, [folders]);

  useEffect(() => {
    api.getAlgorithms().then(r => setAlgorithms(r.algorithms)).catch(() => {});
  }, []);

  const algoList = algorithms.length ? algorithms : FALLBACK_ALGOS;
  const available = useMemo(() => algoList.filter(a => !a.requires_model || modelExists), [algoList, modelExists]);

  useEffect(() => { setCompareAlgos(new Set(available.map(a => a.id))); }, [available]);

  const dataEnd = useMemo(() => result?.sim_end_minutes ?? compareData?.sim_end_minutes ?? 1440, [result, compareData]);
  useEffect(() => { if (dataEnd > 0 && !ganttFixed) setGanttEnd(dataEnd); }, [dataEnd, ganttFixed]);

  const simBaseTime = useMemo(() => {
    if (summary?.sim_base_time) return summary.sim_base_time;
    const rtk = ruleTimekeyFromFolder(selectedFolder);
    return rtk ? simBaseTimeFromRuleTimekey(rtk) : undefined;
  }, [summary, selectedFolder]);

  const axis = useMemo(() => ({
    eqpIds: result?.eqp_ids ?? compareData?.eqp_ids ?? [],
    timeStartMinutes: ganttFixed ? ganttStart : 0,
    timeEndMinutes:   ganttFixed ? ganttEnd   : dataEnd,
    fixedRange: ganttFixed,
    simBaseTime,
  }), [result, compareData, ganttFixed, ganttStart, ganttEnd, dataEnd, simBaseTime]);

  const eqpModelMap = useMemo(() => buildEqpModelMap(result?.event_log ?? []), [result]);

  useEffect(() => {
    setHiddenLegendKeys(new Set());
    setShowConversionBars(true);
  }, [result]);

  const legendItems = useMemo(() => {
    if (!result) return [];
    return buildGanttLegendItems(result.schedule, result.prod_keys, result.oper_ids);
  }, [result]);

  const toggleLegendKey = useCallback((pairKey: string) => {
    setHiddenLegendKeys((prev) => {
      const next = new Set(prev);
      if (next.has(pairKey)) next.delete(pairKey);
      else next.add(pairKey);
      return next;
    });
  }, []);

  const compareEntries = useMemo((): AlgoCompareEntry[] =>
    (compareData?.results ?? []).map(r => ({
      algorithm: r.algorithm ?? "rl",
      label: algoList.find(a => a.id === r.algorithm)?.name ?? (r.algorithm ?? ""),
      result: r,
    })),
  [compareData, algoList]);

  const setResultAndCompare = (res: InferenceResult) => {
    setResult(res);
    setCompareData({ results:[res], errors:[], plan:res.plan, prod_keys:res.prod_keys, oper_ids:res.oper_ids, eqp_ids:res.eqp_ids, sim_end_minutes:res.sim_end_minutes });
  };

  const runInference = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await api.runInference({ algorithm, input_folder: selectedFolder, decision_log: decisionLog, enable_wip_inflow: wipInflow, save_output: true });
      setResultAndCompare(res); setFileSource(null); setTab("gantt");
    } catch(e) { setError(e instanceof Error ? e.message : "추론 실패"); }
    finally { setLoading(false); }
  }, [algorithm, selectedFolder, decisionLog, wipInflow]);

  const loadSaved = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await api.getInferenceResult(selectedFolder);
      setResultAndCompare(res); setFileSource(null); setTab("gantt");
    } catch(e) { setError(e instanceof Error ? e.message : "저장 결과 없음"); }
    finally { setLoading(false); }
  }, [selectedFolder]);

  const loadFile = useCallback(async (file: File) => {
    setLoading(true); setError(null);
    try {
      const res = await loadResultFromFile(file);
      setResultAndCompare(res); setFileSource(file.name); setTab("gantt");
    } catch(e) { setError(e instanceof Error ? e.message : "파일 오류"); }
    finally { setLoading(false); }
  }, []);

  const runCompare = useCallback(async () => {
    const ids = [...compareAlgos];
    if (!ids.length) { setError("알고리즘을 선택하세요."); return; }
    setCmpLoad(true); setError(null);
    try {
      const res = await api.runCompare(ids, { input_folder: selectedFolder, enable_wip_inflow: wipInflow });
      setCompareData(res);
      if (res.results.length === 1) setResult(res.results[0]);
      setTab("compare");
    } catch(e) { setError(e instanceof Error ? e.message : "비교 실패"); }
    finally { setCmpLoad(false); }
  }, [compareAlgos, selectedFolder, wipInflow]);

  const needsModel = algoList.find(a => a.id === algorithm)?.requires_model ?? false;
  const canRun = !needsModel || modelExists;
  const hasResult = !!(result || compareData);

  const ganttChart = useMemo(() => {
    if (!result) return null;
    return buildEnhancedGantt(result.schedule, result.prod_keys, result.oper_ids, axis, {
      labelMode,
      eqpModelMap,
      conversionPlans: result.conversion_plans ?? [],
      hiddenProdOperKeys: hiddenLegendKeys,
      showConversion: showConversionBars,
    });
  }, [result, axis, labelMode, eqpModelMap, hiddenLegendKeys, showConversionBars]);

  const productionChart = useMemo(() => {
    if (!result?.schedule.length) return null;
    return buildProductProductionCharts(
      result.schedule,
      result.plan,
      result.prod_keys,
      result.sim_end_minutes,
      {
        operIds: result.oper_ids,
        timeAxis: {
          timeStartMinutes: axis.timeStartMinutes,
          timeEndMinutes: axis.timeEndMinutes,
          fixedRange: axis.fixedRange,
          simBaseTime: axis.simBaseTime,
        },
      },
    );
  }, [result, axis]);

  const wipChart = useMemo(() => {
    if (!result) return null;
    return buildInferenceWipChart(result.stats, result.plan);
  }, [result]);

  const compareGanttChart = useMemo(() => {
    if (!compareShowGantt || compareEntries.length < 1) return null;
    const compareAxis = buildCompareGanttAxis(compareEntries, compareData, simBaseTime);
    return buildAlgorithmGanttComparison(compareEntries, compareAxis);
  }, [compareShowGantt, compareEntries, compareData, simBaseTime]);

  const compareKpiChart = useMemo(() => {
    if (compareEntries.length < 1) return null;
    return buildAlgorithmKpiComparison(compareEntries);
  }, [compareEntries]);

  const compareAchievementChart = useMemo(() => {
    if (compareEntries.length < 1) return null;
    return buildAlgorithmAchievementComparison(compareEntries);
  }, [compareEntries]);

  const canShowCompare = compareEntries.length > 0;

  useEffect(() => {
    if (tab === "compare" && !canShowCompare) {
      setTab(result ? "gantt" : "table");
    }
  }, [tab, canShowCompare, result]);

  return (
    <div className="detail-page">
      <div className="detail-page-title">
        추론 결과
        <span className="page-badge badge badge-accent">Inference</span>
      </div>

      {/* ── Control panel ── */}
      <aside className="ctrl-panel">
        <div className="card">
          <div className="card-title">데이터셋</div>
          <label className="field-label" htmlFor="infer-folder">경로</label>
          <select id="infer-folder" className="select" value={selectedFolder}
            onChange={e => void (async () => {
              const f = e.target.value;
              if (f === selectedFolder) return;
              setSelectedFolder(f); setResult(null); setCompareData(null);
              await onInputFolderChange(f);
            })()}
            disabled={!config || folderLoading || !folders.length}
          >
            {folders.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          {summary && (
            <p className="hint mt-1">EQP {summary.eqp_count} · LOT {summary.lot_count} · 제품 {summary.prod_count}</p>
          )}
        </div>

        <div className="card">
          <div className="card-title">알고리즘</div>
          <div className="algo-list mb-2">
            {algoList.map(a => {
              const dis = a.requires_model && !modelExists;
              return (
                <label key={a.id} className={`algo-option${algorithm === a.id ? " selected" : ""}${dis ? "" : ""}`}>
                  <input type="radio" name="algo" disabled={dis} checked={algorithm === a.id} onChange={() => setAlgorithm(a.id)} />
                  <span className="algo-dot" style={{ background: ALGO_CHART_COLORS[a.id] ?? "#555" }} />
                  <span className={`algo-name${dis ? " algo-name-dim" : ""}`}>{a.name}{dis ? " (모델 없음)" : ""}</span>
                </label>
              );
            })}
          </div>

          <label className="check-label">
            <input type="checkbox" checked={decisionLog} onChange={e => setDecisionLog(e.target.checked)} disabled={loading} />
            결정 로그 포함
          </label>
          <label className="check-label">
            <input type="checkbox" checked={wipInflow} onChange={e => setWipInflow(e.target.checked)} disabled={loading} />
            유입 재공 이벤트
          </label>

          <div className="gap-row mt-2">
            <button type="button" className={`btn btn-primary${loading ? " loading" : ""}`}
              onClick={runInference} disabled={loading || compareLoading || !canRun || folderLoading || !selectedFolder}>
              {loading ? "" : "▶ 추론 실행"}
            </button>
            <button type="button" className={`btn btn-ghost btn-sm${loading ? " loading" : ""}`}
              onClick={loadSaved} disabled={loading || compareLoading || folderLoading || !selectedFolder}>
              저장 로드
            </button>
            <button type="button" className="btn btn-ghost btn-sm"
              onClick={() => fileRef.current?.click()} disabled={loading || compareLoading}>
              파일 열기
            </button>
            <input ref={fileRef} type="file" accept=".json" style={{ display:"none" }}
              onChange={e => { const f = e.target.files?.[0]; e.target.value=""; if(f) void loadFile(f); }} />
          </div>
          {fileSource && <p className="hint mt-1">파일: <code>{fileSource}</code></p>}
          {needsModel && !modelExists && <p className="hint mt-1" style={{ color:"var(--warn)" }}>⚠ PPO는 모델이 필요합니다</p>}
        </div>

        <div className="card">
          <div className="card-title">알고리즘 비교</div>
          <div className="algo-list mb-2">
            {algoList.map(a => {
              const dis = a.requires_model && !modelExists;
              return (
                <label key={a.id} className={`algo-option${compareAlgos.has(a.id) ? " selected" : ""}`}>
                  <input type="checkbox" disabled={dis || compareLoading} checked={compareAlgos.has(a.id)}
                    onChange={() => setCompareAlgos(prev => { const n = new Set(prev); n.has(a.id) ? n.delete(a.id) : n.add(a.id); return n; })} />
                  <span className="algo-dot" style={{ background: ALGO_CHART_COLORS[a.id] ?? "#555" }} />
                  <span className={`algo-name${dis ? " algo-name-dim" : ""}`}>{a.name}</span>
                </label>
              );
            })}
          </div>
          <label className="check-label">
            <input type="checkbox" checked={compareShowGantt} onChange={e => setCompareShowGantt(e.target.checked)} disabled={compareLoading} />
            비교 시 간트 차트 표시
          </label>
          <button type="button" className={`btn btn-accent${compareLoading ? " loading" : ""}`}
            onClick={runCompare} disabled={compareLoading || loading || compareAlgos.size === 0 || folderLoading || !selectedFolder}>
            {compareLoading ? "" : `비교 실행 (${compareAlgos.size}개)`}
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="content-area">
        {error && <ExpandableErrorBanner message={error} />}
        {compareData?.errors?.length ? (
          <div className="banner banner-warn">일부 알고리즘 실패: {compareData.errors.map(e => `${e.algorithm}: ${e.message}`).join(" / ")}</div>
        ) : null}

        {!hasResult && !loading && !compareLoading && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>좌측 패널에서 추론을 실행하거나 저장된 결과를 불러오세요.</p>
          </div>
        )}

        {hasResult && (
          <>
            <div className="tabs">
              <button type="button" className={`tab-btn${tab === "gantt" ? " active" : ""}`} onClick={() => setTab("gantt")} disabled={!result}>간트 차트</button>
              <button type="button" className={`tab-btn${tab === "events" ? " active" : ""}`} onClick={() => setTab("events")} disabled={!result?.event_log?.length}>이벤트 이력</button>
              <button type="button" className={`tab-btn${tab === "table" ? " active" : ""}`} onClick={() => setTab("table")} disabled={!result}>간트 테이블</button>
              {canShowCompare && (
                <button type="button" className={`tab-btn${tab === "compare" ? " active" : ""}`} onClick={() => setTab("compare")}>알고리즘 비교</button>
              )}
            </div>

            {/* GANTT TAB */}
            {tab === "gantt" && result && (
              <div className="tab-panel gantt-workspace">
                <div className="gantt-workspace-head">
                  <span className="gantt-algo-badge">
                    {algoList.find((a) => a.id === result.algorithm)?.name ?? result.algorithm ?? "—"}
                  </span>
                  <GanttKpiPanel result={result} eqpModelMap={eqpModelMap} />
                </div>

                <div className="gantt-toolbar">
                  <div className="gantt-toolbar-group">
                    <span className="gantt-toolbar-label">바 표시</span>
                    <div className="label-mode-group">
                      {(["lot","car","prod"] as GanttBarLabel[]).map(m => (
                        <label key={m} className={`label-pill${labelMode === m ? " active" : ""}`}>
                          <input type="radio" name="label-mode" value={m} checked={labelMode === m} onChange={() => setLabelMode(m)} />
                          {m === "lot" ? "LOT" : m === "car" ? "CAR" : "제품"}
                        </label>
                      ))}
                    </div>
                  </div>
                  <div className="gantt-toolbar-group time-range-group">
                    <label className="check-label">
                      <input type="checkbox" checked={ganttFixed} onChange={e => {
                        setGanttFixed(e.target.checked);
                        if (e.target.checked) { setGanttStart(0); setGanttEnd(dataEnd); }
                      }} />
                      X축 고정
                    </label>
                    {ganttFixed && (
                      <>
                        <label className="field-label gantt-time-field">
                          시작<input type="number" className="time-input" min={0} value={ganttStart} onChange={e => setGanttStart(Math.max(0, Number(e.target.value)))} />
                        </label>
                        <label className="field-label gantt-time-field">
                          종료<input type="number" className="time-input" value={ganttEnd} onChange={e => setGanttEnd(Number(e.target.value))} />
                        </label>
                      </>
                    )}
                  </div>
                </div>

                {ganttChart && (
                  <div className="chart-wrap gantt-chart-panel">
                    <PlotChart {...ganttChart} scrollable />
                  </div>
                )}

                {result && (legendItems.length > 0 || (result.conversion_plans?.length ?? 0) > 0) && (
                  <GanttLegendPanel
                    items={legendItems}
                    hiddenKeys={hiddenLegendKeys}
                    onToggle={toggleLegendKey}
                    onShowAll={() => setHiddenLegendKeys(new Set())}
                    onHideAll={() => setHiddenLegendKeys(new Set(legendItems.map((item) => item.pairKey)))}
                    showConversion={(result.conversion_plans?.length ?? 0) > 0}
                    conversionHidden={!showConversionBars}
                    onToggleConversion={() => setShowConversionBars((prev) => !prev)}
                  />
                )}

                {productionChart && (
                  <div className="card gantt-production-panel">
                    <div className="gantt-summary-section-title">시간별 제품·공정 누적 생산</div>
                    <p className="gantt-production-hint">
                      제품별 서브플롯에 공정(O) 실적(실선)과 계획(점선)을 표시합니다. X축은 간트 차트와 동일합니다.
                    </p>
                    <div className="chart-wrap gantt-production-chart">
                      <PlotChart {...productionChart} scrollable />
                    </div>
                  </div>
                )}

                {wipChart && (
                  <div className="card gantt-production-panel">
                    <div className="gantt-summary-section-title">재공 처리 현황</div>
                    <p className="gantt-production-hint">
                      제품/공정별 완료 수량, 잔여 재공(미처리 대기), 계획 미달을 표시합니다. 세로 점선이 계획 목표입니다.
                    </p>
                    <div className="chart-wrap">
                      <PlotChart {...wipChart} />
                    </div>
                  </div>
                )}

                <GanttSummaryPanel result={result} eqpModelMap={eqpModelMap} />
              </div>
            )}

            {/* EVENTS TAB */}
            {tab === "events" && result?.event_log && (
              <div className="tab-panel card">
                <div className="card-title">이벤트 이력 ({result.event_log.length.toLocaleString()}건)</div>
                <EventTimeline events={result.event_log} highlightKinds={new Set<string>()} title="" />
              </div>
            )}

            {/* TABLE TAB */}
            {tab === "table" && result && (
              <div className="tab-panel card">
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"0.75rem" }}>
                  <div className="card-title" style={{ marginBottom:0 }}>스케줄 테이블</div>
                  <button type="button" className="btn btn-ghost btn-sm"
                    onClick={() => csv(result.schedule, `schedule_${result.algorithm ?? "result"}.csv`)}>
                    CSV 다운로드
                  </button>
                </div>
                <VirtualTable rows={result.schedule} />
              </div>
            )}

            {/* COMPARE TAB */}
            {tab === "compare" && compareData && canShowCompare && (
              <div className="tab-panel">
                <div className="card mb-2">
                  <div className="card-title">알고리즘 KPI 요약</div>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>알고리즘</th><th>Makespan</th><th>Idle</th><th>공정전환</th><th>제품전환</th><th>평균달성률</th></tr></thead>
                      <tbody>
                        {compareEntries.map(e => {
                          const s = e.result.schedule;
                          const ms = s.length ? Math.max(...s.map(r => r.END_TM)) : 0;
                          const achV = e.result.plan.map(p => {
                            const done = s.filter(r => r.PLAN_PROD_KEY === p.plan_prod_key && r.OPER_ID === p.oper_id).reduce((a,r) => a+(r.WF_QTY??25),0);
                            return Math.min((done/Math.max(p.d0_plan_qty,1))*100,100);
                          });
                          const avg = achV.length ? Math.round(achV.reduce((a,b)=>a+b,0)/achV.length*10)/10 : 0;
                          return (
                            <tr key={e.algorithm}>
                              <td style={{ color: ALGO_CHART_COLORS[e.algorithm] ?? "inherit", fontFamily:"var(--font)", fontWeight:700 }}>{e.label}</td>
                              <td>{ms}분</td><td>{e.result.stats.idle_total}분</td>
                              <td>{e.result.stats.oper_switches}회</td><td>{e.result.stats.prod_switches}회</td>
                              <td style={{ color: avg>=90?"var(--ok)":avg>=70?"var(--warn)":"var(--err)" }}>{avg}%</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
                {compareKpiChart && (
                  <div className="grid-2 mb-2">
                    <div className="card chart-wrap compare-chart-panel">
                      <PlotChart {...compareKpiChart} />
                    </div>
                    {compareAchievementChart ? (
                      <div className="card chart-wrap compare-chart-panel">
                        <PlotChart {...compareAchievementChart} />
                      </div>
                    ) : (
                      <div className="card compare-chart-panel" style={{ display: "grid", placeItems: "center", minHeight: 380, color: "var(--text-muted)", fontSize: "0.85rem" }}>
                        달성률 비교 데이터 없음
                      </div>
                    )}
                  </div>
                )}
                {compareShowGantt && compareGanttChart && (
                  <div className="card chart-wrap gantt-chart-panel">
                    <PlotChart {...compareGanttChart} scrollable />
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
