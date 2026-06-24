import { useCallback, useEffect, useMemo, useState } from "react";

import PlotChart from "../components/PlotChart";

import ChartPanel from "../components/ChartPanel";

import ChartSettingsPanel from "../components/ChartSettingsPanel";

import ChartVisibilityPanel from "../components/ChartVisibilityPanel";

import { useChartVisibility, type ChartVisibilityItem } from "../hooks/useChartVisibility";

import ArrangeTable from "../components/ArrangeTable";

import AbstractArrangeTable from "../components/AbstractArrangeTable";

import { EventTimeline, eventsUpToStep } from "../components/EventTimeline";

import BatchInfoTable from "../components/BatchInfoTable";

import DecisionLogTable from "../components/DecisionLogTable";

import { api } from "../lib/api";

import {

  buildAchievementChart,

  buildAlgorithmAchievementComparison,

  buildAlgorithmGanttComparison,

  buildAlgorithmKpiComparison,

  buildProductProductionCharts,

  buildStepGantt,

  buildSwitchMetrics,

  buildWipChart,

  resultScheduleStats,

  type AlgoCompareEntry,

} from "../lib/charts";

import type {

  AlgorithmCompareResponse,

  AlgorithmId,

  AlgorithmInfo,

  AppConfig,

  DataSummary,

  InferenceResult,

} from "../types";



interface InferencePageProps {

  modelExists: boolean;

  config: AppConfig | null;

  summary: DataSummary | null;

  folderLoading: boolean;

  onInputFolderChange: (folder: string) => void | Promise<void>;

}



function folderPeriodLabel(folder: string): string {

  const parts = folder.split("/");

  return parts[parts.length - 1] ?? folder;

}

function isAlgorithmId(value: unknown): value is AlgorithmId {
  return value === "rl" || value === "minprogress" || value === "earliest_st";
}



type TabId = "sim" | "schedule";



const FALLBACK_ALGOS: AlgorithmInfo[] = [

  { id: "rl", name: "PPO (강화학습)", description: "", requires_model: true },

  { id: "minprogress", name: "Min-Progress (휴리스틱)", description: "", requires_model: false },

  { id: "earliest_st", name: "Earliest-ST (휴리스틱)", description: "", requires_model: false },

];



export default function InferencePage({

  modelExists,

  config,

  summary,

  folderLoading,

  onInputFolderChange,

}: InferencePageProps) {

  const [result, setResult] = useState<InferenceResult | null>(null);

  const [compareData, setCompareData] = useState<AlgorithmCompareResponse | null>(null);

  const [step, setStep] = useState(0);

  const [tab, setTab] = useState<TabId>("schedule");

  const [loading, setLoading] = useState(false);

  const [compareLoading, setCompareLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const [tableAlgo, setTableAlgo] = useState<AlgorithmId | null>(null);

  const [algorithm, setAlgorithm] = useState<AlgorithmId>("rl");

  const [decisionLogEnabled, setDecisionLogEnabled] = useState(false);
  const [includeHistory, setIncludeHistory] = useState(false);
  const [enableWipInflow, setEnableWipInflow] = useState(false);

  const [compareAlgos, setCompareAlgos] = useState<Set<AlgorithmId>>(new Set());

  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);

  const [stepBump, setStepBump] = useState(false);

  const [selectedFolder, setSelectedFolder] = useState("FAC001/infer");

  const datasetFolders = useMemo(
    () =>
      (config?.input_folders?.length ? config.input_folders : config ? [config.input_folder] : []),
    [config],
  );

  const [ganttTimeFixed, setGanttTimeFixed] = useState(false);

  const [ganttTimeStart, setGanttTimeStart] = useState(0);

  const [ganttTimeEnd, setGanttTimeEnd] = useState(1440);



  useEffect(() => {

    if (!config || datasetFolders.length === 0) return;

    setSelectedFolder((prev) => {

      if (datasetFolders.includes(prev)) return prev;

      const infer = datasetFolders.find((f) => f.endsWith("/infer"));

      return infer ?? datasetFolders[0];

    });

  }, [config, datasetFolders]);



  const handleDatasetChange = useCallback(

    async (folder: string) => {

      if (!folder || folder === selectedFolder) return;

      setSelectedFolder(folder);

      setResult(null);

      setCompareData(null);

      await onInputFolderChange(folder);

    },

    [onInputFolderChange, selectedFolder],

  );



  useEffect(() => {

    setStepBump(true);

    const t = window.setTimeout(() => setStepBump(false), 350);

    return () => window.clearTimeout(t);

  }, [step]);



  useEffect(() => {

    api.getAlgorithms().then((res) => setAlgorithms(res.algorithms)).catch(() => {});

  }, []);



  const algoList = algorithms.length ? algorithms : FALLBACK_ALGOS;



  const availableAlgos = useMemo(

    () => algoList.filter((a) => !a.requires_model || modelExists),

    [algoList, modelExists],

  );



  useEffect(() => {

    setCompareAlgos(new Set(availableAlgos.map((a) => a.id)));

  }, [availableAlgos]);



  const selectedAlgo = algoList.find((a) => a.id === algorithm);

  const needsModel = selectedAlgo?.requires_model ?? algorithm === "rl";

  const canRun = !needsModel || modelExists;



  const runInference = useCallback(async () => {

    setLoading(true);

    setError(null);

    try {
      const withReplayPayload = includeHistory || decisionLogEnabled;

      const res = await api.runInference({
        algorithm,
        input_folder: selectedFolder,
        decision_log: decisionLogEnabled,
        include_history: withReplayPayload,
        enable_wip_inflow: enableWipInflow,
      });

      setResult(res);
      if (withReplayPayload) {
        setCompareData(null);
      } else {
        setCompareData({
          results: [res],
          errors: [],
          plan: res.plan,
          prod_keys: res.prod_keys,
          oper_ids: res.oper_ids,
          eqp_ids: res.eqp_ids,
          sim_end_minutes: res.sim_end_minutes,
        });
      }

      setStep(0);

      setTab(withReplayPayload ? "sim" : "schedule");

    } catch (e) {

      setError(e instanceof Error ? e.message : "추론 실패");

    } finally {

      setLoading(false);

    }

  }, [algorithm, selectedFolder, decisionLogEnabled, includeHistory, enableWipInflow]);



  const loadSavedResult = useCallback(async () => {

    setLoading(true);

    setError(null);

    try {

      const res = await api.getInferenceResult(selectedFolder);

      setResult(res);

      setCompareData({
        results: [res],
        errors: [],
        plan: res.plan,
        prod_keys: res.prod_keys,
        oper_ids: res.oper_ids,
        eqp_ids: res.eqp_ids,
        sim_end_minutes: res.sim_end_minutes,
      });

      if (isAlgorithmId(res.algorithm)) {
        setAlgorithm(res.algorithm);
      }

      setStep(0);

      setTab("schedule");

    } catch (e) {

      setError(e instanceof Error ? e.message : "저장된 결과 불러오기 실패");

    } finally {

      setLoading(false);

    }

  }, [selectedFolder]);



  const runCompare = useCallback(async (algoIds?: AlgorithmId[]) => {

    const ids = algoIds ?? [...compareAlgos];

    if (!ids.length) {

      setError("비교할 알고리즘을 하나 이상 선택하세요.");

      return;

    }

    setCompareLoading(true);

    setError(null);

    try {

      const res = await api.runCompare(ids, {
        input_folder: selectedFolder,
        decision_log: decisionLogEnabled,
        include_history: false,
        enable_wip_inflow: enableWipInflow,
      });

      setCompareData(res);

      setTab("schedule");

    } catch (e) {

      setError(e instanceof Error ? e.message : "스케줄링 결과 비교 실패");

    } finally {

      setCompareLoading(false);

    }

  }, [compareAlgos, selectedFolder, decisionLogEnabled, enableWipInflow]);



  const runCompareAll = () => {

    const ids = availableAlgos.map((a) => a.id);

    setCompareAlgos(new Set(ids));

    return runCompare(ids);

  };



  const toggleCompareAlgo = (id: AlgorithmId) => {

    setCompareAlgos((prev) => {

      const next = new Set(prev);

      if (next.has(id)) next.delete(id);

      else next.add(id);

      return next;

    });

  };



  const openInSim = (res: InferenceResult) => {

    setResult(res);

    if (isAlgorithmId(res.algorithm)) setAlgorithm(res.algorithm);

    setStep(0);

    setTab("sim");

  };



  const snap = useMemo(() => {

    if (!result?.history.length) return null;

    const idx = Math.min(step, result.history.length - 1);

    return result.history[idx];

  }, [result, step]);



  const { timelineEvents, currentStepEventKinds, currentStepEvents } = useMemo(() => {

    if (!result?.history.length) {

      return {
        timelineEvents: [],
        currentStepEventKinds: new Set<string>(),
        currentStepEvents: [],
      };

    }

    const { cumulative, current } = eventsUpToStep(

      result.event_log,

      result.history,

      step,

    );

    const kinds = new Set(current.map((e) => e.kind));

    return {
      timelineEvents: cumulative,
      currentStepEventKinds: kinds,
      currentStepEvents: current,
    };

  }, [result, step]);



  const maxStep = result?.history.length ? result.history.length - 1 : 0;



  const dataTimeEnd = useMemo(

    () => result?.sim_end_minutes ?? compareData?.sim_end_minutes ?? 0,

    [result, compareData],

  );



  useEffect(() => {

    if (dataTimeEnd > 0 && !ganttTimeFixed) {

      setGanttTimeEnd(dataTimeEnd);

    }

  }, [dataTimeEnd, ganttTimeFixed]);



  const ganttAxis = useMemo(

    () => ({

      eqpIds: result?.eqp_ids ?? compareData?.eqp_ids ?? [],

      timeStartMinutes: ganttTimeFixed ? ganttTimeStart : 0,

      timeEndMinutes: ganttTimeFixed ? ganttTimeEnd : dataTimeEnd,

      fixedRange: ganttTimeFixed,

    }),

    [result, compareData, ganttTimeFixed, ganttTimeStart, ganttTimeEnd, dataTimeEnd],

  );



  const algorithmLabel = useMemo(() => {

    const id = result?.algorithm ?? algorithm;

    return algoList.find((a) => a.id === id)?.name ?? id;

  }, [result?.algorithm, algorithm, algoList]);



  const compareEntries = useMemo((): AlgoCompareEntry[] => {

    if (!compareData) return [];

    return compareData.results.map((r) => ({

      algorithm: r.algorithm ?? "rl",

      label: algoList.find((a) => a.id === r.algorithm)?.name ?? (r.algorithm ?? ""),

      result: r,

    }));

  }, [compareData, algoList]);



  const compareChartItems = useMemo((): ChartVisibilityItem[] => {

    const items: ChartVisibilityItem[] = [

      { id: "compare-kpi", title: "KPI 차트" },

      { id: "compare-achievement", title: "달성률 차트" },

      { id: "compare-gantt", title: "알고리즘별 간트 비교" },

    ];

    if (compareEntries.length > 0 && compareData && compareData.plan.length > 0) {

      compareEntries.forEach((e) => {

        items.push({

          id: `compare-prod-${e.algorithm}`,

          title: `누적 생산 – ${e.label}`,

        });

      });

    }

    return items;

  }, [compareEntries, compareData]);



  const simChartItems = useMemo(

    (): ChartVisibilityItem[] => [

      { id: "sim-gantt", title: "설비(EQP) 배정 현황" },

      { id: "sim-wip", title: "WIP 수량 현황" },

      { id: "sim-achievement", title: "계획 달성 현황" },

      { id: "sim-prod", title: "제품별 누적 생산량" },

      { id: "sim-switch", title: "전환 횟수" },

    ],

    [],

  );



  const compareCharts = useChartVisibility(

    "inference-compare",

    compareChartItems.map((c) => c.id),

  );

  const simCharts = useChartVisibility(

    "inference-sim",

    simChartItems.map((c) => c.id),

  );



  const tableResult = useMemo(() => {

    if (!tableAlgo || !compareData) return null;

    return compareData.results.find((r) => r.algorithm === tableAlgo) ?? null;

  }, [tableAlgo, compareData]);



  const downloadCsv = (sched: InferenceResult["schedule"], filename: string) => {

    const headers = ["EQP_ID", "LOT_ID", "PLAN_PROD_KEY", "OPER_ID", "START_TM", "END_TM"];

    const rows = sched.map((r) =>

      headers.map((h) => String(r[h as keyof typeof r] ?? "")).join(","),

    );

    const csv = [headers.join(","), ...rows].join("\n");

    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = filename;

    a.click();

    URL.revokeObjectURL(url);

  };



  return (

    <div className="page">

      <h2>Scheduling 추론 및 시각화</h2>

      <section className="card inference-dataset-card">

        <h3>추론 데이터셋</h3>

        <p className="hint">추론·비교에 사용할 입력 JSON 경로를 선택합니다.</p>

        <label className="field-label" htmlFor="inference-dataset">

          dataset 경로

        </label>

        <select

          id="inference-dataset"

          className="input-select inference-dataset-select"

          value={selectedFolder}

          onChange={(e) => void handleDatasetChange(e.target.value)}

          disabled={!config || folderLoading || datasetFolders.length === 0}

        >

          {datasetFolders.map((f) => (

            <option key={f} value={f}>

              {f} ({folderPeriodLabel(f)})

            </option>

          ))}

        </select>

        {summary && (

          <p className="hint dataset-summary">

            EQP {summary.eqp_count} · LOT {summary.lot_count} · 제품 {summary.prod_count} · 공정{" "}

            {summary.oper_count}
            {(summary.batch_info_count ?? 0) > 0 && <> · batch {summary.batch_info_count}</>}

            {folderLoading ? " · 불러오는 중…" : ""}

          </p>

        )}

      </section>

      {summary && (
        <section className="card">
          <h3>Batch Info (PPK × OPER → LOT_CD / TEMP)</h3>
          <BatchInfoTable rows={summary.batch_info ?? []} />
        </section>
      )}

      <section className="card">

        <h3>알고리즘 선택 (단일 추론 · 시뮬레이션 재생용)</h3>

        <div className="algo-group">

          {algoList.map((algo) => {

            const disabled = algo.requires_model && !modelExists;

            return (

              <label key={algo.id} className={`algo-option${disabled ? " disabled" : ""}`}>

                <input

                  type="radio"

                  name="algorithm"

                  checked={algorithm === algo.id}

                  disabled={disabled}

                  onChange={() => setAlgorithm(algo.id)}

                />

                <span className="algo-name">{algo.name}</span>

                {algo.description && (

                  <span className="algo-desc">{algo.description}</span>

                )}

                {disabled && <span className="algo-desc"> (모델 없음)</span>}

              </label>

            );

          })}

        </div>

        {needsModel && !modelExists && (

          <p className="status-warn">

            PPO 알고리즘은 학습된 모델이 필요합니다.

          </p>

        )}

        <label className="inference-option inference-option-log">

          <input

            type="checkbox"

            checked={decisionLogEnabled}

            onChange={(e) => setDecisionLogEnabled(e.target.checked)}

            disabled={loading || compareLoading}

          />

          <span>

            <strong>결정 로그</strong>

            <span className="hint inference-option-hint">

              step별 대상 EQP, 선택 PPK/OPER, feasible 후보, 미할당 사유 기록

            </span>

          </span>

        </label>
        <label className="inference-option inference-option-log">

          <input

            type="checkbox"

            checked={includeHistory}

            onChange={(e) => setIncludeHistory(e.target.checked)}

            disabled={loading || compareLoading}

          />

          <span>

            <strong>시뮬레이션 재생 데이터 포함</strong>

            <span className="hint inference-option-hint">

              OFF이면 history/event payload 생성을 생략해 추론 결과 표시가 빨라집니다.

            </span>

          </span>

        </label>
        <label className="inference-option inference-option-log">

          <input

            type="checkbox"

            checked={enableWipInflow}

            onChange={(e) => setEnableWipInflow(e.target.checked)}

            disabled={loading || compareLoading}

          />

          <span>

            <strong>유입 재공 이벤트 사용</strong>

            <span className="hint inference-option-hint">

              ON이면 공정 완료 후 다음 공정 flow 재공을 유입해 계속 배정합니다.

            </span>

          </span>

        </label>

        <div className="btn-row">

          <button

            type="button"

            className={`btn btn-secondary${loading ? " is-loading" : ""}`}

            onClick={runInference}

            disabled={loading || compareLoading || !canRun || folderLoading}

          >

            {loading ? "추론 진행 중..." : "단일 추론 실행"}

          </button>
          <button

            type="button"

            className={`btn btn-secondary${loading ? " is-loading" : ""}`}

            onClick={loadSavedResult}

            disabled={loading || compareLoading || folderLoading}

          >

            저장 결과 불러오기

          </button>

        </div>

      </section>



      <section className="card">

        <h3>스케줄링 결과 비교</h3>

        <p className="hint">

          선택한 데이터셋으로 알고리즘별 스케줄·KPI·간트를 한 화면에서 비교합니다.

        </p>

        <div className="algo-check-group">

          {algoList.map((algo) => {

            const disabled = algo.requires_model && !modelExists;

            return (

              <label key={algo.id} className={`algo-check${disabled ? " disabled" : ""}`}>

                <input

                  type="checkbox"

                  checked={compareAlgos.has(algo.id)}

                  disabled={disabled || compareLoading}

                  onChange={() => toggleCompareAlgo(algo.id)}

                />

                <span>{algo.name}</span>

                {disabled && <span className="algo-desc"> (모델 없음)</span>}

              </label>

            );

          })}

        </div>

        <div className="btn-row">

          <button

            type="button"

            className={`btn btn-primary${compareLoading ? " is-loading" : ""}`}

            onClick={() => runCompare()}

            disabled={compareLoading || loading || compareAlgos.size === 0 || folderLoading}

          >

            {compareLoading ? "비교 실행 중..." : `선택 알고리즘 비교 (${compareAlgos.size}개)`}

          </button>

          <button

            type="button"

            className={`btn btn-secondary${compareLoading ? " is-loading" : ""}`}

            onClick={runCompareAll}

            disabled={compareLoading || loading || availableAlgos.length === 0 || folderLoading}

          >

            전체 알고리즘 비교

          </button>

        </div>

      </section>



      {error && <p className="error">{error}</p>}



      {!result && !compareData && !loading && !compareLoading && (

        <p className="hint">단일 추론 또는 스케줄링 결과 비교를 실행하세요.</p>

      )}



      {compareData && compareData.errors.length > 0 && (

        <div className="banner banner-warn">

          일부 알고리즘 실행 실패:

          {compareData.errors.map((e) => (

            <span key={e.algorithm}> {e.algorithm}: {e.message}</span>

          ))}

        </div>

      )}



      {(result || compareData) && (

        <div className="result-block">

          {result && tab === "sim" && (

            <p className="result-meta">

              시뮬레이션: <strong>{algorithmLabel}</strong> · 데이터셋 <code>{selectedFolder}</code>

            </p>

          )}

          {compareData && tab === "schedule" && (

            <p className="result-meta">

              스케줄링 결과 비교 · <strong>{compareEntries.length}개 알고리즘</strong> · 데이터셋 <code>{selectedFolder}</code>

            </p>

          )}



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

            showReplay={tab === "sim" && !!result?.history.length}

            step={step}

            maxStep={maxStep}

            stepBump={stepBump}

            stepSimTime={snap?.time}

            stepEvents={currentStepEvents}

            onStepChange={setStep}

          />



          {tab === "schedule" && compareData && compareEntries.length > 0 && (

            <ChartVisibilityPanel

              title="스케줄 비교 차트 표시"

              charts={compareChartItems}

              visibility={compareCharts.visibility}

              onToggle={compareCharts.toggle}

              onShowAll={compareCharts.showAll}

              onHideAll={compareCharts.hideAll}

            />

          )}



          {tab === "sim" && result?.history.length ? (

            <ChartVisibilityPanel

              title="시뮬레이션 차트 표시"

              charts={simChartItems}

              visibility={simCharts.visibility}

              onToggle={simCharts.toggle}

              onShowAll={simCharts.showAll}

              onHideAll={simCharts.hideAll}

            />

          ) : null}



          <div className="tabs">

            <button

              type="button"

              className={tab === "schedule" ? "tab active" : "tab"}

              onClick={() => setTab("schedule")}

              disabled={!compareData}

            >

              스케줄링 결과 비교

            </button>

            <button

              type="button"

              className={tab === "sim" ? "tab active" : "tab"}

              onClick={() => setTab("sim")}

              disabled={!result}

            >

              시뮬레이션 재생

            </button>

          </div>



          {tab === "schedule" && compareData && compareEntries.length > 0 && (

            <div className="tab-panel card-stagger" key="schedule">

              <section className="card">

                <h3>알고리즘별 KPI 요약</h3>

                <div className="table-wrap">

                  <table className="compare-table">

                    <thead>

                      <tr>

                        <th>알고리즘</th>

                        <th>Makespan</th>

                        <th>Idle 합계</th>

                        <th>공정 전환</th>

                        <th>제품 전환</th>

                        <th>평균 달성률</th>

                        <th />

                      </tr>

                    </thead>

                    <tbody>

                      {compareEntries.map((e) => {

                        const s = resultScheduleStats(e.result);

                        const achVals = Object.values(s.achievement);

                        const avgAch = achVals.length

                          ? Math.round((achVals.reduce((a, b) => a + b, 0) / achVals.length) * 10) / 10

                          : 0;

                        return (

                          <tr key={e.algorithm}>

                            <td><strong>{e.label}</strong></td>

                            <td>{s.makespan}</td>

                            <td>{s.idle_total}</td>

                            <td>{s.oper_switches}</td>

                            <td>{s.prod_switches}</td>

                            <td>{avgAch}%</td>

                            <td className="btn-cell">

                              <button

                                type="button"

                                className="btn btn-secondary btn-sm"

                                onClick={() => openInSim(e.result)}

                              >

                                시뮬레이션

                              </button>

                              <button

                                type="button"

                                className="btn btn-secondary btn-sm"

                                onClick={() => setTableAlgo(

                                  tableAlgo === e.algorithm ? null : (e.algorithm as AlgorithmId),

                                )}

                              >

                                {tableAlgo === e.algorithm ? "테이블 닫기" : "스케줄 표"}

                              </button>

                            </td>

                          </tr>

                        );

                      })}

                    </tbody>

                  </table>

                </div>

              </section>



              <div className="grid-2">

                <ChartPanel

                  id="compare-kpi"

                  title="KPI 차트"

                  visible={compareCharts.isVisible("compare-kpi")}

                  onVisibleChange={(v) => compareCharts.setVisible("compare-kpi", v)}

                  renderChart={() => <PlotChart {...buildAlgorithmKpiComparison(compareEntries)} />}

                />

                <ChartPanel

                  id="compare-achievement"

                  title="달성률 차트"

                  visible={compareCharts.isVisible("compare-achievement")}

                  onVisibleChange={(v) => compareCharts.setVisible("compare-achievement", v)}

                  renderChart={() => (

                    <PlotChart {...buildAlgorithmAchievementComparison(compareEntries)} />

                  )}

                />

              </div>



              <ChartPanel

                id="compare-gantt"

                title="알고리즘별 간트 비교"

                visible={compareCharts.isVisible("compare-gantt")}

                onVisibleChange={(v) => compareCharts.setVisible("compare-gantt", v)}

                renderChart={() => (

                  <PlotChart

                    className="plot-chart gantt-chart"

                    {...buildAlgorithmGanttComparison(compareEntries, ganttAxis)}

                  />

                )}

              />



              {compareEntries.length > 0 && compareData.plan.length > 0 && (

                compareEntries.map((e) => (

                  <ChartPanel

                    key={e.algorithm}

                    id={`compare-prod-${e.algorithm}`}

                    title={`제품별 누적 생산량 – ${e.label}`}

                    visible={compareCharts.isVisible(`compare-prod-${e.algorithm}`)}

                    onVisibleChange={(v) => compareCharts.setVisible(`compare-prod-${e.algorithm}`, v)}

                    className="algo-prod-chart"

                    renderChart={() => (

                      <PlotChart

                        {...buildProductProductionCharts(

                          e.result.schedule,

                          e.result.plan,

                          e.result.prod_keys,

                          e.result.sim_end_minutes,

                          {

                            title: `${e.label} – 공정별 누적 생산`,

                            operIds: e.result.oper_ids,

                            timeAxis: ganttAxis,

                          },

                        )}

                      />

                    )}

                  />

                ))

              )}



              {tableResult && tableAlgo && (

                <section className="card">

                  <h3>

                    스케줄 결과 테이블 – {algoList.find((a) => a.id === tableAlgo)?.name ?? tableAlgo}

                  </h3>

                  <div className="table-wrap">

                    <table>

                      <thead>

                        <tr>

                          {["EQP_ID", "LOT_ID", "PLAN_PROD_KEY", "OPER_ID", "START_TM", "END_TM"].map((h) => (

                            <th key={h}>{h}</th>

                          ))}

                        </tr>

                      </thead>

                      <tbody>

                        {tableResult.schedule.map((row) => (

                          <tr key={`${row.EQP_ID}-${row.LOT_ID}-${row.START_TM}`}>

                            <td>{row.EQP_ID}</td>

                            <td>{row.LOT_ID}</td>

                            <td>{row.PLAN_PROD_KEY}</td>

                            <td>{row.OPER_ID}</td>

                            <td>{row.START_TM}</td>

                            <td>{row.END_TM}</td>

                          </tr>

                        ))}

                      </tbody>

                    </table>

                  </div>

                  <button

                    type="button"

                    className="btn btn-secondary"

                    onClick={() => downloadCsv(tableResult.schedule, `schedule_${tableAlgo}.csv`)}

                  >

                    CSV 다운로드

                  </button>

                </section>

              )}

            </div>

          )}



          {tab === "sim" && result && (

            <div className="tab-panel" key="sim">

              {!result.history.length ? (

                <p className="hint">
                  히스토리 데이터가 없습니다. 재생이 필요하면 “시뮬레이션 재생 데이터 포함”을 켜고 다시 실행하세요.
                </p>

              ) : (

                <div className="card-stagger">

                  <ChartPanel

                    id="sim-gantt"

                    title="설비(EQP) 배정 현황"

                    visible={simCharts.isVisible("sim-gantt")}

                    onVisibleChange={(v) => simCharts.setVisible("sim-gantt", v)}

                    renderChart={() => (

                      <PlotChart

                        className="plot-chart gantt-chart"

                        {...buildStepGantt(

                          result.history,

                          step,

                          result.prod_keys,

                          result.oper_ids,

                          ganttAxis,

                          result.conversion_plans ?? [],

                        )}

                      />

                    )}

                  />



                  <div className="grid-2">

                    <ChartPanel

                      id="sim-wip"

                      title="WIP 수량 현황"

                      visible={simCharts.isVisible("sim-wip")}

                      onVisibleChange={(v) => simCharts.setVisible("sim-wip", v)}

                      renderChart={() =>

                        snap ? <PlotChart {...buildWipChart(snap, result.plan)} /> : null

                      }

                    />

                    <ChartPanel

                      id="sim-achievement"

                      title="계획 달성 현황"

                      visible={simCharts.isVisible("sim-achievement")}

                      onVisibleChange={(v) => simCharts.setVisible("sim-achievement", v)}

                      renderChart={() =>

                        snap ? <PlotChart {...buildAchievementChart(snap, result.plan)} /> : null

                      }

                    />

                  </div>



                  <section className="card">

                    <h3>Arrange – Actual (구체 조합)</h3>

                    {snap && (

                      <ArrangeTable

                        rows={snap.arrange_actual ?? snap.arrange ?? []}

                        assigned={snap.assigned}

                        step={snap.step}

                        title="Actual"

                      />

                    )}

                  </section>



                  <section className="card">

                    <h3>Arrange – Abstract (유입 재공)</h3>

                    {snap && (

                      <AbstractArrangeTable

                        rows={snap.arrange_abstract ?? []}

                        assigned={snap.assigned}

                        step={snap.step}

                        simTime={snap.time}

                      />

                    )}

                  </section>



                  <DecisionLogTable

                    entries={result.decision_log ?? []}

                    highlightStep={snap?.step}

                    title={`결정 로그 (step ${snap?.step ?? step})`}

                  />



                  <EventTimeline

                    events={timelineEvents}

                    highlightKinds={currentStepEventKinds}

                    title={`시뮬레이션 이벤트 (step ${step}, 누적 ${timelineEvents.length}건)`}

                  />



                  <ChartPanel

                    id="sim-prod"

                    title="제품별 누적 생산량 (시간 × 수량)"

                    visible={simCharts.isVisible("sim-prod")}

                    onVisibleChange={(v) => simCharts.setVisible("sim-prod", v)}

                    renderChart={() =>

                      snap ? (

                        <PlotChart

                          {...buildProductProductionCharts(

                            snap.schedule,

                            result.plan,

                            result.prod_keys,

                            ganttAxis.timeEndMinutes,

                            {

                              title: `제품별 누적 생산량 – 공정별 (스텝 ${snap.step})`,

                              operIds: result.oper_ids,

                              timeAxis: ganttAxis,

                            },

                          )}

                        />

                      ) : null

                    }

                  />



                  <ChartPanel

                    id="sim-switch"

                    title="전환 횟수"

                    visible={simCharts.isVisible("sim-switch")}

                    onVisibleChange={(v) => simCharts.setVisible("sim-switch", v)}

                    renderChart={() => (snap ? <PlotChart {...buildSwitchMetrics(snap)} /> : null)}

                  />

                </div>

              )}

            </div>

          )}

        </div>

      )}

    </div>

  );

}


