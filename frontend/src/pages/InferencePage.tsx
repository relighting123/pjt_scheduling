import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlotChart from "../components/PlotChart";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import FullscreenPanel from "../components/FullscreenPanel";
import GanttKpiPanel from "../components/GanttKpiPanel";
import GanttLegendPanel from "../components/GanttLegendPanel";
import GanttSummaryPanel from "../components/GanttSummaryPanel";
import StepDebugger from "../components/StepDebugger";
import { EventTimeline } from "../components/EventTimeline";
import { api } from "../lib/api";
import { loadResultFromFile } from "../lib/resultFile";
import { downloadExcel } from "../lib/exportExcel";
import {
  buildEnhancedGantt,
  buildGanttLegendItems,
  buildProductProductionCharts,
  buildInferenceWipChart,
  ganttStepMarkerShape,
  type GanttBarLabel,
  ALGO_CHART_COLORS,
} from "../lib/charts";
import { buildEqpModelMap } from "../lib/metrics";
import { ruleTimekeyFromFolder, simBaseTimeFromRuleTimekey, parseSimBaseMs } from "../lib/ganttTime";
import type {
  AlgorithmId, AlgorithmInfo,
  AppConfig, DataSummary, InferenceResult, DecisionLogEntry,
} from "../types";

interface Props {
  modelExists: boolean;
  config: AppConfig | null;
  summary: DataSummary | null;
  folderLoading: boolean;
  onInputFolderChange: (f: string) => void | Promise<void>;
}

function facIdFromFolder(folder: string): string {
  return folder.split("/")[0] ?? "";
}

function parseOptionalInt(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const n = Number(trimmed);
  return Number.isNaN(n) ? undefined : n;
}

function buildInferOptions(
  selectedFolder: string,
  opts: {
    facIdOverride: string;
    ruleTimekey: string;
    fromDate: string;
    toDate: string;
    prevcnt: string;
    lotCd: string;
    nodb: boolean;
    decisionLog: boolean;
    wipInflow: boolean;
    includeHistory: boolean;
    dbLoad: boolean;
    dbAlias: string;
    noHistory: boolean;
    maxConversions: string;
    maxConversionsPerEqp: string;
    conversionMinutes: string;
  },
) {
  const facId = opts.facIdOverride.trim() || facIdFromFolder(selectedFolder);
  const maxConv = parseOptionalInt(opts.maxConversions);
  const maxConvEqp = parseOptionalInt(opts.maxConversionsPerEqp);
  const convMin = parseOptionalInt(opts.conversionMinutes);
  const prevcnt = parseOptionalInt(opts.prevcnt);
  const hasRange = opts.fromDate.trim() && opts.toDate.trim();
  return {
    input_folder: selectedFolder,
    ...(facId ? { fac_id: facId } : {}),
    ...(opts.ruleTimekey.trim() ? { rule_timekey: opts.ruleTimekey.trim() } : {}),
    ...(!opts.ruleTimekey.trim() && hasRange
      ? { from_date: opts.fromDate.trim(), to_date: opts.toDate.trim() }
      : {}),
    ...(!opts.ruleTimekey.trim() && !hasRange && prevcnt != null ? { prevcnt } : {}),
    ...(opts.lotCd.trim() ? { lot_cd: opts.lotCd.trim() } : {}),
    nodb: opts.nodb,
    decision_log: opts.decisionLog,
    enable_wip_inflow: opts.wipInflow,
    include_history: opts.includeHistory,
    db_load: opts.dbLoad,
    ...(opts.dbAlias.trim() ? { db_alias: opts.dbAlias.trim() } : {}),
    no_history: opts.noHistory,
    ...(maxConv != null ? { max_conversions: maxConv } : {}),
    ...(maxConvEqp != null ? { max_conversions_per_eqp: maxConvEqp } : {}),
    ...(convMin != null ? { conversion_minutes: convMin } : {}),
  };
}

type MainTab = "gantt" | "events" | "table" | "debug";
const ROWS = 200;

function VirtualTable({ rows }: { rows: InferenceResult["schedule"] }) {
  const [page, setPage] = useState(0);
  const total = rows.length;
  const pages = Math.ceil(total / ROWS);
  const visible = rows.slice(page * ROWS, (page + 1) * ROWS);
  const cols = ["EQP_ID","LOT_ID","CARRIER_ID","PLAN_PROD_ATTR_VAL","OPER_ID","ST","START_TM","END_TM","PROC_TIME","WF_QTY"] as const;

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
        <table className="data-table">
          <thead><tr>{cols.map(c => <th key={c} className={["START_TM","END_TM","PROC_TIME","WF_QTY"].includes(c) ? "num" : undefined}>{c}</th>)}</tr></thead>
          <tbody>
            {visible.map((r, i) => (
              <tr key={`${r.EQP_ID}-${r.LOT_ID}-${r.START_TM}-${i}`}>
                <td>{r.EQP_ID}</td><td>{r.LOT_ID}</td><td>{r.CARRIER_ID ?? ""}</td>
                <td>{r.PLAN_PROD_ATTR_VAL}</td><td>{r.OPER_ID ?? ""}</td><td>{r.ST ?? ""}</td>
                <td className="num">{r.START_TM}</td><td className="num">{r.END_TM}</td><td className="num">{r.PROC_TIME ?? ""}</td><td className="num">{r.WF_QTY ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function csv(sched: InferenceResult["schedule"], name: string) {
  const H = ["EQP_ID","LOT_ID","CARRIER_ID","PLAN_PROD_ATTR_VAL","OPER_ID","ST","START_TM","END_TM","PROC_TIME","WF_QTY"];
  const body = sched.map(r => H.map(h => String(r[h as keyof typeof r] ?? "")).join(",")).join("\n");
  const blob = new Blob(["﻿"+H.join(",")+"\n"+body], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
  a.click(); URL.revokeObjectURL(a.href);
}

function excelSchedule(sched: InferenceResult["schedule"], name: string) {
  const H = ["EQP_ID","LOT_ID","CARRIER_ID","PLAN_PROD_ATTR_VAL","OPER_ID","ST","START_TM","END_TM","PROC_TIME","WF_QTY"];
  const rows = sched.map(r => H.map(h => r[h as keyof typeof r] ?? ""));
  downloadExcel(name, H, rows);
}

function excelEventLog(events: NonNullable<InferenceResult["event_log"]>, name: string) {
  const H = ["시각(분)", "이벤트", "장비", "LOT", "제품", "공정"];
  const rows = events.map(ev => [ev.time, ev.kind, ev.eqp_id, ev.lot_id ?? "", ev.PLAN_PROD_ATTR_VAL ?? "", ev.oper_id ?? ""]);
  downloadExcel(name, H, rows);
}

export default function InferencePage({ modelExists, config, summary, folderLoading, onInputFolderChange }: Props) {
  const [result, setResult]           = useState<InferenceResult | null>(null);
  const [algorithm, setAlgorithm]     = useState<AlgorithmId>("scheduling_rl");
  const [algorithms, setAlgorithms]   = useState<AlgorithmInfo[]>([]);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [tab, setTab]                 = useState<MainTab>("gantt");
  const [fileSource, setFileSource]   = useState<string | null>(null);
  const fileRef                       = useRef<HTMLInputElement>(null);

  const [selectedFolder, setSelectedFolder] = useState("");
  const [facIdOverride, setFacIdOverride]   = useState("");
  const [decisionLog, setDecisionLog]       = useState(false);
  const [wipInflow, setWipInflow]           = useState(false);
  const [ruleTimekey, setRuleTimekey]       = useState("");
  const [fromDate, setFromDate]             = useState("");
  const [toDate, setToDate]                 = useState("");
  const [prevcnt, setPrevcnt]               = useState("");
  const [lotCd, setLotCd]                   = useState("");
  const [nodb, setNodb]                     = useState(false);
  const [includeHistory, setIncludeHistory] = useState(false);
  const [dbLoad, setDbLoad]                 = useState(false);
  const [dbAlias, setDbAlias]               = useState("");
  const [noHistory, setNoHistory]           = useState(false);
  const [maxConversions, setMaxConversions] = useState("");
  const [maxConversionsPerEqp, setMaxConversionsPerEqp] = useState("");
  const [conversionMinutes, setConversionMinutes] = useState("");
  const [lastInferMeta, setLastInferMeta]   = useState<string | null>(null);

  const defaultConversionMinutes = config?.default_env?.conversion_minutes ?? 60;

  const [labelMode, setLabelMode]         = useState<GanttBarLabel>("lot");
  const [ganttFixed, setGanttFixed]       = useState(false);
  const [ganttStart, setGanttStart]       = useState(0);
  const [ganttEnd, setGanttEnd]           = useState(1440);
  const [hiddenLegendKeys, setHiddenLegendKeys] = useState<Set<string>>(new Set());
  const [showConversionBars, setShowConversionBars] = useState(true);
  const [debugStep, setDebugStep] = useState<DecisionLogEntry | null>(null);

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

  const algoList = algorithms;

  // 간트 x축 기본 끝 = sim_end와 실제 스케줄 끝 중 큰 값.
  const dataEnd = useMemo(() => {
    if (!result) return 1440;
    const schedEnd = result.schedule?.length
      ? Math.max(...result.schedule.map((r) => r.END_TM))
      : 0;
    return Math.max(result.sim_end_minutes ?? 0, schedEnd, 1);
  }, [result]);
  // 스케줄이 지평선을 넘는 경우, 「X축 고정」 종료값 상한으로 안내
  const scheduleEnd = useMemo(
    () => (result?.schedule?.length ? Math.max(...result.schedule.map((r) => r.END_TM)) : 0),
    [result],
  );
  useEffect(() => { if (dataEnd > 0 && !ganttFixed) setGanttEnd(dataEnd); }, [dataEnd, ganttFixed]);

  const simBaseTime = useMemo(() => {
    const resultBase = result?.sim_base_time;
    if (resultBase && parseSimBaseMs(resultBase) != null) return resultBase;
    // 2순위: summary base(파싱 가능할 때만)
    if (summary?.sim_base_time && parseSimBaseMs(summary.sim_base_time) != null) {
      return summary.sim_base_time;
    }
    // 3순위: 폴더 경로의 RULE_TIMEKEY
    const rtk = ruleTimekeyFromFolder(selectedFolder);
    return rtk ? simBaseTimeFromRuleTimekey(rtk) : undefined;
  }, [result, summary, selectedFolder]);

  // 간트 기준 시각(0분) = RULE_TIMEKEY. 차트 상단 헤더로 표시.
  const ganttBaseInfo = useMemo(() => {
    const rtk = ruleTimekeyFromFolder(selectedFolder);
    const ms = parseSimBaseMs(simBaseTime);
    if (ms == null) return { rtk, baseText: null as string | null };
    const d = new Date(ms);
    const p = (n: number) => String(n).padStart(2, "0");
    const baseText = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
    return { rtk, baseText };
  }, [simBaseTime, selectedFolder]);

  const axis = useMemo(() => ({
    eqpIds: result?.eqp_ids ?? [],
    timeStartMinutes: ganttFixed ? ganttStart : 0,
    timeEndMinutes:   ganttFixed ? ganttEnd   : dataEnd,
    fixedRange: ganttFixed,
    simBaseTime,
  }), [result, ganttFixed, ganttStart, ganttEnd, dataEnd, simBaseTime]);

  const eqpModelMap = useMemo(() => buildEqpModelMap(result?.event_log ?? []), [result]);

  useEffect(() => {
    setHiddenLegendKeys(new Set());
    setShowConversionBars(true);
  }, [result]);

  const legendItems = useMemo(() => {
    if (!result) return [];
    return buildGanttLegendItems(result.schedule, result.prod_keys, result.oper_ids);
  }, [result]);

  const hasForcedLotStat = useMemo(
    () => (result?.schedule ?? []).some((r) => r.LOT_STAT_CD && r.LOT_STAT_CD !== "WAIT"),
    [result],
  );

  const toggleLegendKey = useCallback((pairKey: string) => {
    setHiddenLegendKeys((prev) => {
      const next = new Set(prev);
      if (next.has(pairKey)) next.delete(pairKey);
      else next.add(pairKey);
      return next;
    });
  }, []);

  const syncInferFolder = useCallback(async (folder?: string) => {
    if (!folder) return;
    setSelectedFolder(folder);
    await onInputFolderChange(folder);
  }, [onInputFolderChange]);

  const runInference = useCallback(async () => {
    setLoading(true); setError(null); setLastInferMeta(null);
    try {
      const res = await api.runInference({
        algorithm,
        save_output: true,
        ...buildInferOptions(selectedFolder, {
          facIdOverride,
          ruleTimekey,
          fromDate,
          toDate,
          prevcnt,
          lotCd,
          nodb,
          decisionLog,
          wipInflow,
          includeHistory,
          dbLoad,
          dbAlias,
          noHistory,
          maxConversions,
          maxConversionsPerEqp,
          conversionMinutes,
        }),
      });
      setResult(res); setFileSource(null); setTab("gantt");
      if (res.infer_meta) {
        const meta = res.infer_meta;
        const metaText = [
          `FAC=${meta.fac_id}`,
          `RULE_TIMEKEY=${meta.rule_timekey}`,
          meta.lot_cd ? `LOT_CD=${meta.lot_cd}` : null,
          meta.fetched_from_db ? "DB 조회" : "기존 JSON",
          meta.db_loaded ? "DB 적재 완료" : null,
          `전환 ${res.stats.conversions ?? 0}회`,
        ].filter(Boolean).join(" · ");
        setLastInferMeta(metaText);
        await syncInferFolder(meta.input_folder);
      }
    } catch(e) { setError(e instanceof Error ? e.message : "추론 실패"); }
    finally { setLoading(false); }
  }, [
    algorithm, selectedFolder, facIdOverride, decisionLog, wipInflow, ruleTimekey, fromDate, toDate,
    prevcnt, lotCd, nodb, includeHistory, dbLoad, dbAlias, noHistory, maxConversions,
    maxConversionsPerEqp, conversionMinutes, syncInferFolder,
  ]);

  const loadSaved = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await api.getInferenceResult(selectedFolder);
      setResult(res); setFileSource(null); setTab("gantt");
    } catch(e) { setError(e instanceof Error ? e.message : "저장 결과 없음"); }
    finally { setLoading(false); }
  }, [selectedFolder]);

  const loadFile = useCallback(async (file: File) => {
    setLoading(true); setError(null);
    try {
      const res = await loadResultFromFile(file);
      setResult(res); setFileSource(file.name); setTab("gantt");
    } catch(e) { setError(e instanceof Error ? e.message : "파일 오류"); }
    finally { setLoading(false); }
  }, []);

  const hasRuleTimekey = ruleTimekey.trim().length > 0;
  const hasDateRange = fromDate.trim().length > 0 || toDate.trim().length > 0;
  const hasPrevcntVal = prevcnt.trim().length > 0;

  const needsModel = algoList.find(a => a.id === algorithm)?.requires_model ?? false;
  const canRun = !needsModel || modelExists;
  const hasResult = !!result;

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

  const debugGanttChart = useMemo(() => {
    if (!ganttChart) return null;
    const marker = ganttStepMarkerShape(debugStep?.sim_time, axis);
    if (!marker) return ganttChart;
    return {
      ...ganttChart,
      layout: { ...ganttChart.layout, shapes: [...(ganttChart.layout.shapes ?? []), marker] },
    };
  }, [ganttChart, debugStep, axis]);

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
              setSelectedFolder(f); setResult(null);
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
          <div className="card-title">실행 옵션</div>
          <p className="hint mb-2">CLI <code>infer</code> 와 동일한 옵션입니다. 기본은 Oracle에서 input JSON을 조회한 뒤 추론합니다.</p>

          <label className="field-label" htmlFor="infer-fac-id">FAC_ID</label>
          <input
            id="infer-fac-id"
            className="input"
            type="text"
            placeholder={facIdFromFolder(selectedFolder) || "미지정 시 경로 첫 세그먼트"}
            value={facIdOverride}
            onChange={e => setFacIdOverride(e.target.value)}
            disabled={loading}
          />

          <label className="field-label mt-2" htmlFor="infer-rule-timekey">RULE_TIMEKEY</label>
          <input
            id="infer-rule-timekey"
            className="input"
            type="text"
            placeholder="미지정 시 최신"
            value={ruleTimekey}
            onChange={e => setRuleTimekey(e.target.value)}
            disabled={loading || hasDateRange || hasPrevcntVal}
          />

          <label className="field-label mt-2">구간 (RULE_TIMEKEY, BETWEEN)</label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              id="infer-from"
              className="input"
              type="text"
              placeholder="시작"
              value={fromDate}
              onChange={e => setFromDate(e.target.value)}
              disabled={loading || hasRuleTimekey || hasPrevcntVal}
            />
            <input
              id="infer-to"
              className="input"
              type="text"
              placeholder="종료"
              value={toDate}
              onChange={e => setToDate(e.target.value)}
              disabled={loading || hasRuleTimekey || hasPrevcntVal}
            />
          </div>

          <label className="field-label mt-2" htmlFor="infer-prevcnt">PREVCNT (최근 N개)</label>
          <input
            id="infer-prevcnt"
            className="input-number"
            type="number"
            min={1}
            placeholder="미지정 시 최신 1건"
            value={prevcnt}
            onChange={e => setPrevcnt(e.target.value)}
            disabled={loading || hasRuleTimekey || hasDateRange}
          />
          <p className="hint mt-1">RULE_TIMEKEY / 구간 / PREVCNT 중 하나만 사용됩니다 (미지정 시 최신).</p>

          <label className="field-label mt-2" htmlFor="infer-lot-cd">LOT_CD</label>
          <input
            id="infer-lot-cd"
            className="input"
            type="text"
            placeholder="미지정 시 전체 (discrete_arrange 제외)"
            value={lotCd}
            onChange={e => setLotCd(e.target.value)}
            disabled={loading}
          />

          <label className="check-label mt-2">
            <input
              type="checkbox"
              checked={nodb}
              onChange={e => setNodb(e.target.checked)}
              disabled={loading}
            />
            기존 JSON 사용 (--nodb, DB 조회 생략)
          </label>

          <label className="check-label">
            <input
              type="checkbox"
              checked={includeHistory}
              onChange={e => setIncludeHistory(e.target.checked)}
              disabled={loading}
            />
            history/event 포함 (--include-history)
          </label>

          <label className="check-label">
            <input
              type="checkbox"
              checked={dbLoad}
              onChange={e => setDbLoad(e.target.checked)}
              disabled={loading}
            />
            추론 후 DB 적재 (--db-load)
          </label>

          {dbLoad && (
            <>
              <label className="field-label mt-2" htmlFor="infer-db-alias">DB alias</label>
              <input
                id="infer-db-alias"
                className="input"
                type="text"
                placeholder="미지정 시 default"
                value={dbAlias}
                onChange={e => setDbAlias(e.target.value)}
                disabled={loading}
              />
              <label className="check-label mt-2">
                <input
                  type="checkbox"
                  checked={noHistory}
                  onChange={e => setNoHistory(e.target.checked)}
                  disabled={loading}
                />
                HIS 테이블 적재 생략 (--no-history)
              </label>
            </>
          )}

          {lastInferMeta && (
            <p className="hint mt-2">최근 실행: {lastInferMeta}</p>
          )}
        </div>

        <div className="card">
          <div className="card-title">컨버전 설정</div>
          <p className="hint mb-2">
            LOT_CD/TEMP 전환 횟수·소요 시간을 제한합니다. 비우면 무제한(시간은 기본 {defaultConversionMinutes}분).
          </p>

          <label className="field-label" htmlFor="infer-max-conv">컨버전 가능 횟수 (전체)</label>
          <input
            id="infer-max-conv"
            className="input"
            type="number"
            min={0}
            placeholder="무제한"
            value={maxConversions}
            onChange={e => setMaxConversions(e.target.value)}
            disabled={loading}
          />

          <label className="field-label mt-2" htmlFor="infer-max-conv-eqp">컨버전 가능 횟수 (EQP별)</label>
          <input
            id="infer-max-conv-eqp"
            className="input"
            type="number"
            min={0}
            placeholder="무제한"
            value={maxConversionsPerEqp}
            onChange={e => setMaxConversionsPerEqp(e.target.value)}
            disabled={loading}
          />

          <label className="field-label mt-2" htmlFor="infer-conv-min">전환 소요 시간 (분)</label>
          <input
            id="infer-conv-min"
            className="input"
            type="number"
            min={0}
            placeholder={`기본 ${defaultConversionMinutes}분`}
            value={conversionMinutes}
            onChange={e => setConversionMinutes(e.target.value)}
            disabled={loading}
          />
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
          <p className="hint" style={{ marginTop: "-0.25rem", marginBottom: "0.5rem" }}>
            스텝 디버거에서 스텝별 배정·차단 사유를 보려면 켜고 추론을 실행하세요.
          </p>
          <label className="check-label">
            <input type="checkbox" checked={wipInflow} onChange={e => setWipInflow(e.target.checked)} disabled={loading} />
            유입 재공 이벤트
          </label>

          <div className="gap-row mt-2">
            <button type="button" className={`btn btn-primary${loading ? " loading" : ""}`}
              onClick={runInference} disabled={loading || !canRun || folderLoading || !selectedFolder}>
              {loading ? "" : "▶ 추론 실행"}
            </button>
            <button type="button" className={`btn btn-ghost btn-sm${loading ? " loading" : ""}`}
              onClick={loadSaved} disabled={loading || folderLoading || !selectedFolder}>
              저장 로드
            </button>
            <button type="button" className="btn btn-ghost btn-sm"
              onClick={() => fileRef.current?.click()} disabled={loading}>
              파일 열기
            </button>
            <input ref={fileRef} type="file" accept=".json" style={{ display:"none" }}
              onChange={e => { const f = e.target.files?.[0]; e.target.value=""; if(f) void loadFile(f); }} />
          </div>
          {fileSource && <p className="hint mt-1">파일: <code>{fileSource}</code></p>}
          {needsModel && !modelExists && <p className="hint mt-1" style={{ color:"var(--warn)" }}>⚠ PPO는 모델이 필요합니다</p>}
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="content-area">
        {error && <ExpandableErrorBanner message={error} />}

        {!hasResult && !loading && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>좌측 패널에서 추론을 실행하거나 저장된 결과를 불러오세요.</p>
          </div>
        )}

        {hasResult && (
          <>
            <div className="tabs">
              <button type="button" className={`tab-btn${tab === "gantt" ? " active" : ""}`} onClick={() => setTab("gantt")} disabled={!result}>간트 차트</button>
              <button
                type="button"
                className={`tab-btn${tab === "events" ? " active" : ""}`}
                onClick={() => setTab("events")}
                disabled={!result?.event_log?.length}
                title={result?.event_log?.length ? undefined : "「history/event 포함」을 켜고 추론을 다시 실행하면 이벤트 이력을 볼 수 있습니다."}
              >
                이벤트 이력
              </button>
              <button type="button" className={`tab-btn${tab === "table" ? " active" : ""}`} onClick={() => setTab("table")} disabled={!result}>간트 테이블</button>
              <button
                type="button"
                className={`tab-btn${tab === "debug" ? " active" : ""}`}
                onClick={() => setTab("debug")}
                disabled={!result}
              >
                스텝 디버거
              </button>
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
                        {scheduleEnd > dataEnd && (
                          <button type="button" className="btn btn-ghost btn-xs"
                            onClick={() => { setGanttStart(0); setGanttEnd(scheduleEnd); }}>
                            전체({scheduleEnd}분)
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {(ganttBaseInfo.rtk || ganttBaseInfo.baseText) && (
                  <div className="gantt-base-info">
                    {ganttBaseInfo.rtk && (
                      <span className="gantt-base-rtk">RULE_TIMEKEY <b>{ganttBaseInfo.rtk}</b></span>
                    )}
                    {ganttBaseInfo.baseText && (
                      <span className="gantt-base-time">기준 시각 <b>{ganttBaseInfo.baseText}</b> = 0분 · 이후 눈금은 실제 시각(HH:mm)</span>
                    )}
                  </div>
                )}

                {ganttChart && (
                  <FullscreenPanel title="간트 차트" className="gantt-chart-panel-wrap">
                    <div className="chart-wrap gantt-chart-panel">
                      <PlotChart {...ganttChart} scrollable />
                    </div>
                  </FullscreenPanel>
                )}

                {hasForcedLotStat && (
                  <div className="gantt-base-info gantt-forced-legend">
                    <span><span className="gantt-forced-swatch" style={{ background: "#16a34a" }} /> PROC(강제 배정)</span>
                    <span><span className="gantt-forced-swatch" style={{ background: "#ca8a04" }} /> LOAD/SELE/RESV(강제 배정)</span>
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
                  <FullscreenPanel title="시간별 제품·공정 누적 생산" className="card gantt-production-panel">
                    <p className="gantt-production-hint">
                      제품별 서브플롯에 공정(O) 실적(실선)과 계획(점선)을 표시합니다. X축은 간트 차트와 동일합니다.
                    </p>
                    <div className="chart-wrap gantt-production-chart">
                      <PlotChart {...productionChart} scrollable />
                    </div>
                  </FullscreenPanel>
                )}

                {wipChart && (
                  <FullscreenPanel title="재공 처리 현황" className="card gantt-production-panel">
                    <p className="gantt-production-hint">
                      제품/공정별 완료 수량, 잔여 재공(미처리 대기), 계획 미달을 표시합니다. 세로 점선이 계획 목표입니다.
                    </p>
                    <div className="chart-wrap">
                      <PlotChart {...wipChart} />
                    </div>
                  </FullscreenPanel>
                )}

                <GanttSummaryPanel result={result} eqpModelMap={eqpModelMap} />
              </div>
            )}

            {/* EVENTS TAB */}
            {tab === "events" && result?.event_log && (
              <FullscreenPanel
                title={`이벤트 이력 (${result.event_log.length.toLocaleString()}건)`}
                className="card tab-panel"
                actions={
                  <button type="button" className="btn btn-ghost btn-sm"
                    onClick={() => excelEventLog(result.event_log!, `event_log_${result.algorithm ?? "result"}.xls`)}>
                    엑셀 다운로드
                  </button>
                }
              >
                <EventTimeline events={result.event_log} highlightKinds={new Set<string>()} title="" />
              </FullscreenPanel>
            )}

            {/* TABLE TAB */}
            {tab === "table" && result && (
              <FullscreenPanel
                title="스케줄 테이블"
                className="card tab-panel"
                actions={
                  <>
                    <button type="button" className="btn btn-ghost btn-sm"
                      onClick={() => csv(result.schedule, `schedule_${result.algorithm ?? "result"}.csv`)}>
                      CSV 다운로드
                    </button>
                    <button type="button" className="btn btn-ghost btn-sm"
                      onClick={() => excelSchedule(result.schedule, `schedule_${result.algorithm ?? "result"}.xls`)}>
                      엑셀 다운로드
                    </button>
                  </>
                }
              >
                <VirtualTable rows={result.schedule} />
              </FullscreenPanel>
            )}

            {/* STEP DEBUGGER TAB */}
            {tab === "debug" && result && (
              <div className="tab-panel">
                {result.decision_log?.length ? (
                  <div className="stepdbg-page">
                    {debugGanttChart && (
                      <FullscreenPanel title="간트 차트 (스텝 동기화)" className="stepdbg-gantt-wrap chart-wrap gantt-chart-panel">
                        <PlotChart {...debugGanttChart} scrollable />
                      </FullscreenPanel>
                    )}
                    <StepDebugger entries={result.decision_log} onStepChange={setDebugStep} />
                  </div>
                ) : (
                  <StepDebugger entries={[]} />
                )}
              </div>
            )}

          </>
        )}
      </div>
    </div>
  );
}
