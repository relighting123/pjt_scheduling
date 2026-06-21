import type { Data, Layout } from "plotly.js";
import type { ConversionPlan, HistorySnap, InferenceResult, PlanRecord, ScheduleRecord, TestBenchmarkDataset, TrainSeries } from "../types";
import { buildColorMap, OPER_BORDER_COLORS, PROD_COLORS } from "./colors";

export interface GanttAxisOptions {
  eqpIds: string[];
  timeStartMinutes?: number;
  timeEndMinutes: number;
  /** true면 timeStartMinutes~timeEndMinutes 구간으로 X축 고정 */
  fixedRange?: boolean;
}

export function resolveGanttTimeRange(axis: GanttAxisOptions): [number, number] {
  if (axis.fixedRange) {
    const start = Math.max(0, axis.timeStartMinutes ?? 0);
    const end = Math.max(start + 1, axis.timeEndMinutes ?? 1);
    return [start, end];
  }
  const end = Math.max(axis.timeEndMinutes ?? 0, 1);
  return [0, end];
}

function sortedEqpIds(eqpIds: string[]): string[] {
  return [...eqpIds].sort();
}

function legendTraces(prodKeys: string[], operIds: string[], schedule: ScheduleRecord[]): Data[] {
  const prodColorMap = buildColorMap(prodKeys, PROD_COLORS);
  const operColorMap = buildColorMap(operIds, OPER_BORDER_COLORS);
  const traces: Data[] = [];

  prodKeys.forEach((pk) => {
    traces.push({
      type: "bar",
      orientation: "h",
      x: [0],
      y: [""],
      name: pk,
      marker: { color: prodColorMap[pk] ?? "#888888" },
      showlegend: true,
      visible: schedule.some((r) => r.PLAN_PROD_KEY === pk) ? true : "legendonly",
    });
  });

  operIds.forEach((op) => {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: [null],
      y: [null],
      name: `[OPER] ${op}`,
      marker: {
        size: 12,
        color: operColorMap[op] ?? "#222222",
        symbol: "square",
        line: { width: 2, color: "white" },
      },
      showlegend: true,
    });
  });

  return traces;
}

/** 간트 공통 스타일 (둥근 pill 바 + 소프트 그리드) */
const GANTT_THEME = {
  plotBg: "#f8f9fc",
  paperBg: "#ffffff",
  gridColor: "rgba(148, 163, 184, 0.18)",
  gridWidth: 1,
  fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
  titleColor: "#1e293b",
  axisColor: "#64748b",
  barRadius: 10,
  barOpacity: 0.92,
} as const;

function ganttBarMarker(
  fillColor: string,
  operColor: string,
  visible: boolean,
) {
  return {
    color: fillColor,
    opacity: visible ? GANTT_THEME.barOpacity : 0.12,
    line: {
      color: operColor,
      width: visible ? 2 : 0,
    },
    cornerradius: GANTT_THEME.barRadius,
  };
}

function ganttTraces(
  schedule: ScheduleRecord[],
  prodKeys: string[],
  operIds: string[],
  highlightMax?: number,
): Data[] {
  const prodColorMap = buildColorMap(prodKeys, PROD_COLORS);
  const operColorMap = buildColorMap(operIds, OPER_BORDER_COLORS);
  const traces: Data[] = [];

  schedule.forEach((rec, idx) => {
    const visible = highlightMax === undefined || idx <= highlightMax;
    const width = rec.END_TM - rec.START_TM;
    traces.push({
      type: "bar",
      orientation: "h",
      x: [width],
      y: [rec.EQP_ID],
      base: [rec.START_TM],
      marker: ganttBarMarker(
        prodColorMap[rec.PLAN_PROD_KEY] ?? "#94a3b8",
        operColorMap[rec.OPER_ID ?? ""] ?? "#475569",
        visible,
      ),
      text: visible ? rec.LOT_ID : "",
      textposition: "inside",
      insidetextanchor: "middle",
      textfont: { size: 11, color: "#ffffff", family: GANTT_THEME.fontFamily },
      hovertemplate:
        `<b>LOT: ${rec.LOT_ID}</b><br>` +
        `EQP: ${rec.EQP_ID}<br>` +
        `제품: ${rec.PLAN_PROD_KEY}<br>` +
        `공정: ${rec.OPER_ID ?? "N/A"}<br>` +
        `시작: ${rec.START_TM}분<br>` +
        `종료: ${rec.END_TM}분<br>` +
        `소요: ${width}분<extra></extra>`,
      showlegend: false,
    } as Data);
  });

  return [...traces, ...legendTraces(prodKeys, operIds, schedule)];
}

function conversionLegendTrace(hasConversion: boolean): Data {
  return {
    type: "bar",
    orientation: "h",
    x: [0],
    y: [""],
    name: "Conversion",
    marker: {
      color: "#f59e0b",
      opacity: 0.88,
      line: { color: "#b45309", width: 2 },
      cornerradius: 6,
    },
    showlegend: true,
    visible: hasConversion ? true : "legendonly",
  };
}

function conversionTraces(
  plans: ConversionPlan[],
  visibleUntilTime: number,
): Data[] {
  return plans
    .filter((p) => p.conv_start_min < visibleUntilTime)
    .map((p) => {
      const end = Math.min(p.conv_end_min, visibleUntilTime);
      const width = Math.max(end - p.conv_start_min, 0);
      if (width <= 0) return null;
      return {
        type: "bar",
        orientation: "h",
        x: [width],
        y: [p.eqp_id],
        base: [p.conv_start_min],
        marker: {
          color: "#f59e0b",
          opacity: 0.88,
          line: { color: "#b45309", width: 2 },
          cornerradius: 6,
        },
        text: `CONV ${p.from_lot_cd}→${p.to_lot_cd}`,
        textposition: "inside",
        insidetextanchor: "middle",
        textfont: { size: 10, color: "#1e293b", family: GANTT_THEME.fontFamily },
        hovertemplate:
          `<b>Conversion</b><br>` +
          `EQP: ${p.eqp_id}<br>` +
          `${p.from_lot_cd} → ${p.to_lot_cd}<br>` +
          `시작: ${p.conv_start_min}분<br>` +
          `종료: ${p.conv_end_min}분<br>` +
          `소요: ${p.conv_end_min - p.conv_start_min}분<extra></extra>`,
        showlegend: false,
      } as Data;
    })
    .filter((t): t is Data => t !== null);
}

function buildGanttLayout(
  title: string,
  axis: GanttAxisOptions,
): Partial<Layout> {
  const eqps = sortedEqpIds(axis.eqpIds);
  const [timeStart, timeEnd] = resolveGanttTimeRange(axis);

  return {
    title: {
      text: title,
      font: { size: 15, color: GANTT_THEME.titleColor, family: GANTT_THEME.fontFamily },
    },
    xaxis: {
      title: { text: "시뮬레이션 시간 (분)", font: { size: 12, color: GANTT_THEME.axisColor } },
      showgrid: true,
      gridcolor: GANTT_THEME.gridColor,
      gridwidth: GANTT_THEME.gridWidth,
      zeroline: false,
      range: [timeStart, timeEnd],
      tickfont: { size: 11, color: GANTT_THEME.axisColor },
      ...(axis.fixedRange ? { fixedrange: true } : {}),
    },
    yaxis: {
      categoryorder: "array",
      categoryarray: eqps,
      title: { text: "설비(EQP)", font: { size: 12, color: GANTT_THEME.axisColor } },
      tickfont: { size: 11, color: GANTT_THEME.axisColor },
      showgrid: false,
    },
    barmode: "overlay",
    bargap: 0.35,
    legend: {
      title: { text: "제품 / 공정", font: { size: 11 } },
      orientation: "v",
      x: 1.02,
      bgcolor: "rgba(255,255,255,0.85)",
      bordercolor: "rgba(148,163,184,0.25)",
      borderwidth: 1,
    },
    height: Math.max(350, 72 * Math.max(eqps.length, 1)),
    plot_bgcolor: GANTT_THEME.plotBg,
    paper_bgcolor: GANTT_THEME.paperBg,
    margin: { l: 88, r: 188, t: 56, b: 52 },
    hoverlabel: {
      bgcolor: "#ffffff",
      bordercolor: "rgba(148,163,184,0.35)",
      font: { family: GANTT_THEME.fontFamily, size: 12 },
    },
  };
}

export function buildStepGantt(
  history: HistorySnap[],
  step: number,
  prodKeys: string[],
  operIds: string[],
  axis: GanttAxisOptions,
  conversionPlans: ConversionPlan[] = [],
): { data: Data[]; layout: Partial<Layout> } {
  if (!history.length) {
    return {
      data: legendTraces(prodKeys, operIds, []),
      layout: buildGanttLayout("스케줄 간트 차트", axis),
    };
  }
  const snap = history[Math.min(step, history.length - 1)];
  const schedule = snap.schedule;
  const convBars = conversionTraces(conversionPlans, snap.time + 1);
  const hasConv = convBars.length > 0;
  return {
    data: [
      ...ganttTraces(schedule, prodKeys, operIds, schedule.length - 1),
      ...convBars,
      conversionLegendTrace(hasConv),
    ],
    layout: buildGanttLayout(
      `Scheduling 간트 (스텝 ${snap.step} / 시각 ${snap.time}분)`,
      axis,
    ),
  };
}

export function buildWipChart(snap: HistorySnap, plan: PlanRecord[]): { data: Data[]; layout: Partial<Layout> } {
  const completed = snap.completed;
  const waiting = snap.wip_waiting ?? {};
  const labels: string[] = [];
  const done: number[] = [];
  const wipWait: number[] = [];
  const remaining: number[] = [];

  plan.forEach((p) => {
    const key = `${p.plan_prod_key}|${p.oper_id}`;
    const finished = completed[key] ?? 0;
    const waitQty = waiting[key] ?? 0;
    const total = p.d0_plan_qty;
    labels.push(`${p.plan_prod_key}\n${p.oper_id}`);
    done.push(finished);
    wipWait.push(waitQty);
    remaining.push(Math.max(total - finished - waitQty, 0));
  });

  return {
    data: [
      { type: "bar", name: "완료", x: labels, y: done, marker: { color: "#55A868" } },
      { type: "bar", name: "대기 WIP", x: labels, y: wipWait, marker: { color: "#DD8452" } },
      { type: "bar", name: "잔여 계획", x: labels, y: remaining, marker: { color: "#C44E52" } },
    ],
    layout: {
      title: { text: `WIP 수량 현황 (스텝 ${snap.step})` },
      xaxis: { title: { text: "제품 / 공정" } },
      yaxis: { title: { text: "웨이퍼 수량 (매)" } },
      barmode: "stack",
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.25 },
      height: 320,
      margin: { t: 50, b: 80 },
    },
  };
}

export function buildAchievementChart(snap: HistorySnap, plan: PlanRecord[]): { data: Data[]; layout: Partial<Layout> } {
  const completed = snap.completed;
  const labels: string[] = [];
  const rates: number[] = [];
  const texts: string[] = [];

  plan.forEach((p) => {
    const key = `${p.plan_prod_key}|${p.oper_id}`;
    const finished = completed[key] ?? 0;
    const target = p.d0_plan_qty;
    const rate = Math.min((finished / Math.max(target, 1)) * 100, 100);
    labels.push(`${p.plan_prod_key} / ${p.oper_id}`);
    rates.push(Math.round(rate * 10) / 10);
    texts.push(`${finished}/${target}매  (${Math.round(rate * 10) / 10}%)`);
  });

  const colors = rates.map((r) =>
    r >= 100 ? "#55A868" : r >= 60 ? "#DD8452" : "#C44E52",
  );

  return {
    data: [{
      type: "bar",
      orientation: "h",
      x: rates,
      y: labels,
      marker: { color: colors },
      text: texts,
      textposition: "outside",
    }],
    layout: {
      title: { text: `계획 달성률 (스텝 ${snap.step})` },
      xaxis: { title: { text: "달성률 (%)" }, range: [0, 115] },
      shapes: [{ type: "line", x0: 100, x1: 100, y0: 0, y1: 1, yref: "paper", line: { dash: "dash", color: "#4C72B0", width: 1.5 } }],
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      height: 320,
      margin: { l: 150, r: 120, t: 50, b: 40 },
    },
  };
}

function opersForProduct(plan: PlanRecord[], prod: string): string[] {
  return [...new Set(
    plan.filter((p) => p.plan_prod_key === prod).map((p) => p.oper_id),
  )].sort();
}

function operPlanQty(plan: PlanRecord[], prod: string, operId: string): number {
  const row = plan.find((p) => p.plan_prod_key === prod && p.oper_id === operId);
  return row?.d0_plan_qty ?? 0;
}

function cumulativeProductionSeries(
  schedule: ScheduleRecord[],
  prod: string,
  operId: string,
  timeEnd: number,
): { x: number[]; y: number[] } {
  const events = schedule
    .filter((r) => r.PLAN_PROD_KEY === prod && (r.OPER_ID ?? "") === operId)
    .map((r) => ({ t: r.START_TM, q: r.WF_QTY ?? 25 }))
    .sort((a, b) => a.t - b.t || a.q - b.q);

  let cum = 0;
  const x: number[] = [0];
  const y: number[] = [0];

  for (const e of events) {
    if (e.t > x[x.length - 1]) {
      x.push(e.t);
      y.push(cum);
    }
    cum += e.q;
    x.push(e.t);
    y.push(cum);
  }

  if (x[x.length - 1] < timeEnd) {
    x.push(timeEnd);
    y.push(cum);
  }

  return { x, y };
}

function subplotAxisNames(index: number): { x: string; y: string } {
  if (index === 0) return { x: "x", y: "y" };
  const n = index + 1;
  return { x: `x${n}`, y: `y${n}` };
}

export interface ProductProductionChartOptions {
  title?: string;
  overlaySchedule?: ScheduleRecord[];
  overlayLabel?: string;
  operIds?: string[];
  /** 간트 X축과 동일한 시간 범위를 쓸 때 전달 */
  timeAxis?: Pick<GanttAxisOptions, "timeStartMinutes" | "timeEndMinutes" | "fixedRange">;
}

export function buildProductProductionCharts(
  schedule: ScheduleRecord[],
  plan: PlanRecord[],
  prodKeys: string[],
  timeEndMinutes: number,
  options: ProductProductionChartOptions = {},
): { data: Data[]; layout: Partial<Layout> } {
  const prods = [...prodKeys].sort();
  const n = Math.max(prods.length, 1);
  const [timeStart, timeEnd] = resolveGanttTimeRange({
    eqpIds: [],
    timeStartMinutes: options.timeAxis?.timeStartMinutes,
    timeEndMinutes: options.timeAxis?.timeEndMinutes ?? timeEndMinutes,
    fixedRange: options.timeAxis?.fixedRange,
  });
  const allOpers = options.operIds ?? [...new Set(plan.map((p) => p.oper_id))].sort();
  const operColorMap = buildColorMap(allOpers, OPER_BORDER_COLORS);
  const data: Data[] = [];
  const layout: Partial<Layout> = {
    title: { text: options.title ?? "제품별 누적 생산량 (공정별)", font: { size: 16 } },
    grid: { rows: n, columns: 1, pattern: "independent", roworder: "top to bottom" },
    height: Math.max(300 * n, 320),
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    margin: { l: 70, r: 160, t: 60, b: 50 },
    showlegend: true,
    legend: { orientation: "v", x: 1.02, y: 1 },
  };

  prods.forEach((prod, i) => {
    const { x: xAxis, y: yAxis } = subplotAxisNames(i);
    const xKey = (i === 0 ? "xaxis" : `xaxis${i + 1}`) as keyof Layout;
    const yKey = (i === 0 ? "yaxis" : `yaxis${i + 1}`) as keyof Layout;
    const opers = opersForProduct(plan, prod);

    (layout as Record<string, unknown>)[xKey] = {
      title: i === n - 1 ? { text: "시뮬레이션 시간 (분)" } : undefined,
      range: [timeStart, timeEnd],
      showgrid: true,
      gridcolor: "#E5E5E5",
      ...(options.timeAxis?.fixedRange ? { fixedrange: true } : {}),
    };
    (layout as Record<string, unknown>)[yKey] = {
      title: { text: `${prod} 누적 생산 (매)` },
      rangemode: "tozero",
      showgrid: true,
      gridcolor: "#F0F0F0",
    };

    opers.forEach((oper) => {
      const color = operColorMap[oper] ?? "#888888";
      const planQty = operPlanQty(plan, prod, oper);
      const showInLegend = i === 0;

      const actual = cumulativeProductionSeries(schedule, prod, oper, timeEnd);
      data.push({
        type: "scatter",
        mode: "lines",
        name: `${oper} 실적`,
        x: actual.x,
        y: actual.y,
        line: { color, width: 2.5, shape: "hv" },
        xaxis: xAxis,
        yaxis: yAxis,
        legendgroup: `${oper}-actual`,
        showlegend: showInLegend,
        hovertemplate: `${prod} / ${oper}<br>시간: %{x}분<br>누적: %{y}매<extra></extra>`,
      });

      data.push({
        type: "scatter",
        mode: "lines",
        name: `${oper} 계획`,
        x: [0, timeEnd],
        y: [0, planQty],
        line: { color, width: 1.5, dash: "dash" },
        xaxis: xAxis,
        yaxis: yAxis,
        legendgroup: `${oper}-plan`,
        showlegend: showInLegend,
        hovertemplate: `${prod} / ${oper} 계획<br>시간: %{x}분<br>목표: %{y}매<extra></extra>`,
      });

      if (options.overlaySchedule) {
        const overlay = cumulativeProductionSeries(options.overlaySchedule, prod, oper, timeEnd);
        data.push({
          type: "scatter",
          mode: "lines",
          name: `${oper} ${options.overlayLabel ?? "초기"}`,
          x: overlay.x,
          y: overlay.y,
          line: { color, width: 1.5, dash: "dot", shape: "hv" },
          xaxis: xAxis,
          yaxis: yAxis,
          legendgroup: `${oper}-overlay`,
          showlegend: showInLegend,
          hovertemplate: `${prod} / ${oper} 초기<br>시간: %{x}분<br>누적: %{y}매<extra></extra>`,
        });
      }
    });
  });

  return { data, layout };
}

export function buildSwitchMetrics(snap: HistorySnap): { data: Data[]; layout: Partial<Layout> } {
  return {
    data: [
      {
        type: "indicator",
        mode: "number",
        value: snap.oper_sw,
        title: { text: "공정 전환 횟수" },
        number: { font: { color: "#C44E52", size: 40 } },
        domain: { x: [0, 0.45], y: [0, 1] },
      },
      {
        type: "indicator",
        mode: "number",
        value: snap.prod_sw,
        title: { text: "제품 전환 횟수" },
        number: { font: { color: "#DD8452", size: 40 } },
        domain: { x: [0.55, 1], y: [0, 1] },
      },
    ],
    layout: {
      height: 180,
      margin: { t: 40, b: 10 },
      paper_bgcolor: "white",
    },
  };
}

interface ScheduleStats {
  makespan: number;
  idle_total: number;
  oper_switches: number;
  prod_switches: number;
  achievement: Record<string, number>;
}

function computeStats(schedule: ScheduleRecord[], plan: PlanRecord[]): ScheduleStats {
  if (!schedule.length) {
    return { makespan: 0, idle_total: 0, oper_switches: 0, prod_switches: 0, achievement: {} };
  }

  const makespan = Math.max(...schedule.map((r) => r.END_TM));
  const eqpSeq: Record<string, ScheduleRecord[]> = {};
  schedule.forEach((r) => {
    (eqpSeq[r.EQP_ID] ??= []).push(r);
  });

  let operSw = 0;
  let prodSw = 0;
  let idleTotal = 0;

  Object.values(eqpSeq).forEach((recs) => {
    recs.sort((a, b) => a.START_TM - b.START_TM);
    for (let i = 1; i < recs.length; i++) {
      if (recs[i].OPER_ID !== recs[i - 1].OPER_ID) operSw++;
      if (recs[i].PLAN_PROD_KEY !== recs[i - 1].PLAN_PROD_KEY) prodSw++;
      idleTotal += Math.max(recs[i].START_TM - recs[i - 1].END_TM, 0);
    }
  });

  const completed: Record<string, number> = {};
  schedule.forEach((r) => {
    const key = `${r.PLAN_PROD_KEY}|${r.OPER_ID ?? ""}`;
    completed[key] = (completed[key] ?? 0) + (r.WF_QTY ?? 25);
  });

  const achievement: Record<string, number> = {};
  plan.forEach((p) => {
    const key = `${p.plan_prod_key}|${p.oper_id}`;
    const label = `${p.plan_prod_key}/${p.oper_id}`;
    const done = completed[key] ?? 0;
    achievement[label] = Math.round((done / Math.max(p.d0_plan_qty, 1)) * 1000) / 10;
  });

  return { makespan, idle_total: idleTotal, oper_switches: operSw, prod_switches: prodSw, achievement };
}

export function buildComparisonKpi(
  initial: ScheduleRecord[],
  post: ScheduleRecord[],
  plan: PlanRecord[],
): { data: Data[]; layout: Partial<Layout> } {
  const initS = computeStats(initial, plan);
  const postS = computeStats(post, plan);
  const metrics = ["Makespan(분)", "Idle 합계(분)", "공정 전환", "제품 전환"];

  return {
    data: [
      { type: "bar", name: "초기 스케줄", x: metrics, y: [initS.makespan, initS.idle_total, initS.oper_switches, initS.prod_switches], marker: { color: "#4C72B0" } },
      { type: "bar", name: "Scheduling", x: metrics, y: [postS.makespan, postS.idle_total, postS.oper_switches, postS.prod_switches], marker: { color: "#55A868" } },
    ],
    layout: {
      title: { text: "초기 스케줄 vs Scheduling KPI 비교" },
      barmode: "group",
      yaxis: { title: { text: "값" } },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.2 },
      height: 360,
      margin: { t: 60, b: 80 },
    },
  };
}

export function buildAchievementComparison(
  initial: ScheduleRecord[],
  post: ScheduleRecord[],
  plan: PlanRecord[],
): { data: Data[]; layout: Partial<Layout> } {
  const initS = computeStats(initial, plan);
  const postS = computeStats(post, plan);
  const labels = [...new Set([...Object.keys(initS.achievement), ...Object.keys(postS.achievement)])].sort();

  return {
    data: [
      { type: "bar", name: "초기 스케줄", x: labels, y: labels.map((l) => initS.achievement[l] ?? 0), marker: { color: "#4C72B0" } },
      { type: "bar", name: "Scheduling", x: labels, y: labels.map((l) => postS.achievement[l] ?? 0), marker: { color: "#55A868" } },
    ],
    layout: {
      title: { text: "제품/공정별 계획 달성률 비교 (%)" },
      barmode: "group",
      yaxis: { title: { text: "달성률 (%)" }, range: [0, 120] },
      xaxis: { title: { text: "제품 / 공정" } },
      shapes: [{ type: "line", x0: 0, x1: 1, y0: 100, y1: 100, xref: "paper", line: { dash: "dash", color: "red", width: 1 } }],
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.25 },
      height: 360,
      margin: { t: 60, b: 80 },
    },
  };
}

export const ALGO_CHART_COLORS: Record<string, string> = {
  rl: "#4C72B0",
  minprogress: "#55A868",
  earliest_st: "#DD8452",
};

export interface AlgoCompareEntry {
  algorithm: string;
  label: string;
  result: InferenceResult;
}

export function resultScheduleStats(result: InferenceResult): ScheduleStats {
  const sched = result.schedule;
  const base = computeStats(sched, result.plan);
  return {
    ...base,
    idle_total: result.stats.idle_total,
    oper_switches: result.stats.oper_switches,
    prod_switches: result.stats.prod_switches,
    makespan: sched.length ? Math.max(...sched.map((r) => r.END_TM)) : 0,
  };
}

export function buildAlgorithmKpiComparison(
  entries: AlgoCompareEntry[],
): { data: Data[]; layout: Partial<Layout> } {
  const metrics = ["Makespan(분)", "Idle 합계(분)", "공정 전환", "제품 전환"];
  const data: Data[] = entries.map((e) => {
    const s = resultScheduleStats(e.result);
    return {
      type: "bar" as const,
      name: e.label,
      x: metrics,
      y: [s.makespan, s.idle_total, s.oper_switches, s.prod_switches],
      marker: { color: ALGO_CHART_COLORS[e.algorithm] ?? "#888" },
    };
  });

  return {
    data,
    layout: {
      title: { text: "알고리즘별 KPI 비교" },
      barmode: "group",
      yaxis: { title: { text: "값" } },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.2 },
      height: 380,
      margin: { t: 60, b: 80 },
    },
  };
}

export function buildAlgorithmAchievementComparison(
  entries: AlgoCompareEntry[],
): { data: Data[]; layout: Partial<Layout> } {
  const labelSet = new Set<string>();
  entries.forEach((e) => {
    Object.keys(resultScheduleStats(e.result).achievement).forEach((k) => labelSet.add(k));
  });
  const labels = [...labelSet].sort();

  const data: Data[] = entries.map((e) => {
    const ach = resultScheduleStats(e.result).achievement;
    return {
      type: "bar" as const,
      name: e.label,
      x: labels,
      y: labels.map((l) => ach[l] ?? 0),
      marker: { color: ALGO_CHART_COLORS[e.algorithm] ?? "#888" },
    };
  });

  return {
    data,
    layout: {
      title: { text: "알고리즘별 계획 달성률 비교 (%)" },
      barmode: "group",
      yaxis: { title: { text: "달성률 (%)" }, range: [0, 120] },
      xaxis: { title: { text: "제품 / 공정" } },
      shapes: [{
        type: "line",
        x0: 0,
        x1: 1,
        y0: 100,
        y1: 100,
        xref: "paper",
        line: { dash: "dash", color: "red", width: 1 },
      }],
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.25 },
      height: 380,
      margin: { t: 60, b: 100 },
    },
  };
}

function subplotAxisPair(index: number, total: number): {
  xName: string;
  yName: string;
  xKey: string;
  yKey: string;
  domain: [number, number];
} {
  const rowH = 1 / total;
  const gap = 0.04;
  const top = 1 - index * rowH;
  const bottom = top - rowH + (index < total - 1 ? gap : 0);
  const n = index + 1;
  return {
    xName: index === 0 ? "x" : `x${n}`,
    yName: index === 0 ? "y" : `y${n}`,
    xKey: index === 0 ? "xaxis" : `xaxis${n}`,
    yKey: index === 0 ? "yaxis" : `yaxis${n}`,
    domain: [bottom, top - gap * 0.5] as [number, number],
  };
}

export function buildAlgorithmGanttComparison(
  entries: AlgoCompareEntry[],
  axis: GanttAxisOptions,
): { data: Data[]; layout: Partial<Layout> } {
  if (!entries.length) {
    return { data: [], layout: { height: 300 } };
  }

  const eqps = sortedEqpIds(axis.eqpIds);
  const [timeStart, timeEnd] = resolveGanttTimeRange(axis);
  const n = entries.length;
  const data: Data[] = [];
  const layout: Partial<Layout> = {
    grid: { rows: n, columns: 1, pattern: "independent", roworder: "top to bottom" },
    height: Math.max(300 * n, 420),
    barmode: "overlay",
    bargap: 0.35,
    showlegend: n === 1,
    plot_bgcolor: GANTT_THEME.plotBg,
    paper_bgcolor: GANTT_THEME.paperBg,
    legend: {
      x: 1.02,
      bgcolor: "rgba(255,255,255,0.85)",
      bordercolor: "rgba(148,163,184,0.25)",
      borderwidth: 1,
    },
    hoverlabel: {
      bgcolor: "#ffffff",
      bordercolor: "rgba(148,163,184,0.35)",
      font: { family: GANTT_THEME.fontFamily, size: 12 },
    },
    annotations: [],
    margin: { l: 88, r: 188, t: 44, b: 52 },
  };

  entries.forEach((entry, i) => {
    const { xName, yName, xKey, yKey, domain } = subplotAxisPair(i, n);
    const prodKeys = entry.result.prod_keys;
    const operIds = entry.result.oper_ids;
    const traces = ganttTraces(entry.result.schedule, prodKeys, operIds).map((t) => ({
      ...t,
      xaxis: xName,
      yaxis: yName,
      showlegend: i === 0 && (t as { showlegend?: boolean }).showlegend !== false,
    }));
    data.push(...traces);

    const convPlans = entry.result.conversion_plans ?? [];
    const maxEnd = Math.max(
      ...entry.result.schedule.map((r) => r.END_TM),
      ...convPlans.map((p) => p.conv_end_min),
      1,
    );
    const convTraces = conversionTraces(convPlans, maxEnd + 1).map((t) => ({
      ...t,
      xaxis: xName,
      yaxis: yName,
      showlegend: false,
    }));
    data.push(...convTraces);
    if (i === 0 && convTraces.length > 0) {
      data.push({
        ...conversionLegendTrace(true),
        xaxis: xName,
        yaxis: yName,
      });
    }

    (layout as Record<string, unknown>)[xKey] = {
      domain: [0, 1],
      anchor: yName,
      title: i === n - 1 ? { text: "시뮬레이션 시간 (분)", font: { size: 12, color: GANTT_THEME.axisColor } } : undefined,
      showgrid: true,
      gridcolor: GANTT_THEME.gridColor,
      gridwidth: GANTT_THEME.gridWidth,
      zeroline: false,
      tickfont: { size: 11, color: GANTT_THEME.axisColor },
      range: [timeStart, timeEnd],
      ...(axis.fixedRange ? { fixedrange: true } : {}),
    };
    (layout as Record<string, unknown>)[yKey] = {
      domain,
      anchor: xName,
      title: { text: "설비(EQP)", font: { size: 12, color: GANTT_THEME.axisColor } },
      tickfont: { size: 11, color: GANTT_THEME.axisColor },
      categoryorder: "array",
      categoryarray: eqps,
      showgrid: false,
    };

    const yMid = (domain[0] + domain[1]) / 2;
    layout.annotations = [
      ...(layout.annotations ?? []),
      {
        text: entry.label,
        xref: "paper",
        yref: "paper",
        x: 0,
        y: yMid,
        xanchor: "left",
        yanchor: "middle",
        showarrow: false,
        font: { size: 13, color: ALGO_CHART_COLORS[entry.algorithm] ?? "#333" },
      },
    ];
  });

  return { data, layout };
}

export type TestMetricKey =
  | "makespan"
  | "idle_total"
  | "oper_switches"
  | "prod_switches"
  | "avg_achievement";

export interface TestMetricDef {
  key: TestMetricKey;
  label: string;
  yTitle: string;
}

export const TEST_METRICS: TestMetricDef[] = [
  { key: "makespan", label: "Makespan", yTitle: "분" },
  { key: "idle_total", label: "Idle 합계", yTitle: "분" },
  { key: "oper_switches", label: "공정 전환", yTitle: "횟수" },
  { key: "prod_switches", label: "제품 전환", yTitle: "횟수" },
  { key: "avg_achievement", label: "평균 달성률", yTitle: "%" },
];

export interface TestBenchmarkChartRow {
  input_folder: string;
  label: string;
  entries: AlgoCompareEntry[];
}

function metricValue(result: InferenceResult, key: TestMetricKey): number | null {
  if (!result.schedule?.length) {
    return null;
  }
  const s = resultScheduleStats(result);
  switch (key) {
    case "makespan":
      return s.makespan;
    case "idle_total":
      return s.idle_total;
    case "oper_switches":
      return s.oper_switches;
    case "prod_switches":
      return s.prod_switches;
    case "avg_achievement": {
      const vals = Object.values(s.achievement);
      return vals.length ? Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10 : 0;
    }
    default:
      return null;
  }
}

export function buildTestMetricChart(
  metric: TestMetricDef,
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
  algoLabels: Record<string, string>,
  selectedLabel?: string,
): { data: Data[]; layout: Partial<Layout> } {
  if (rows.length < 2) {
    return buildTestMetricSingleDatasetChart(metric, rows[0], algorithms, algoLabels);
  }
  return buildTestMetricLineChart(metric, rows, algorithms, algoLabels, selectedLabel);
}

function buildTestMetricSingleDatasetChart(
  metric: TestMetricDef,
  row: TestBenchmarkChartRow | undefined,
  algorithms: string[],
  algoLabels: Record<string, string>,
): { data: Data[]; layout: Partial<Layout> } {
  const algoNames = algorithms.map((a) => algoLabels[a] ?? a);
  const values = algorithms.map((algo) => {
    const entry = row?.entries.find((e) => e.algorithm === algo);
    return entry ? metricValue(entry.result, metric.key) : 0;
  });

  return {
    data: [{
      type: "bar" as const,
      x: algoNames,
      y: values,
      marker: { color: algorithms.map((a) => ALGO_CHART_COLORS[a] ?? "#888") },
      text: values.map((v) => String(v ?? "")),
      textposition: "outside" as const,
      hovertemplate: `<b>%{x}</b><br>${metric.label}: %{y}<extra></extra>`,
      showlegend: false,
    }],
    layout: {
      title: { text: `${metric.label} · ${row?.label ?? ""}`, font: { size: 14 } },
      xaxis: { title: { text: "알고리즘" } },
      yaxis: { title: { text: metric.yTitle }, rangemode: "tozero" },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      height: 340,
      margin: { t: 50, b: 60, l: 55, r: 20 },
    },
  };
}

function buildTestMetricLineChart(
  metric: TestMetricDef,
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
  algoLabels: Record<string, string>,
  selectedLabel?: string,
): { data: Data[]; layout: Partial<Layout> } {
  const categories = rows.map((r) => r.input_folder);
  const tickText = rows.map((r) => r.label);
  const activeAlgos = algorithms.filter((algo) =>
    rows.some((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry && metricValue(entry.result, metric.key) != null;
    }),
  );
  const data: Data[] = activeAlgos.map((algo) => ({
    type: "scatter" as const,
    mode: "lines+markers" as const,
    name: algoLabels[algo] ?? algo,
    x: categories,
    y: rows.map((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry ? metricValue(entry.result, metric.key) : null;
    }),
    connectgaps: false,
    marker: {
      size: 9,
      color: ALGO_CHART_COLORS[algo] ?? "#888",
      line: { width: 1, color: "#fff" },
    },
    line: { color: ALGO_CHART_COLORS[algo] ?? "#888", width: 2 },
    hovertemplate:
      `<b>%{customdata}</b><br>${algoLabels[algo] ?? algo}: %{y}<extra></extra>`,
    customdata: tickText,
  }));

  const selectedIdx = selectedLabel
    ? rows.findIndex((r) => r.label === selectedLabel || r.input_folder.endsWith(selectedLabel))
    : -1;
  const selectedCategory = selectedIdx >= 0 ? categories[selectedIdx] : undefined;

  return {
    data,
    layout: {
      title: { text: metric.label, font: { size: 14 } },
      xaxis: {
        title: { text: "데이터셋 (RULE_TIMEKEY)" },
        tickangle: -35,
        type: "category",
        categoryorder: "array",
        categoryarray: categories,
        tickvals: categories,
        ticktext: tickText,
      },
      yaxis: { title: { text: metric.yTitle }, rangemode: "tozero" },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      legend: { orientation: "h", y: -0.35 },
      height: 340,
      margin: { t: 50, b: 90, l: 55, r: 20 },
      ...(selectedCategory
        ? {
            shapes: [{
              type: "line",
              xref: "x",
              yref: "paper",
              x0: selectedCategory,
              x1: selectedCategory,
              y0: 0,
              y1: 1,
              line: { color: "rgba(79, 110, 247, 0.45)", width: 2, dash: "dot" },
            }],
          }
        : {}),
    },
  };
}

export function benchmarkRowsFromResponse(
  datasets: TestBenchmarkDataset[],
  algoLabels: Record<string, string>,
): TestBenchmarkChartRow[] {
  const byFolder = new Map<string, TestBenchmarkChartRow>();
  for (const d of datasets) {
    if (!d.results.length) continue;
    byFolder.set(d.input_folder, {
      input_folder: d.input_folder,
      label: d.label,
      entries: d.results.map((r) => ({
        algorithm: r.algorithm ?? "rl",
        label: algoLabels[r.algorithm ?? "rl"] ?? (r.algorithm ?? "rl"),
        result: r,
      })),
    });
  }
  return [...byFolder.values()].sort((a, b) => a.input_folder.localeCompare(b.input_folder));
}

const TRAIN_CHART_BASE: Partial<Layout> = {
  plot_bgcolor: "white",
  paper_bgcolor: "white",
  height: 300,
  margin: { t: 44, b: 48, l: 55, r: 20 },
  xaxis: { title: { text: "Timesteps" } },
};

export function buildTrainRewardChart(
  series: TrainSeries,
): { data: Data[]; layout: Partial<Layout> } {
  const data: Data[] = [];
  if (series.timesteps.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Rollout ep_rew_mean",
      x: series.timesteps,
      y: series.ep_rew_mean,
      line: { color: "#4f6ef7", width: 2 },
      hovertemplate: "step %{x:,}<br>reward %{y:.2f}<extra></extra>",
    });
  }
  if (series.eval_timesteps.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines+markers",
      name: "Eval mean reward",
      x: series.eval_timesteps,
      y: series.eval_reward,
      line: { color: "#e67e22", width: 2 },
      marker: { size: 7 },
      hovertemplate: "eval @ %{x:,}<br>reward %{y:.2f}<extra></extra>",
    });
  }
  return {
    data,
    layout: {
      ...TRAIN_CHART_BASE,
      title: { text: "보상 수렴", font: { size: 14 } },
      yaxis: { title: { text: "Reward" } },
      showlegend: data.length > 1,
      legend: { orientation: "h", y: 1.12, x: 0 },
    },
  };
}

export function buildTrainLossChart(
  series: TrainSeries,
): { data: Data[]; layout: Partial<Layout> } {
  const data: Data[] = [];
  if (series.timesteps.length > 0 && series.policy_loss.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Policy loss",
      x: series.timesteps,
      y: series.policy_loss,
      line: { color: "#9b59b6", width: 2 },
    });
  }
  if (series.timesteps.length > 0 && series.value_loss.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Value loss",
      x: series.timesteps,
      y: series.value_loss,
      line: { color: "#16a085", width: 2 },
    });
  }
  return {
    data,
    layout: {
      ...TRAIN_CHART_BASE,
      title: { text: "Loss", font: { size: 14 } },
      yaxis: { title: { text: "Loss" } },
      showlegend: data.length > 1,
      legend: { orientation: "h", y: 1.12, x: 0 },
    },
  };
}

export function buildTrainExplainedVarChart(
  series: TrainSeries,
): { data: Data[]; layout: Partial<Layout> } {
  const data: Data[] = [];
  if (series.timesteps.length > 0 && series.explained_variance.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Explained variance",
      x: series.timesteps,
      y: series.explained_variance,
      line: { color: "#34495e", width: 2 },
      fill: "tozeroy",
      fillcolor: "rgba(52, 73, 94, 0.08)",
    });
  }
  return {
    data,
    layout: {
      ...TRAIN_CHART_BASE,
      title: { text: "Explained variance", font: { size: 14 } },
      yaxis: { title: { text: "Variance" }, range: [-0.05, 1.05] },
      showlegend: false,
    },
  };
}

export function hasTrainChartData(series: TrainSeries): boolean {
  return (
    series.timesteps.length > 0
    || series.eval_timesteps.length > 0
  );
}
