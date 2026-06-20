import type { Data, Layout } from "plotly.js";
import type { HistorySnap, InferenceResult, PlanRecord, ScheduleRecord } from "../types";
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
      marker: {
        color: prodColorMap[rec.PLAN_PROD_KEY] ?? "#888888",
        opacity: visible ? 1 : 0.15,
        line: { color: operColorMap[rec.OPER_ID ?? ""] ?? "#222222", width: 3 },
      },
      text: visible ? rec.LOT_ID : "",
      textposition: "inside",
      insidetextanchor: "middle",
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

function buildGanttLayout(
  title: string,
  axis: GanttAxisOptions,
): Partial<Layout> {
  const eqps = sortedEqpIds(axis.eqpIds);
  const [timeStart, timeEnd] = resolveGanttTimeRange(axis);

  return {
    title: { text: title, font: { size: 16 } },
    xaxis: {
      title: { text: "시뮬레이션 시간 (분)" },
      showgrid: true,
      gridcolor: "#E5E5E5",
      range: [timeStart, timeEnd],
      ...(axis.fixedRange ? { fixedrange: true } : {}),
    },
    yaxis: {
      categoryorder: "array",
      categoryarray: eqps,
      title: { text: "설비(EQP)" },
    },
    barmode: "overlay",
    legend: { title: { text: "제품 / 공정" }, orientation: "v", x: 1.02 },
    height: Math.max(350, 60 * Math.max(eqps.length, 1)),
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    margin: { l: 80, r: 180, t: 60, b: 60 },
  };
}

export function buildStepGantt(
  history: HistorySnap[],
  step: number,
  prodKeys: string[],
  operIds: string[],
  axis: GanttAxisOptions,
): { data: Data[]; layout: Partial<Layout> } {
  if (!history.length) {
    return {
      data: legendTraces(prodKeys, operIds, []),
      layout: buildGanttLayout("스케줄 간트 차트", axis),
    };
  }
  const snap = history[Math.min(step, history.length - 1)];
  const schedule = snap.schedule;
  return {
    data: ganttTraces(schedule, prodKeys, operIds, schedule.length - 1),
    layout: buildGanttLayout(
      `Post-Scheduling 간트 (스텝 ${snap.step} / 시각 ${snap.time}분)`,
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
      { type: "bar", name: "Post-Scheduling", x: metrics, y: [postS.makespan, postS.idle_total, postS.oper_switches, postS.prod_switches], marker: { color: "#55A868" } },
    ],
    layout: {
      title: { text: "초기 스케줄 vs Post-Scheduling KPI 비교" },
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
      { type: "bar", name: "Post-Scheduling", x: labels, y: labels.map((l) => postS.achievement[l] ?? 0), marker: { color: "#55A868" } },
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
    height: Math.max(280 * n, 400),
    barmode: "overlay",
    showlegend: n === 1,
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    legend: { x: 1.02 },
    annotations: [],
    margin: { l: 80, r: 180, t: 40, b: 50 },
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

    (layout as Record<string, unknown>)[xKey] = {
      domain: [0, 1],
      anchor: yName,
      title: i === n - 1 ? { text: "시뮬레이션 시간 (분)" } : undefined,
      showgrid: true,
      gridcolor: "#E5E5E5",
      range: [timeStart, timeEnd],
      ...(axis.fixedRange ? { fixedrange: true } : {}),
    };
    (layout as Record<string, unknown>)[yKey] = {
      domain,
      anchor: xName,
      title: { text: "설비(EQP)" },
      categoryorder: "array",
      categoryarray: eqps,
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
