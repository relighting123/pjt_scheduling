import type { Data, Layout } from "plotly.js";
import type { ConversionPlan, InferenceResult, InferenceStats, PlanRecord, ScheduleRecord, TestBenchmarkDataset, TrainSeries } from "../types";
import { buildColorMap } from "./colors";
import { buildShortCodeMap } from "./ganttLabels";
import {
  formatGanttMinuteLabel,
  ganttAxisValue,
  ganttBarAxisCoords,
  ganttTickFormat,
  ganttXMinClamp,
  minutesToTimestamp,
  parseSimBaseMs,
} from "./ganttTime";
import { computeInferenceKpi } from "./metrics";
import type { EqpUtil, EqpScheduleSummary, ModelUtil, TatRow, AchievementRow } from "./metrics";

export type GanttBarLabel = "lot" | "car" | "prod";

export interface GanttAxisOptions {
  eqpIds: string[];
  timeStartMinutes?: number;
  timeEndMinutes: number;
  /** true면 timeStartMinutes~timeEndMinutes 구간으로 X축 고정 */
  fixedRange?: boolean;
  /** 시뮬 기준 시각 (RULE_TIMEKEY, API summary.sim_base_time) */
  simBaseTime?: string;
}

export interface PlotChartSpec {
  data: Data[];
  layout: Partial<Layout>;
  clampXMin?: number;
}

function resolveGanttBaseMs(axis: GanttAxisOptions): number | null {
  return parseSimBaseMs(axis.simBaseTime);
}

export function resolveGanttTimeRange(axis: GanttAxisOptions): [number, number] {
  const start = Math.max(0, axis.timeStartMinutes ?? 0);
  if (axis.fixedRange) {
    const end = Math.max(start + 1, axis.timeEndMinutes ?? 1);
    return [start, end];
  }
  const end = Math.max(axis.timeEndMinutes ?? 0, start + 1, 1);
  return [0, end];
}

/** 스케줄 배열들에서 최대 END_TM(분) */
export function maxScheduleEndMinutes(schedules: ScheduleRecord[][]): number {
  const ends = schedules.flat().map((r) => r.END_TM);
  return ends.length ? Math.max(...ends) : 0;
}

/** 알고리즘 비교 응답 기준 X축 종료(분) – sim_end와 실제 스케줄 끝 중 큰 값 */
export function resolveCompareTimeEndMinutes(
  compareData: {
    sim_end_minutes?: number;
    results?: { schedule: ScheduleRecord[] }[];
  } | null,
): number {
  const schedules = compareData?.results?.map((r) => r.schedule ?? []) ?? [];
  const maxSched = maxScheduleEndMinutes(schedules);
  return Math.max(compareData?.sim_end_minutes ?? 0, maxSched, 1);
}

/** 간트 전용 팔레트 — 앱 UI accent 톤 */
const GANTT_PROD_COLORS = [
  "#3b6ef0", "#6366f1", "#0ea5e9", "#14b8a6", "#22c55e",
  "#84cc16", "#d97706", "#f97316", "#ef4444", "#ec4899",
];

const GANTT_OPER_BORDERS = [
  "#1d4ed8", "#4338ca", "#0369a1", "#0f766e", "#15803d",
  "#4d7c0f", "#b45309", "#c2410c", "#b91c1c", "#be185d",
];

/** 간트·차트 공통 폰트 (한글 지원) */
export const CHART_FONT = "'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif";

/** 간트 공통 스타일 — 앱 UI 톤 */
const GANTT_THEME = {
  plotBg: "#eef2f7",
  paperBg: "#ffffff",
  gridColor: "rgba(15, 23, 42, 0.12)",
  gridWidth: 1,
  fontFamily: CHART_FONT,
  titleColor: "#1b1b18",
  axisColor: "#3d3d38",
  barRadius: 0,
  barOpacity: 1.0,
  convFill: "#fbbf24",
  convBorder: "#b45309",
  rowBorderColor: "rgba(15, 23, 42, 0.22)",
  rowGridColor: "rgba(15, 23, 42, 0.13)",
} as const;

/** Plotly hover 툴팁 공통 스타일 */
export const CHART_HOVERLABEL: NonNullable<Layout["hoverlabel"]> = {
  bgcolor: "#ffffff",
  bordercolor: "rgba(15, 23, 42, 0.14)",
  font: {
    family: CHART_FONT,
    size: 12,
    color: "#1b1b18",
  },
  align: "left",
  namelength: -1,
};

const GANTT_HOVERLABEL = CHART_HOVERLABEL;

const GANTT_LAYOUT_FONT: NonNullable<Layout["font"]> = {
  family: GANTT_THEME.fontFamily,
  color: GANTT_THEME.titleColor,
  size: 12,
};

const GANTT_PROD_OPER_PALETTE = [...GANTT_PROD_COLORS, ...GANTT_OPER_BORDERS];

function ganttProdOperKey(prodKey: string, operId: string): string {
  return `${prodKey}|${operId}`;
}

function ganttProdOperPairs(
  schedule: ScheduleRecord[],
  prodKeys: string[],
  operIds: string[],
): string[] {
  const seen = new Set<string>();
  for (const r of schedule) {
    seen.add(ganttProdOperKey(r.PLAN_PROD_ATTR_VAL, r.OPER_ID ?? ""));
  }
  if (seen.size > 0) return [...seen].sort();
  const pairs: string[] = [];
  for (const pk of prodKeys) {
    for (const op of operIds) {
      pairs.push(ganttProdOperKey(pk, op));
    }
  }
  return pairs.sort();
}

function ganttProdOperColorMap(pairKeys: string[]): Record<string, string> {
  return buildColorMap(pairKeys, GANTT_PROD_OPER_PALETTE);
}

// LOT_STAT_CD 강제 배정(PROC/LOAD/SELE/RESV) 색상. WAIT은 기존 제품×공정 색상 유지.
const FORCED_LOT_STAT_FILL: Record<string, string> = {
  PROC: "#16a34a",
  LOAD: "#ca8a04",
  SELE: "#ca8a04",
  RESV: "#ca8a04",
};

function forcedLotStatFillColor(lotStatCd?: string): string | undefined {
  return lotStatCd ? FORCED_LOT_STAT_FILL[lotStatCd] : undefined;
}

function ganttBarMarker(fillColor: string, visible: boolean) {
  return {
    color: fillColor,
    opacity: visible ? GANTT_THEME.barOpacity : 0.14,
    line: {
      color: visible ? "rgba(0, 0, 0, 0.50)" : "rgba(148, 163, 184, 0.2)",
      width: visible ? 1.5 : 0,
    },
    cornerradius: GANTT_THEME.barRadius,
  };
}

function conversionBarMarker() {
  return {
    color: GANTT_THEME.convFill,
    opacity: 0.9,
    line: { color: GANTT_THEME.convBorder, width: 1.25 },
    cornerradius: GANTT_THEME.barRadius,
  } as Record<string, unknown>;
}

/** 간트 X축: 0 미만 pan 방지, 상단 눈금, sim base 있으면 시각(HH:mm) */
function ganttXAxisLayout(
  timeStart: number,
  timeEnd: number,
  extra: Record<string, unknown> = {},
  fixedRange?: boolean,
  baseMs: number | null = null,
): Record<string, unknown> {
  const start = Math.max(0, timeStart);
  const end = Math.max(start + 1, timeEnd);

  if (baseMs != null) {
    return {
      side: "top",
      type: "date",
      showgrid: true,
      gridcolor: GANTT_THEME.gridColor,
      gridwidth: GANTT_THEME.gridWidth,
      zeroline: false,
      range: [minutesToTimestamp(start, baseMs), minutesToTimestamp(end, baseMs)],
      tickformat: ganttTickFormat(end - start),
      hoverformat: "%H:%M",
      tickfont: { size: 11, color: GANTT_THEME.axisColor },
      ...(fixedRange ? { fixedrange: true } : {}),
      ...extra,
    };
  }

  return {
    side: "top",
    showgrid: true,
    gridcolor: GANTT_THEME.gridColor,
    gridwidth: GANTT_THEME.gridWidth,
    zeroline: false,
    range: [start, end],
    rangemode: "nonnegative",
    minallowed: 0,
    tickfont: { size: 11, color: GANTT_THEME.axisColor },
    ...(fixedRange ? { fixedrange: true } : {}),
    ...extra,
  };
}

function ganttXAxisTitle(baseMs: number | null) {
  return {
    text: baseMs != null ? "시각" : "시뮬레이션 시간 (분)",
    font: { size: 12, color: GANTT_THEME.axisColor },
    standoff: 8,
  };
}

/** 간트 Y축 + 장비|바 구분선·행 구분선(표 느낌) */
function ganttYAxisLayout(
  categoryarray: string[],
  title?: { text: string; font?: { size: number; color: string } },
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    categoryorder: "array",
    categoryarray,
    title,
    tickfont: { size: 11, color: GANTT_THEME.axisColor },
    showgrid: false,
    showline: true,
    linecolor: GANTT_THEME.rowBorderColor,
    linewidth: 1,
    zeroline: false,
    fixedrange: true,
    ...extra,
  };
}

function ganttEqpBarDividerShape(): NonNullable<Layout["shapes"]>[number] {
  return {
    type: "line",
    xref: "paper",
    yref: "paper",
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    line: { color: GANTT_THEME.rowBorderColor, width: 1.25 },
    layer: "below",
  };
}

function ganttRowDividerShapes(rowCount: number, yAxis = "y"): NonNullable<Layout["shapes"]> {
  if (rowCount <= 0) return [];
  const line = { color: GANTT_THEME.rowGridColor, width: 1 };
  const yref = yAxis as NonNullable<NonNullable<Layout["shapes"]>[number]["yref"]>;
  const shapes: NonNullable<Layout["shapes"]> = [];
  for (let i = 0; i <= rowCount; i++) {
    shapes.push({
      type: "line",
      xref: "paper",
      yref,
      x0: 0,
      x1: 1,
      y0: i - 0.5,
      y1: i - 0.5,
      line,
      layer: "below",
    });
  }
  return shapes;
}

function ganttTableGridShapes(rowCount: number, yAxis = "y"): NonNullable<Layout["shapes"]> {
  return [ganttEqpBarDividerShape(), ...ganttRowDividerShapes(rowCount, yAxis)];
}

/** 스텝 디버거의 현재 스텝 시각을 간트 X축에 표시하는 세로 마커 (동기화용). */
export function ganttStepMarkerShape(
  simTime: number | undefined | null,
  axis: Pick<GanttAxisOptions, "simBaseTime">,
): NonNullable<Layout["shapes"]>[number] | null {
  if (simTime == null) return null;
  const baseMs = resolveGanttBaseMs(axis as GanttAxisOptions);
  const x = ganttAxisValue(simTime, baseMs);
  return {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    line: { color: "#dc2626", width: 2, dash: "dot" },
    layer: "above",
  };
}

function sortedEqpIds(eqpIds: string[]): string[] {
  return [...eqpIds].sort();
}

function legendTraces(
  prodKeys: string[],
  operIds: string[],
  schedule: ScheduleRecord[],
  prodCodes?: Record<string, string>,
  operCodes?: Record<string, string>,
): Data[] {
  const pairs = ganttProdOperPairs(schedule, prodKeys, operIds);
  const colorMap = ganttProdOperColorMap(pairs);

  return pairs.map((pairKey) => {
    const sep = pairKey.indexOf("|");
    const pk = sep >= 0 ? pairKey.slice(0, sep) : pairKey;
    const op = sep >= 0 ? pairKey.slice(sep + 1) : "";
    const prodCode = prodCodes?.[pk] ?? pk;
    const operCode = operCodes?.[op] ?? op;
    const inSchedule = schedule.some(
      (r) => ganttProdOperKey(r.PLAN_PROD_ATTR_VAL, r.OPER_ID ?? "") === pairKey,
    );
    return {
      type: "scatter",
      mode: "markers",
      x: [null],
      y: [null],
      name: `${prodCode}/${operCode}`,
      marker: {
        size: 10,
        color: colorMap[pairKey] ?? "#94a3b8",
        symbol: "square",
        line: { width: 0 },
      },
      showlegend: true,
      hoverinfo: "skip",
      visible: inSchedule ? true : "legendonly",
    } as Data;
  });
}

function ganttTraces(
  schedule: ScheduleRecord[],
  prodKeys: string[],
  operIds: string[],
  highlightMax?: number,
  baseMs: number | null = null,
): Data[] {
  const pairs = ganttProdOperPairs(schedule, prodKeys, operIds);
  const prodOperColorMap = ganttProdOperColorMap(pairs);
  const traces: Data[] = [];

  schedule.forEach((rec, idx) => {
    const visible = highlightMax === undefined || idx <= highlightMax;
    const width = rec.END_TM - rec.START_TM;
    const { base, x } = ganttBarAxisCoords(rec.START_TM, width, baseMs);
    const pairKey = ganttProdOperKey(rec.PLAN_PROD_ATTR_VAL, rec.OPER_ID ?? "");
    const forcedColor = forcedLotStatFillColor(rec.LOT_STAT_CD);
    traces.push({
      type: "bar",
      orientation: "h",
      x: [x],
      y: [rec.EQP_ID],
      base: [base],
      marker: ganttBarMarker(forcedColor ?? prodOperColorMap[pairKey] ?? "#94a3b8", visible),
      text: visible && width >= 20 ? rec.LOT_ID : "",
      textposition: "inside",
      insidetextanchor: "middle",
      textfont: { size: 10, color: "#ffffff", family: GANTT_THEME.fontFamily },
      hovertemplate:
        `<b>LOT: ${rec.LOT_ID}</b><br>` +
        `EQP: ${rec.EQP_ID}<br>` +
        `제품: ${rec.PLAN_PROD_ATTR_VAL}<br>` +
        `공정: ${rec.OPER_ID ?? "N/A"}<br>` +
        (rec.LOT_STAT_CD ? `상태: ${rec.LOT_STAT_CD}<br>` : "") +
        `시작: ${formatGanttMinuteLabel(rec.START_TM, baseMs)}<br>` +
        `종료: ${formatGanttMinuteLabel(rec.END_TM, baseMs)}<br>` +
        `소요: ${width}분<extra></extra>`,
      showlegend: false,
    } as Data);
  });

  return [...traces, ...legendTraces(prodKeys, operIds, schedule)];
}

function conversionLegendTrace(hasConversion: boolean): Data {
  return {
    type: "scatter",
    mode: "markers",
    x: [null],
    y: [null],
    name: "Conversion",
    marker: {
      size: 10,
      color: GANTT_THEME.convFill,
      symbol: "square",
      line: { color: GANTT_THEME.convBorder, width: 1.25 },
    },
    showlegend: true,
    hoverinfo: "skip",
    visible: hasConversion ? true : "legendonly",
  };
}

/** 전환 전이 텍스트: "LC_A / T650 → LC_B / T600" (TEMP 있으면 함께 표기) */
function conversionTransitionText(p: ConversionPlan): string {
  const side = (lc?: string, tp?: string) =>
    tp ? `${lc ?? "-"} / ${tp}` : `${lc ?? "-"}`;
  return `${side(p.from_lot_cd, p.from_temp)} → ${side(p.to_lot_cd, p.to_temp)}`;
}

function conversionTraces(
  plans: ConversionPlan[],
  visibleUntilTime: number,
  baseMs: number | null = null,
): Data[] {
  return plans
    .filter((p) => p.conv_start_min < visibleUntilTime)
    .map((p) => {
      const end = Math.min(p.conv_end_min, visibleUntilTime);
      const width = Math.max(end - p.conv_start_min, 0);
      if (width <= 0) return null;
      const { base, x } = ganttBarAxisCoords(p.conv_start_min, width, baseMs);
      return {
        type: "bar",
        orientation: "h",
        x: [x],
        y: [p.eqp_id],
        base: [base],
        marker: conversionBarMarker(),
        text: `CONV`,
        textposition: "inside",
        insidetextanchor: "middle",
        textfont: { size: 10, color: "#1e293b", family: GANTT_THEME.fontFamily },
        hovertemplate:
          `<b>Conversion</b><br>` +
          `EQP: ${p.eqp_id}<br>` +
          `${conversionTransitionText(p)}<br>` +
          `시작: ${formatGanttMinuteLabel(p.conv_start_min, baseMs)}<br>` +
          `종료: ${formatGanttMinuteLabel(p.conv_end_min, baseMs)}<br>` +
          `소요: ${p.conv_end_min - p.conv_start_min}분<extra></extra>`,
        showlegend: false,
      } as Data;
    })
    .filter((t): t is Data => t !== null);
}

const GANTT_PAN_LAYOUT: Pick<Layout, "dragmode"> = { dragmode: "pan" };

const GANTT_LEGEND: Partial<Layout>["legend"] = {
  orientation: "h",
  x: 0.5,
  xanchor: "center",
  y: -0.14,
  yanchor: "top",
  bgcolor: "rgba(255,255,255,0.96)",
  bordercolor: "rgba(15, 23, 42, 0.08)",
  borderwidth: 1,
  font: { size: 11, color: "#5c5c58", family: CHART_FONT },
};

const SHARED_DARK: Partial<Layout> = {
  plot_bgcolor: "#f8fafc",
  paper_bgcolor: "#ffffff",
  font: { family: CHART_FONT, color: "#1b1b18" },
  hoverlabel: CHART_HOVERLABEL,
  dragmode: false,
  xaxis: { gridcolor: "rgba(15,23,42,0.07)", color: "#475569", zerolinecolor: "rgba(15,23,42,0.12)" },
  yaxis: { gridcolor: "rgba(15,23,42,0.07)", color: "#475569", zerolinecolor: "rgba(15,23,42,0.12)" },
};

function mergeSharedLayout(extra: Partial<Layout>): Partial<Layout> {
  const xaxis = { ...SHARED_DARK.xaxis, ...extra.xaxis } as Partial<Layout>["xaxis"];
  const yaxis = { ...SHARED_DARK.yaxis, ...extra.yaxis } as Partial<Layout>["yaxis"];
  return { ...SHARED_DARK, ...extra, xaxis, yaxis };
}

function opersForProduct(
  plan: PlanRecord[],
  prod: string,
  schedule: ScheduleRecord[],
  operOrder: string[],
): string[] {
  const fromPlan = [...new Set(
    plan.filter((p) => p.plan_prod_key === prod).map((p) => p.oper_id),
  )];
  const fromSched = [...new Set(
    schedule
      .filter((r) => r.PLAN_PROD_ATTR_VAL === prod)
      .map((r) => r.OPER_ID ?? "")
      .filter(Boolean),
  )];
  const merged = [...new Set([...fromPlan, ...fromSched])];
  const operIdx = Object.fromEntries(operOrder.map((k, i) => [k, i]));
  return merged.sort(
    (a, b) => (operIdx[a] ?? operOrder.length) - (operIdx[b] ?? operOrder.length)
      || a.localeCompare(b),
  );
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
  baseMs: number | null = null,
): { x: number[]; y: number[] } {
  const events = schedule
    .filter((r) => r.PLAN_PROD_ATTR_VAL === prod && (r.OPER_ID ?? "") === operId)
    .map((r) => ({ t: r.START_TM, q: r.WF_QTY ?? 25 }))
    .sort((a, b) => a.t - b.t || a.q - b.q);

  let cum = 0;
  const x: number[] = [ganttAxisValue(0, baseMs)];
  const y: number[] = [0];

  for (const e of events) {
    const tx = ganttAxisValue(e.t, baseMs);
    if (tx > x[x.length - 1]) {
      x.push(tx);
      y.push(cum);
    }
    cum += e.q;
    x.push(tx);
    y.push(cum);
  }

  const endX = ganttAxisValue(timeEnd, baseMs);
  if (x[x.length - 1] < endX) {
    x.push(endX);
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
  timeAxis?: Pick<GanttAxisOptions, "timeStartMinutes" | "timeEndMinutes" | "fixedRange" | "simBaseTime">;
}

export function buildProductProductionCharts(
  schedule: ScheduleRecord[],
  plan: PlanRecord[],
  prodKeys: string[],
  timeEndMinutes: number,
  options: ProductProductionChartOptions = {},
): PlotChartSpec | null {
  const prods = prodKeys.length
    ? [...prodKeys]
    : [...new Set(schedule.map((r) => r.PLAN_PROD_ATTR_VAL))].sort();
  if (!prods.length || !schedule.length) return null;

  const operOrder = options.operIds?.length
    ? options.operIds
    : [...new Set([...plan.map((p) => p.oper_id), ...schedule.map((r) => r.OPER_ID ?? "").filter(Boolean)])].sort();
  const prodCodeMap = buildShortCodeMap(prods, "P").codeByKey;
  const operCodeMap = buildShortCodeMap(operOrder, "O").codeByKey;

  const n = prods.length;
  const axisOpts: GanttAxisOptions = {
    eqpIds: [],
    timeStartMinutes: options.timeAxis?.timeStartMinutes,
    timeEndMinutes: options.timeAxis?.timeEndMinutes ?? timeEndMinutes,
    fixedRange: options.timeAxis?.fixedRange,
    simBaseTime: options.timeAxis?.simBaseTime,
  };
  const [timeStart, timeEnd] = resolveGanttTimeRange(axisOpts);
  const baseMs = resolveGanttBaseMs(axisOpts);
  const timeHover = baseMs == null ? "%{x}분" : "%{x|%H:%M}";
  const prodOperPairs = prods.flatMap((prod) =>
    opersForProduct(plan, prod, schedule, operOrder).map((oper) => ganttProdOperKey(prod, oper)),
  );
  const prodOperColorMap = ganttProdOperColorMap([...new Set(prodOperPairs)].sort());
  const data: Data[] = [];
  const layout: Partial<Layout> = {
    title: {
      text: options.title ?? "시간별 제품·공정 누적 생산",
      font: { size: 14, color: GANTT_THEME.titleColor, family: GANTT_THEME.fontFamily },
    },
    font: { family: GANTT_THEME.fontFamily, color: GANTT_THEME.axisColor },
    grid: { rows: n, columns: 1, pattern: "independent", roworder: "top to bottom" },
    height: Math.min(Math.max(260 * n, 300), 960),
    plot_bgcolor: GANTT_THEME.plotBg,
    paper_bgcolor: GANTT_THEME.paperBg,
    margin: { l: 72, r: 150, t: 88, b: 40 },
    showlegend: true,
    legend: { orientation: "v", x: 1.02, y: 1, font: { size: 11 } },
  };

  prods.forEach((prod, i) => {
    const { x: xAxis, y: yAxis } = subplotAxisNames(i);
    const xKey = (i === 0 ? "xaxis" : `xaxis${i + 1}`) as keyof Layout;
    const yKey = (i === 0 ? "yaxis" : `yaxis${i + 1}`) as keyof Layout;
    const opers = opersForProduct(plan, prod, schedule, operOrder);
    const prodCode = prodCodeMap[prod] ?? prod;

    (layout as Record<string, unknown>)[xKey] = {
      title: i === 0 ? ganttXAxisTitle(baseMs) : undefined,
      ...ganttXAxisLayout(timeStart, timeEnd, {}, options.timeAxis?.fixedRange, baseMs),
    };
    (layout as Record<string, unknown>)[yKey] = {
      title: { text: `${prodCode} 누적 생산 (매)` },
      rangemode: "tozero",
      showgrid: true,
      gridcolor: GANTT_THEME.gridColor,
      zerolinecolor: GANTT_THEME.gridColor,
    };

    opers.forEach((oper) => {
      const color = prodOperColorMap[ganttProdOperKey(prod, oper)] ?? "#64748b";
      const planQty = operPlanQty(plan, prod, oper);
      const showInLegend = i === 0;
      const operCode = operCodeMap[oper] ?? oper;
      const pairLabel = `${prodCode}/${operCode}`;
      const planXStart = ganttAxisValue(timeStart, baseMs);
      const planXEnd = ganttAxisValue(timeEnd, baseMs);

      const actual = cumulativeProductionSeries(schedule, prod, oper, timeEnd, baseMs);
      data.push({
        type: "scatter",
        mode: "lines",
        name: `${operCode} 실적`,
        x: actual.x,
        y: actual.y,
        line: { color, width: 2.5, shape: "hv" },
        xaxis: xAxis,
        yaxis: yAxis,
        legendgroup: `${prod}|${oper}-actual`,
        showlegend: showInLegend,
        hovertemplate: `${pairLabel}<br>시간: ${timeHover}<br>누적: %{y}매<extra></extra>`,
      });

      if (planQty > 0) {
        data.push({
          type: "scatter",
          mode: "lines",
          name: `${operCode} 계획`,
          x: [planXStart, planXEnd],
          y: [planQty, planQty],
          line: { color, width: 1.5, dash: "dash" },
          xaxis: xAxis,
          yaxis: yAxis,
          legendgroup: `${prod}|${oper}-plan`,
          showlegend: showInLegend,
          hovertemplate: `${pairLabel} 계획<br>목표: ${planQty}매<extra></extra>`,
        });
      }

      if (options.overlaySchedule) {
        const overlay = cumulativeProductionSeries(options.overlaySchedule, prod, oper, timeEnd, baseMs);
        data.push({
          type: "scatter",
          mode: "lines",
          name: `${operCode} ${options.overlayLabel ?? "초기"}`,
          x: overlay.x,
          y: overlay.y,
          line: { color, width: 1.5, dash: "dot", shape: "hv" },
          xaxis: xAxis,
          yaxis: yAxis,
          legendgroup: `${prod}|${oper}-overlay`,
          showlegend: showInLegend,
          hovertemplate: `${pairLabel} 초기<br>시간: ${timeHover}<br>누적: %{y}매<extra></extra>`,
        });
      }
    });
  });

  if (!data.length) return null;
  return { data, layout, clampXMin: ganttXMinClamp(baseMs) };
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
      if (recs[i].PLAN_PROD_ATTR_VAL !== recs[i - 1].PLAN_PROD_ATTR_VAL) prodSw++;
      idleTotal += Math.max(recs[i].START_TM - recs[i - 1].END_TM, 0);
    }
  });

  const completed: Record<string, number> = {};
  schedule.forEach((r) => {
    const key = `${r.PLAN_PROD_ATTR_VAL}|${r.OPER_ID ?? ""}`;
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

/** achievement 라벨 `PPK/OPER` → `P1/O1` 축약 (차트 X축용) */
function abbreviateProdOperLabels(labels: string[]): string[] {
  const prods = new Set<string>();
  const opers = new Set<string>();
  for (const label of labels) {
    const slash = label.indexOf("/");
    if (slash < 0) continue;
    prods.add(label.slice(0, slash));
    opers.add(label.slice(slash + 1));
  }
  const prodMap = buildShortCodeMap([...prods], "P").codeByKey;
  const operMap = buildShortCodeMap([...opers], "O").codeByKey;
  return labels.map((label) => {
    const slash = label.indexOf("/");
    if (slash < 0) return label;
    const prod = label.slice(0, slash);
    const oper = label.slice(slash + 1);
    const p = prodMap[prod] ?? prod;
    const o = operMap[oper] ?? oper;
    return `${p}/${o}`;
  });
}

export const ALGO_CHART_COLORS: Record<string, string> = {
  scheduling_rl: "#8172B3",
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
  const metricDefs = [
    { label: "Makespan", unit: "분", key: "makespan" as const },
    { label: "Idle 합계", unit: "분", key: "idle_total" as const },
    { label: "공정 전환", unit: "회", key: "oper_switches" as const },
    { label: "제품 전환", unit: "회", key: "prod_switches" as const },
  ];
  type RawKey = "makespan" | "idle_total" | "oper_switches" | "prod_switches";

  const algoNames = entries.map((e) => e.label);
  const rawByAlgo = entries.map((e) => {
    const s = resultScheduleStats(e.result);
    return { makespan: s.makespan, idle_total: s.idle_total, oper_switches: s.oper_switches, prod_switches: s.prod_switches };
  });

  const axisRefs = [
    { x: "x" as const, y: "y" as const, xKey: "xaxis", yKey: "yaxis" },
    { x: "x2" as const, y: "y2" as const, xKey: "xaxis2", yKey: "yaxis2" },
    { x: "x3" as const, y: "y3" as const, xKey: "xaxis3", yKey: "yaxis3" },
    { x: "x4" as const, y: "y4" as const, xKey: "xaxis4", yKey: "yaxis4" },
  ];
  const domainX: [number, number][] = [[0, 0.44], [0.56, 1.0], [0, 0.44], [0.56, 1.0]];
  const domainY: [number, number][] = [[0.57, 1.0], [0.57, 1.0], [0, 0.43], [0, 0.43]];
  const sharedAxisStyle = { gridcolor: "rgba(15,23,42,0.07)", color: "#475569", zerolinecolor: "rgba(15,23,42,0.12)" };

  const data: Data[] = metricDefs.map((metric, mi) => {
    const vals = rawByAlgo.map((r) => r[metric.key as RawKey] as number);
    return {
      type: "bar" as const,
      x: algoNames,
      y: vals,
      text: vals.map((v) => v.toLocaleString()),
      textposition: "outside" as const,
      textfont: { size: 11 },
      cliponaxis: false,
      marker: { color: entries.map((e) => ALGO_CHART_COLORS[e.algorithm] ?? "#888") },
      hovertemplate: `%{x}: %{y:,} ${metric.unit}<extra></extra>`,
      xaxis: axisRefs[mi].x,
      yaxis: axisRefs[mi].y,
      showlegend: false,
      name: metric.label,
    } as Data;
  });

  const annotations = metricDefs.map((metric, mi) => ({
    text: `<b>${metric.label} (${metric.unit})</b>`,
    x: (domainX[mi][0] + domainX[mi][1]) / 2,
    y: domainY[mi][1] + 0.005,
    xref: "paper" as const,
    yref: "paper" as const,
    showarrow: false,
    font: { size: 12 },
    xanchor: "center" as const,
    yanchor: "bottom" as const,
  }));

  const extraAxes: Record<string, unknown> = {};
  metricDefs.forEach((metric, mi) => {
    const vals = rawByAlgo.map((r) => r[metric.key as RawKey] as number);
    const maxVal = Math.max(...vals, 1);
    extraAxes[axisRefs[mi].xKey] = { ...sharedAxisStyle, domain: domainX[mi], anchor: axisRefs[mi].y, automargin: true };
    extraAxes[axisRefs[mi].yKey] = { ...sharedAxisStyle, domain: domainY[mi], anchor: axisRefs[mi].x, automargin: true, rangemode: "tozero" as const, range: [0, maxVal * 1.35], title: { text: metric.unit } };
  });

  return {
    data,
    layout: {
      ...mergeSharedLayout({
        title: { text: "알고리즘별 KPI 비교" },
        height: 520,
        margin: { t: 64, b: 56, l: 56, r: 16 },
        annotations,
      }),
      ...extraAxes,
    } as Partial<Layout>,
  };
}

export function buildAlgorithmAchievementComparison(
  entries: AlgoCompareEntry[],
): { data: Data[]; layout: Partial<Layout> } | null {
  const labelSet = new Set<string>();
  entries.forEach((e) => {
    Object.keys(resultScheduleStats(e.result).achievement).forEach((k) => labelSet.add(k));
  });
  const labels = [...labelSet].sort();
  if (!labels.length) return null;
  const shortLabels = abbreviateProdOperLabels(labels);

  const data: Data[] = entries.map((e) => {
    const ach = resultScheduleStats(e.result).achievement;
    return {
      type: "bar" as const,
      name: e.label,
      x: shortLabels,
      y: labels.map((l) => ach[l] ?? 0),
      customdata: labels,
      hovertemplate: "%{customdata}<br>달성률: %{y}%<extra></extra>",
      marker: { color: ALGO_CHART_COLORS[e.algorithm] ?? "#888" },
    };
  });

  return {
    data,
    layout: mergeSharedLayout({
      title: { text: "알고리즘별 계획 달성률 비교 (%)" },
      barmode: "group",
      yaxis: { title: { text: "달성률 (%)" }, range: [0, 120], automargin: true },
      xaxis: { title: { text: "P / O" }, automargin: true, tickangle: labels.length > 6 ? -35 : 0 },
      shapes: [{
        type: "line",
        x0: 0,
        x1: 1,
        y0: 100,
        y1: 100,
        xref: "paper",
        yref: "y",
        line: { dash: "dash", color: "red", width: 1 },
      }],
      legend: { orientation: "h", y: -0.25, x: 0.5, xanchor: "center" },
      height: Math.max(380, 48 + labels.length * 8),
      margin: { t: 60, b: labels.length > 6 ? 120 : 100, l: 56, r: 16 },
    }),
  };
}

function subplotAxisPair(index: number, total: number): {
  xName: string;
  yName: string;
  xKey: string;
  yKey: string;
  domain: [number, number];
  titleY: number;
} {
  const titleBand = total > 1 ? 0.034 : 0;
  const rowGap = total > 1 ? 0.04 : 0;
  const rowH = 1 / total;
  const rowTop = 1 - index * rowH;
  const rowBottom = rowTop - rowH;
  const plotTop = rowTop - titleBand;
  const plotBottom = rowBottom + (index < total - 1 ? rowGap : 0);
  const n = index + 1;
  return {
    xName: index === 0 ? "x" : `x${n}`,
    yName: index === 0 ? "y" : `y${n}`,
    xKey: index === 0 ? "xaxis" : `xaxis${n}`,
    yKey: index === 0 ? "yaxis" : `yaxis${n}`,
    domain: [plotBottom, plotTop],
    titleY: plotTop + 0.008,
  };
}

export function buildCompareGanttAxis(
  entries: AlgoCompareEntry[],
  compareData: { eqp_ids?: string[]; sim_end_minutes?: number } | null,
  simBaseTime?: string,
  overrides?: Pick<GanttAxisOptions, "timeStartMinutes" | "timeEndMinutes" | "fixedRange">,
): GanttAxisOptions {
  const simEnd = resolveCompareTimeEndMinutes(
    compareData
      ? {
          sim_end_minutes: compareData.sim_end_minutes,
          results: entries.map((e) => ({ schedule: e.result.schedule })),
        }
      : { results: entries.map((e) => ({ schedule: e.result.schedule })) },
  );
  const eqpFromCompare = compareData?.eqp_ids ?? [];
  const eqpFromSched = [...new Set(entries.flatMap((e) => e.result.schedule.map((r) => r.EQP_ID)))];
  const eqpIds = eqpFromCompare.length ? eqpFromCompare : eqpFromSched;
  return {
    eqpIds: sortedEqpIds(eqpIds),
    timeStartMinutes: overrides?.fixedRange ? (overrides.timeStartMinutes ?? 0) : 0,
    timeEndMinutes: overrides?.fixedRange ? (overrides.timeEndMinutes ?? simEnd) : simEnd,
    fixedRange: overrides?.fixedRange ?? false,
    simBaseTime,
  };
}

export function buildAlgorithmGanttComparison(
  entries: AlgoCompareEntry[],
  axis: GanttAxisOptions,
): PlotChartSpec {
  if (!entries.length) {
    return { data: [], layout: { height: 300 } };
  }

  const eqps = sortedEqpIds(axis.eqpIds);
  const [timeStart, timeEnd] = resolveGanttTimeRange(axis);
  const baseMs = resolveGanttBaseMs(axis);
  const n = entries.length;
  const data: Data[] = [];
  const layout: Partial<Layout> = {
    ...GANTT_PAN_LAYOUT,
    grid: { rows: n, columns: 1, pattern: "independent", roworder: "top to bottom" },
    height: Math.max(280 * n + 40, 420),
    barmode: "overlay",
    bargap: 0.20,
    showlegend: n === 1,
    plot_bgcolor: GANTT_THEME.plotBg,
    paper_bgcolor: GANTT_THEME.paperBg,
    font: GANTT_LAYOUT_FONT,
    hovermode: "closest",
    hoverdistance: 30,
    legend: n === 1 ? { title: { text: "제품×공정", font: { size: 11 } }, ...GANTT_LEGEND } : undefined,
    hoverlabel: GANTT_HOVERLABEL,
    annotations: [],
    margin: { l: 92, r: 20, t: 56, b: n === 1 ? 72 : 48 },
  };

  entries.forEach((entry, i) => {
    const { xName, yName, xKey, yKey, domain, titleY } = subplotAxisPair(i, n);
    const prodKeys = entry.result.prod_keys;
    const operIds = entry.result.oper_ids;
    const traces = ganttTraces(entry.result.schedule, prodKeys, operIds, undefined, baseMs).map((t) => ({
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
    const convTraces = conversionTraces(convPlans, maxEnd + 1, baseMs).map((t) => ({
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
      title: i === 0 ? ganttXAxisTitle(baseMs) : undefined,
      ...ganttXAxisLayout(timeStart, timeEnd, {}, axis.fixedRange, baseMs),
    };
    // y축 제목은 생략(알고리즘 라벨 주석과 좌상단에서 겹침 방지). 눈금(EQP id)만 표시.
    (layout as Record<string, unknown>)[yKey] = ganttYAxisLayout(eqps,
      undefined,
      { domain, anchor: xName },
    );

    layout.shapes = [
      ...(layout.shapes ?? []),
      ...(i === 0 ? [ganttEqpBarDividerShape()] : []),
      ...ganttRowDividerShapes(eqps.length, yName),
    ];

    layout.annotations = [
      ...(layout.annotations ?? []),
      {
        text: `<b>${entry.label}</b>`,
        xref: "paper",
        yref: "paper",
        x: 0.01,
        y: titleY,
        xanchor: "left",
        yanchor: "bottom",
        showarrow: false,
        font: { size: 13, color: ALGO_CHART_COLORS[entry.algorithm] ?? "#333" },
      },
    ];
  });

  return { data, layout, clampXMin: ganttXMinClamp(baseMs) };
}

export type TestMetricKey =
  | "makespan"
  | "oper_switches"
  | "prod_switches"
  | "tool_switches"
  | "avg_util"
  | "avg_achievement"
  | "avg_target_achievement";

export interface TestMetricDef {
  key: TestMetricKey;
  label: string;
  yTitle: string;
}

const PERCENT_METRIC_KEYS = new Set<TestMetricKey>([
  "avg_util",
  "avg_achievement",
  "avg_target_achievement",
]);

export const TEST_METRICS: TestMetricDef[] = [
  { key: "makespan", label: "Makespan", yTitle: "분" },
  { key: "oper_switches", label: "공정 전환", yTitle: "횟수" },
  { key: "prod_switches", label: "제품 전환", yTitle: "횟수" },
  { key: "tool_switches", label: "Tool 전환", yTitle: "횟수" },
  { key: "avg_util", label: "평균 가동률", yTitle: "%" },
  { key: "avg_achievement", label: "평균 달성률", yTitle: "%" },
  { key: "avg_target_achievement", label: "평균 타겟달성률", yTitle: "%" },
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
  switch (key) {
    case "makespan":
      return resultScheduleStats(result).makespan;
    case "oper_switches":
      return resultScheduleStats(result).oper_switches;
    case "prod_switches":
      return resultScheduleStats(result).prod_switches;
    case "tool_switches":
      return computeInferenceKpi(result).toolSwitches;
    case "avg_util":
      return computeInferenceKpi(result).avgUtilPct;
    case "avg_achievement": {
      const vals = Object.values(resultScheduleStats(result).achievement);
      return vals.length ? Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10 : 0;
    }
    case "avg_target_achievement":
      return computeInferenceKpi(result).avgTargetAchPct;
    default:
      return null;
  }
}

export interface MetricSummaryStat {
  avg: number;
  min: number;
  max: number;
  n: number;
}

export interface MetricSummaryRow {
  key: TestMetricKey;
  label: string;
  yTitle: string;
  perAlgo: Record<string, MetricSummaryStat>;
}

/** 전체 기간(rows)에 대해 KPI별·알고리즘별 평균/최댓값/최솟값 집계. */
export function buildMetricSummaryRows(
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
): MetricSummaryRow[] {
  return TEST_METRICS.map((metric) => {
    const perAlgo: Record<string, MetricSummaryStat> = {};
    algorithms.forEach((algo) => {
      const vals: number[] = [];
      rows.forEach((row) => {
        const entry = row.entries.find((e) => e.algorithm === algo);
        const v = entry ? metricValue(entry.result, metric.key) : null;
        if (v != null) vals.push(v);
      });
      if (vals.length) {
        const avg = Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;
        perAlgo[algo] = { avg, min: Math.min(...vals), max: Math.max(...vals), n: vals.length };
      }
    });
    return { key: metric.key, label: metric.label, yTitle: metric.yTitle, perAlgo };
  });
}

/** 요약 행(전체 기간 평균)을 알고리즘별 막대로 시각화. activeAlgos가 없으면 null (표시할 데이터 없음). */
export function buildMetricSummaryChart(
  row: MetricSummaryRow,
  algorithms: string[],
  algoLabels: Record<string, string>,
): { data: Data[]; layout: Partial<Layout> } | null {
  const activeAlgos = algorithms.filter((a) => row.perAlgo[a]);
  if (!activeAlgos.length) return null;
  const algoNames = activeAlgos.map((a) => algoLabels[a] ?? a);
  const avgs = activeAlgos.map((a) => row.perAlgo[a].avg);

  const isPercent = PERCENT_METRIC_KEYS.has(row.key);
  const maxVal = Math.max(...avgs, isPercent ? 100 : 1);
  const yRange: [number, number] = isPercent ? [0, 110] : [0, Math.max(maxVal, 1) * 1.25];

  return {
    data: [{
      type: "bar" as const,
      x: algoNames,
      y: avgs,
      marker: { color: activeAlgos.map((a) => ALGO_CHART_COLORS[a] ?? "#888") },
      text: avgs.map((v) => v.toLocaleString()),
      textposition: "outside" as const,
      textfont: { size: 12 },
      cliponaxis: false,
      hovertemplate: `<b>%{x}</b><br>평균: %{y}${row.yTitle}<extra></extra>`,
      showlegend: false,
    }],
    layout: mergeSharedLayout({
      title: { text: `${row.label} — 평균`, font: { size: 14 } },
      xaxis: { type: "category" as const, title: { text: "알고리즘" } },
      yaxis: { title: { text: row.yTitle }, range: yRange },
      height: 300,
      margin: { t: 50, b: 60, l: 55, r: 20 },
    }),
  };
}

export type TestChartType = "bar" | "line";

/** 표시할 데이터가 없으면 null (호출부에서 빈 상태 UI 처리). */
export function buildTestMetricChart(
  metric: TestMetricDef,
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
  algoLabels: Record<string, string>,
  selectedLabel?: string,
  chartType: TestChartType = "line",
): { data: Data[]; layout: Partial<Layout> } | null {
  if (rows.length < 2 || chartType === "bar") {
    return buildTestMetricBarChart(metric, rows, algorithms, algoLabels, selectedLabel);
  }
  return buildTestMetricLineChart(metric, rows, algorithms, algoLabels, selectedLabel);
}

/** 여러 기간의 값을 알고리즘별 그룹 막대로 시각화 (단일 기간이면 알고리즘별 막대 1건). */
function buildTestMetricBarChart(
  metric: TestMetricDef,
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
  algoLabels: Record<string, string>,
  selectedLabel?: string,
): { data: Data[]; layout: Partial<Layout> } | null {
  if (rows.length <= 1) {
    return buildTestMetricSingleDatasetChart(metric, rows[0], algorithms, algoLabels);
  }

  const categories = rows.map((r) => r.input_folder);
  const tickText = rows.map((r) => r.label);
  const activeAlgos = algorithms.filter((algo) =>
    rows.some((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry && metricValue(entry.result, metric.key) != null;
    }),
  );
  if (!activeAlgos.length) return null;

  const yValues = activeAlgos.map((algo) =>
    rows.map((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry ? (metricValue(entry.result, metric.key) ?? 0) : 0;
    }),
  );
  const allVals = yValues.flat();
  const maxVal = allVals.length ? Math.max(...allVals) : 1;
  const yRange: [number, number] =
    PERCENT_METRIC_KEYS.has(metric.key) ? [0, 110] : [0, Math.max(maxVal, 1) * 1.25];

  const selectedIdx = selectedLabel
    ? rows.findIndex((r) => r.label === selectedLabel || r.input_folder.endsWith(selectedLabel))
    : -1;
  const selectedCategory = selectedIdx >= 0 ? categories[selectedIdx] : undefined;

  return {
    data: activeAlgos.map((algo, ai) => ({
      type: "bar" as const,
      name: algoLabels[algo] ?? algo,
      x: categories,
      y: yValues[ai],
      marker: { color: ALGO_CHART_COLORS[algo] ?? "#888" },
      customdata: tickText,
      hovertemplate: `<b>%{customdata}</b><br>${algoLabels[algo] ?? algo}: %{y:,}<extra></extra>`,
    })),
    layout: mergeSharedLayout({
      title: { text: metric.label, font: { size: 14 } },
      barmode: "group",
      xaxis: {
        type: "category" as const,
        tickangle: -35,
        categoryorder: "array",
        categoryarray: categories,
        tickvals: categories,
        ticktext: tickText,
      },
      yaxis: { title: { text: metric.yTitle }, range: yRange, rangemode: "tozero" as const },
      legend: { orientation: "h", y: -0.44 },
      height: 340,
      margin: { t: 50, b: 120, l: 55, r: 20 },
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
    }),
  };
}

function buildTestMetricSingleDatasetChart(
  metric: TestMetricDef,
  row: TestBenchmarkChartRow | undefined,
  algorithms: string[],
  algoLabels: Record<string, string>,
): { data: Data[]; layout: Partial<Layout> } | null {
  const activeAlgos = algorithms.filter((algo) => {
    const entry = row?.entries.find((e) => e.algorithm === algo);
    return entry && metricValue(entry.result, metric.key) != null;
  });
  if (!activeAlgos.length) return null;

  const algoNames = activeAlgos.map((a) => algoLabels[a] ?? a);
  const values = activeAlgos.map((algo) => {
    const entry = row?.entries.find((e) => e.algorithm === algo);
    return entry ? (metricValue(entry.result, metric.key) ?? 0) : 0;
  });

  const maxVal = Math.max(...values, PERCENT_METRIC_KEYS.has(metric.key) ? 100 : 1);
  const yMax = PERCENT_METRIC_KEYS.has(metric.key) ? 110 : maxVal * 1.3;

  return {
    data: [{
      type: "bar" as const,
      x: algoNames,
      y: values,
      marker: { color: activeAlgos.map((a) => ALGO_CHART_COLORS[a] ?? "#888") },
      text: values.map((v) => (v != null ? v.toLocaleString() : "")),
      textposition: "outside" as const,
      textfont: { size: 12 },
      cliponaxis: false,
      hovertemplate: `<b>%{x}</b><br>${metric.label}: %{y}<extra></extra>`,
      showlegend: false,
    }],
    layout: {
      title: { text: row?.label ? `${metric.label} · ${row.label}` : metric.label, font: { size: 14 } },
      ...SHARED_DARK,
      xaxis: { ...SHARED_DARK.xaxis, type: "category" as const, title: { text: "알고리즘" } },
      yaxis: { ...SHARED_DARK.yaxis, title: { text: metric.yTitle }, range: [0, yMax] },
      height: 340,
      margin: { t: 50, b: 72, l: 55, r: 20 },
    },
  };
}

function buildTestMetricLineChart(
  metric: TestMetricDef,
  rows: TestBenchmarkChartRow[],
  algorithms: string[],
  algoLabels: Record<string, string>,
  selectedLabel?: string,
): { data: Data[]; layout: Partial<Layout> } | null {
  if (rows.length <= 1) {
    return buildTestMetricSingleDatasetChart(metric, rows[0], algorithms, algoLabels);
  }
  const categories = rows.map((r) => r.input_folder);
  const tickText = rows.map((r) => r.label);
  const activeAlgos = algorithms.filter((algo) =>
    rows.some((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry && metricValue(entry.result, metric.key) != null;
    }),
  );
  if (!activeAlgos.length) return null;
  const yValues = activeAlgos.map((algo) =>
    rows.map((row) => {
      const entry = row.entries.find((e) => e.algorithm === algo);
      return entry ? metricValue(entry.result, metric.key) : null;
    }),
  );
  const allVals = yValues.flat().filter((v): v is number => v != null);
  const maxVal = allVals.length ? Math.max(...allVals) : 1;
  const yRange: [number, number] =
    PERCENT_METRIC_KEYS.has(metric.key) ? [0, 110] : [0, Math.max(maxVal, 1) * 1.25];

  const showText = rows.length <= 8;
  const data: Data[] = activeAlgos.map((algo, ai) => ({
    type: "scatter" as const,
    mode: showText ? ("text+lines+markers" as const) : ("lines+markers" as const),
    name: algoLabels[algo] ?? algo,
    x: categories,
    y: yValues[ai],
    text: showText ? yValues[ai].map((v) => (v != null ? v.toLocaleString() : "")) : undefined,
    textposition: "top center" as const,
    textfont: { size: 10 },
    connectgaps: false,
    marker: {
      size: 9,
      color: ALGO_CHART_COLORS[algo] ?? "#888",
      line: { width: 1, color: "#fff" },
    },
    line: { color: ALGO_CHART_COLORS[algo] ?? "#888", width: 2 },
    hovertemplate:
      `<b>%{customdata}</b><br>${algoLabels[algo] ?? algo}: %{y:,}<extra></extra>`,
    customdata: tickText,
  }));

  const selectedIdx = selectedLabel
    ? rows.findIndex((r) => r.label === selectedLabel || r.input_folder.endsWith(selectedLabel))
    : -1;
  const selectedCategory = selectedIdx >= 0 ? categories[selectedIdx] : undefined;

  return {
    data,
    layout: mergeSharedLayout({
      title: { text: metric.label, font: { size: 14 } },
      xaxis: {
        tickangle: -35,
        type: "category",
        categoryorder: "array",
        categoryarray: categories,
        tickvals: categories,
        ticktext: tickText,
      },
      yaxis: { title: { text: metric.yTitle }, range: yRange, rangemode: "tozero" as const },
      legend: { orientation: "h", y: -0.44 },
      height: 340,
      margin: { t: 50, b: 120, l: 55, r: 20 },
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
    }),
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
        algorithm: r.algorithm ?? "scheduling_rl",
        label: algoLabels[r.algorithm ?? "scheduling_rl"] ?? (r.algorithm ?? "scheduling_rl"),
        result: r,
      })),
    });
  }
  return [...byFolder.values()].sort((a, b) => a.input_folder.localeCompare(b.input_folder));
}

const CHART_LIGHT = { plot_bgcolor: "#f8fafc", paper_bgcolor: "#ffffff" } as const;

const TRAIN_CHART_BASE: Partial<Layout> = {
  ...CHART_LIGHT,
  height: 300,
  margin: { t: 60, b: 48, l: 55, r: 20 },
  xaxis: { title: { text: "Timesteps" }, color: "#475569", gridcolor: "rgba(15,23,42,0.07)" },
  font: { family: CHART_FONT, color: "#1b1b18" },
};

export function buildTrainRewardChart(
  series: TrainSeries,
): { data: Data[]; layout: Partial<Layout> } {
  const data: Data[] = [];
  if (series.timesteps.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Rollout reward",
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
      legend: { orientation: "h", y: 1.18, x: 0 },
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
      legend: { orientation: "h", y: 1.18, x: 0 },
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

// ── Enhanced Gantt with model Y-axis and label mode ───────────────────────

function getEqpLabel(id: string, modelMap: Record<string, string>): string {
  const model = modelMap[id];
  return model ? `${model} / ${id}` : id;
}

function barText(
  rec: ScheduleRecord,
  mode: GanttBarLabel,
  prodCodes: Record<string, string>,
): string {
  switch (mode) {
    case "car": return rec.CARRIER_ID ?? rec.LOT_ID;
    case "prod": return prodCodes[rec.PLAN_PROD_ATTR_VAL] ?? rec.PLAN_PROD_ATTR_VAL;
    default: return rec.LOT_ID;
  }
}

interface GanttBarSegment {
  records: ScheduleRecord[];
}

function canMergeProdOperSegment(prev: ScheduleRecord, next: ScheduleRecord): boolean {
  return (
    prev.EQP_ID === next.EQP_ID
    && prev.PLAN_PROD_ATTR_VAL === next.PLAN_PROD_ATTR_VAL
    && (prev.OPER_ID ?? "") === (next.OPER_ID ?? "")
    && prev.END_TM === next.START_TM
  );
}

/** 제품 모드: 동일 EQP에서 연속된 동일 제품·공정 스케줄을 하나의 bar로 묶음 */
function buildGanttBarSegments(
  schedule: ScheduleRecord[],
  labelMode: GanttBarLabel,
): GanttBarSegment[] {
  if (labelMode !== "prod" || schedule.length === 0) {
    return schedule.map((rec) => ({ records: [rec] }));
  }

  const byEqp = new Map<string, ScheduleRecord[]>();
  for (const rec of schedule) {
    const list = byEqp.get(rec.EQP_ID) ?? [];
    list.push(rec);
    byEqp.set(rec.EQP_ID, list);
  }

  const segments: GanttBarSegment[] = [];
  for (const recs of byEqp.values()) {
    const sorted = [...recs].sort(
      (a, b) => a.START_TM - b.START_TM || a.END_TM - b.END_TM,
    );
    let current: ScheduleRecord[] = [];
    for (const rec of sorted) {
      const prev = current[current.length - 1];
      if (prev && canMergeProdOperSegment(prev, rec)) {
        current.push(rec);
      } else {
        if (current.length) segments.push({ records: current });
        current = [rec];
      }
    }
    if (current.length) segments.push({ records: current });
  }

  return segments;
}

function segmentTimeRange(segment: GanttBarSegment): { start: number; end: number; width: number } {
  const first = segment.records[0];
  const last = segment.records[segment.records.length - 1];
  const start = first.START_TM;
  const end = last.END_TM;
  return { start, end, width: end - start };
}

function inflowHoverLines(records: GanttBarSegment["records"], baseMs: number | null): string {
  // 유입 재공 = OPER_IN_TIME>0(시뮬 중 투입)만. ABSTRACT는 초기 재공 출처일 뿐 유입 아님.
  const inflowTimes = records
    .map((r) => r.OPER_IN_TIME ?? 0)
    .filter((t) => t > 0);
  if (inflowTimes.length === 0) return "";

  const lines: string[] = ["─────────────────", "유형: 유입 재공"];
  const minT = Math.min(...inflowTimes);
  const maxT = Math.max(...inflowTimes);
  const minLabel = formatGanttMinuteLabel(minT, baseMs);
  lines.push(minT === maxT
    ? `유입 가능 시각: ${minLabel}`
    : `유입 시각 범위: ${minLabel} ~ ${formatGanttMinuteLabel(maxT, baseMs)}`);
  const totalWf = records.reduce((s, r) => s + (r.WF_QTY ?? 0), 0);
  if (totalWf > 0) lines.push(`유입 재공량: ${totalWf}매`);
  return lines.join("<br>") + "<br>";
}

function segmentHoverTemplate(
  segment: GanttBarSegment,
  prodCodes: Record<string, string>,
  operCodes: Record<string, string>,
  baseMs: number | null,
): string {
  const { records } = segment;
  const first = records[0];
  const { start, end, width } = segmentTimeRange(segment);
  const oper = first.OPER_ID ?? "N/A";
  const prodLabel = prodCodes[first.PLAN_PROD_ATTR_VAL] ?? first.PLAN_PROD_ATTR_VAL;
  const operLabel = operCodes[oper] ?? oper;
  const startLabel = formatGanttMinuteLabel(start, baseMs);
  const endLabel = formatGanttMinuteLabel(end, baseMs);
  const inflowLines = inflowHoverLines(records, baseMs);

  if (records.length === 1) {
    const rec = first;
    return (
      `<b>LOT: ${rec.LOT_ID}</b><br>` +
      (rec.CARRIER_ID ? `CAR: ${rec.CARRIER_ID}<br>` : "") +
      `EQP: ${rec.EQP_ID}<br>` +
      `제품: ${prodLabel} (${rec.PLAN_PROD_ATTR_VAL})<br>` +
      `공정: ${operLabel}<br>` +
      (rec.LOT_STAT_CD ? `상태: ${rec.LOT_STAT_CD}<br>` : "") +
      `시작: ${startLabel} · 종료: ${endLabel} · 소요: ${width}분<br>` +
      inflowLines +
      `<extra></extra>`
    );
  }

  const lotSummary = records.length <= 5
    ? records.map((r) => r.LOT_ID).join(", ")
    : `${records.slice(0, 3).map((r) => r.LOT_ID).join(", ")} 외 ${records.length - 3}건`;

  return (
    `<b>제품: ${prodLabel}</b> (${first.PLAN_PROD_ATTR_VAL})<br>` +
    `공정: ${operLabel}<br>` +
    `EQP: ${first.EQP_ID}<br>` +
    `병합 LOT ${records.length}건: ${lotSummary}<br>` +
    `시작: ${startLabel} · 종료: ${endLabel} · 소요: ${width}분<br>` +
    inflowLines +
    `<extra></extra>`
  );
}

export interface GanttLegendItem {
  pairKey: string;
  prodKey: string;
  operId: string;
  label: string;
  color: string;
  inSchedule: boolean;
}

export function buildGanttLegendItems(
  schedule: ScheduleRecord[],
  prodKeys: string[],
  operIds: string[],
): GanttLegendItem[] {
  const prodCodeMap = buildShortCodeMap(prodKeys, "P").codeByKey;
  const operCodeMap = buildShortCodeMap(operIds, "O").codeByKey;
  const pairs = ganttProdOperPairs(schedule, prodKeys, operIds);
  const colorMap = ganttProdOperColorMap(pairs);

  return pairs.map((pairKey) => {
    const sep = pairKey.indexOf("|");
    const pk = sep >= 0 ? pairKey.slice(0, sep) : pairKey;
    const op = sep >= 0 ? pairKey.slice(sep + 1) : "";
    return {
      pairKey,
      prodKey: pk,
      operId: op,
      label: `${prodCodeMap[pk] ?? pk}/${operCodeMap[op] ?? op}`,
      color: colorMap[pairKey] ?? "#94a3b8",
      inSchedule: schedule.some(
        (r) => ganttProdOperKey(r.PLAN_PROD_ATTR_VAL, r.OPER_ID ?? "") === pairKey,
      ),
    };
  });
}

const ENHANCED_GANTT_BARGAP = 0.18;

function inflowLegendTrace(hasInflow: boolean): Data {
  return {
    type: "scatter",
    mode: "markers",
    x: [null],
    y: [null],
    name: "유입 재공",
    marker: {
      size: 12,
      color: "rgba(255,255,255,0)",
      symbol: "square",
      line: { color: "rgba(15,23,42,0.72)", width: 2 },
    },
    showlegend: hasInflow,
    hoverinfo: "skip",
    visible: hasInflow ? true : "legendonly",
  } as Data;
}

export function buildEnhancedGantt(
  schedule: ScheduleRecord[],
  prodKeys: string[],
  operIds: string[],
  axis: GanttAxisOptions,
  options: {
    labelMode?: GanttBarLabel;
    eqpModelMap?: Record<string, string>;
    conversionPlans?: ConversionPlan[];
    title?: string;
    hiddenProdOperKeys?: ReadonlySet<string>;
    showConversion?: boolean;
  } = {},
): PlotChartSpec {
  const {
    labelMode = "lot",
    eqpModelMap = {},
    conversionPlans = [],
    title,
    hiddenProdOperKeys,
    showConversion = true,
  } = options;
  const prodCodeMap = buildShortCodeMap(prodKeys, "P").codeByKey;
  const operCodeMap = buildShortCodeMap(operIds, "O").codeByKey;
  const pairs = ganttProdOperPairs(schedule, prodKeys, operIds);
  const prodOperColorMap = ganttProdOperColorMap(pairs);
  const [timeStart, timeEnd] = resolveGanttTimeRange(axis);
  const baseMs = resolveGanttBaseMs(axis);

  const sortedEqps = sortedEqpIds(axis.eqpIds);
  const eqpLabels = sortedEqps.map((id) => getEqpLabel(id, eqpModelMap));

  const data: Data[] = [];
  const barSegments = buildGanttBarSegments(schedule, labelMode);

  barSegments.forEach((segment) => {
    const rec = segment.records[0];
    const pairKey = ganttProdOperKey(rec.PLAN_PROD_ATTR_VAL, rec.OPER_ID ?? "");
    if (hiddenProdOperKeys?.has(pairKey)) return;
    const { start, width } = segmentTimeRange(segment);
    const label = getEqpLabel(rec.EQP_ID, eqpModelMap);
    const showText = width >= (labelMode === "prod" ? 18 : 24);
    const { base, x } = ganttBarAxisCoords(start, width, baseMs);
    // 유입 재공 = 시뮬 중 투입(OPER_IN_TIME>0)만. ABSTRACT(초기 재공 출처)는 유입이 아님.
    const isInflowSeg = segment.records.some((r) => (r.OPER_IN_TIME ?? 0) > 0);
    const forcedColor = forcedLotStatFillColor(rec.LOT_STAT_CD);
    const baseMarker = ganttBarMarker(forcedColor ?? prodOperColorMap[pairKey] ?? "#94a3b8", true);
    const marker = isInflowSeg
      ? { ...baseMarker, opacity: 0.82, line: { ...(baseMarker.line as object), width: 0 } }
      : baseMarker;
    data.push({
      type: "bar",
      orientation: "h",
      x: [x],
      y: [label],
      base: [base],
      marker,
      text: showText ? barText(rec, labelMode, prodCodeMap) : "",
      textposition: "inside",
      insidetextanchor: "middle",
      textfont: { size: 10, color: "#ffffff", family: GANTT_THEME.fontFamily },
      hovertemplate: segmentHoverTemplate(segment, prodCodeMap, operCodeMap, baseMs),
      showlegend: false,
    } as Data);
  });

  // Conversion bars
  if (showConversion) {
    conversionPlans.forEach((p) => {
      const w = Math.max(p.conv_end_min - p.conv_start_min, 0);
      if (w <= 0) return;
      const label = getEqpLabel(p.eqp_id, eqpModelMap);
      const { base, x } = ganttBarAxisCoords(p.conv_start_min, w, baseMs);
      data.push({
        type: "bar",
        orientation: "h",
        x: [x],
        y: [label],
        base: [base],
        marker: conversionBarMarker(),
        text: w >= 20 ? "CONV" : "",
        textposition: "inside",
        insidetextanchor: "middle",
        textfont: { size: 10, color: "#1e293b", family: GANTT_THEME.fontFamily },
        hovertemplate: `<b>Conversion</b><br>EQP: ${p.eqp_id}<br>${conversionTransitionText(p)}<br>` +
          `시작: ${formatGanttMinuteLabel(p.conv_start_min, baseMs)} · 종료: ${formatGanttMinuteLabel(p.conv_end_min, baseMs)}<extra></extra>`,
        showlegend: false,
      } as Data);
    });
  }

  // 유입 재공 범례
  const hasInflow = barSegments.some((seg) =>
    seg.records.some((r) => (r.OPER_IN_TIME ?? 0) > 0),
  );
  data.push(inflowLegendTrace(hasInflow));

  const layout: Partial<Layout> = {
    ...GANTT_PAN_LAYOUT,
    title: title ? { text: title, font: { size: 15, color: GANTT_THEME.titleColor, family: GANTT_THEME.fontFamily } } : undefined,
    // x축 제목 제거: 스크롤 스티키 시 눈금과 겹침 방지 + 기준 일자는 차트 상단 헤더에 표기.
    // 눈금은 RULE_TIMEKEY(=0분) 기준 실제 시각으로 표시(baseMs 있을 때).
    xaxis: {
      title: undefined,
      ...ganttXAxisLayout(timeStart, timeEnd, {}, axis.fixedRange, baseMs),
    },
    yaxis: ganttYAxisLayout(eqpLabels, {
      text: "설비 (모델 / 호기)",
      font: { size: 12, color: GANTT_THEME.axisColor },
    }, { tickfont: { size: 10, color: GANTT_THEME.axisColor } }),
    shapes: ganttTableGridShapes(eqpLabels.length),
    barmode: "overlay",
    bargap: ENHANCED_GANTT_BARGAP,
    showlegend: hasInflow,
    legend: hasInflow ? {
      title: { text: "범례", font: { size: 11 } },
      ...GANTT_LEGEND,
    } : undefined,
    height: Math.max(350, 72 * Math.max(sortedEqps.length, 1)),
    plot_bgcolor: GANTT_THEME.plotBg,
    paper_bgcolor: GANTT_THEME.paperBg,
    margin: { l: 160, r: 20, t: 48, b: hasInflow ? 72 : 40 },
    font: GANTT_LAYOUT_FONT,
    hovermode: "closest",
    hoverdistance: 30,
    hoverlabel: GANTT_HOVERLABEL,
  };

  return { data, layout, clampXMin: ganttXMinClamp(baseMs) };
}

export function buildEqpUtilChart(utils: EqpUtil[]): { data: Data[]; layout: Partial<Layout> } {
  const sorted = [...utils].sort((a, b) => b.utilPct - a.utilPct);
  const labels = sorted.map((u) => (u.model ? `${u.model}/${u.eqp_id}` : u.eqp_id));
  const values = sorted.map((u) => u.utilPct);
  const colors = values.map((v) => (v >= 80 ? "#55A868" : v >= 50 ? "#4C72B0" : "#DD8452"));

  return {
    data: [{
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      marker: { color: colors },
      text: values.map((v) => `${v}%`),
      textposition: "outside",
      hovertemplate: "%{y}<br>가동률: %{x}%<extra></extra>",
      showlegend: false,
    }],
    layout: {
      title: { text: "장비별 가동률 (%)", font: { size: 13 } },
      xaxis: { range: [0, 115], title: { text: "가동률 (%)" } },
      height: Math.max(300, 28 * Math.max(utils.length, 6)),
      margin: { l: 140, r: 60, t: 40, b: 56 },
      ...SHARED_DARK,
    },
  };
}

export function buildModelUtilChart(utils: ModelUtil[]): { data: Data[]; layout: Partial<Layout> } {
  const labels = utils.map((u) => u.model);
  const values = utils.map((u) => u.avgUtilPct);
  const colors = values.map((v) => (v >= 80 ? "#55A868" : v >= 50 ? "#4C72B0" : "#DD8452"));

  return {
    data: [{
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      marker: { color: colors },
      text: values.map((v) => `${v}%`),
      textposition: "outside",
      hovertemplate: "%{y}<br>평균 가동률: %{x}%<extra></extra>",
      showlegend: false,
    }],
    layout: {
      title: { text: "장비모델별 평균 가동률 (%)", font: { size: 13 } },
      xaxis: { range: [0, 115], title: { text: "평균 가동률 (%)" } },
      height: Math.max(260, 40 * Math.max(utils.length, 4)),
      margin: { l: 120, r: 60, t: 40, b: 56 },
      ...SHARED_DARK,
    },
  };
}

export function buildEqpIdleChart(rows: EqpScheduleSummary[]): { data: Data[]; layout: Partial<Layout> } {
  const sorted = [...rows].sort((a, b) => b.idlePct - a.idlePct);
  const labels = sorted.map((r) => (r.model ? `${r.model}/${r.eqp_id}` : r.eqp_id));
  const values = sorted.map((r) => r.idlePct);
  const colors = values.map((v) => (v <= 20 ? "#55A868" : v <= 50 ? "#DD8452" : "#C44E52"));

  return {
    data: [{
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      marker: { color: colors },
      text: values.map((v) => `${v}%`),
      textposition: "outside",
      hovertemplate: "%{y}<br>유휴율: %{x}%<extra></extra>",
      showlegend: false,
    }],
    layout: {
      title: { text: "장비별 유휴율 (%)", font: { size: 13 } },
      xaxis: { range: [0, 115], title: { text: "유휴율 (%)" } },
      height: Math.max(300, 28 * Math.max(rows.length, 6)),
      margin: { l: 140, r: 60, t: 40, b: 56 },
      ...SHARED_DARK,
    },
  };
}

export function buildTatChart(rows: TatRow[]): { data: Data[]; layout: Partial<Layout> } {
  return {
    data: [
      {
        type: "bar",
        name: "평균 TAT",
        x: rows.map((r) => r.prod),
        y: rows.map((r) => r.avgMin),
        marker: { color: "#4C72B0" },
        hovertemplate: "%{x}<br>평균 TAT: %{y}분<extra></extra>",
      },
      {
        type: "bar",
        name: "최대 TAT",
        x: rows.map((r) => r.prod),
        y: rows.map((r) => r.maxMin),
        marker: { color: "#DD8452", opacity: 0.6 },
        hovertemplate: "%{x}<br>최대 TAT: %{y}분<extra></extra>",
      },
    ],
    layout: {
      title: { text: "제품별 TAT (Turn Around Time)", font: { size: 13 } },
      barmode: "group",
      xaxis: { title: { text: "제품" } },
      yaxis: { title: { text: "TAT (분)" } },
      height: 300,
      margin: { t: 44, b: 80, l: 55, r: 20 },
      ...SHARED_DARK,
      legend: { orientation: "h", y: -0.32 },
    },
  };
}

export function buildAchievementTableChart(rows: AchievementRow[]): { data: Data[]; layout: Partial<Layout> } {
  const labels = rows.map((r) => `${r.prod}/${r.oper}`);
  const planColors = rows.map((r) => r.pct >= 100 ? "#55A868" : r.pct >= 60 ? "#DD8452" : "#C44E52");
  const targetColors = rows.map((r) => r.targetPct >= 100 ? "#3b8a5a" : r.targetPct >= 60 ? "#b86830" : "#8b2e31");
  return {
    data: [
      {
        type: "bar",
        orientation: "h",
        name: "계획달성률(D0)",
        x: rows.map((r) => Math.min(r.pct, 100)),
        y: labels,
        text: rows.map((r) => `${r.pct}%`),
        textposition: "outside",
        marker: { color: planColors },
        hovertemplate: "%{y}<br>계획달성률: %{x}% (%{customdata}매/%{meta}매)<extra></extra>",
        customdata: rows.map((r) => r.doneQty),
        meta: rows.map((r) => r.planQty),
      } as Data,
      {
        type: "bar",
        orientation: "h",
        name: "타겟달성률(D1)",
        x: rows.map((r) => Math.min(r.targetPct, 100)),
        y: labels,
        text: rows.map((r) => `${r.targetPct}%`),
        textposition: "outside",
        marker: { color: targetColors, opacity: 0.65 },
        hovertemplate: "%{y}<br>타겟달성률: %{x}% (%{customdata}매/%{meta}매)<extra></extra>",
        customdata: rows.map((r) => r.doneQty),
        meta: rows.map((r) => r.targetQty),
      } as Data,
    ],
    layout: {
      title: { text: "제품/공정별 달성률 (계획 vs 타겟)", font: { size: 13 } },
      barmode: "group",
      xaxis: { range: [0, 130], title: { text: "달성률 (%)" } },
      shapes: [{ type: "line" as const, x0: 100, x1: 100, y0: 0, y1: 1, yref: "paper" as const, line: { dash: "dash" as const, color: "#4C72B0", width: 1.5 } }],
      legend: { orientation: "h", y: -0.15 },
      height: Math.max(300, 40 * Math.max(rows.length, 5)),
      margin: { l: 140, r: 100, t: 40, b: 80 },
      ...SHARED_DARK,
    },
  };
}

export function buildInferenceWipChart(
  stats: InferenceStats,
  plan: PlanRecord[],
): { data: Data[]; layout: Partial<Layout> } {
  const completed = stats.completed_qty ?? {};
  const remaining = stats.remaining_wip ?? {};

  const labels: string[] = [];
  const done: number[] = [];
  const wipLeft: number[] = [];
  const planGap: number[] = [];
  const planQtys: number[] = [];

  plan.forEach((p) => {
    const key = `${p.plan_prod_key}|${p.oper_id}`;
    const fin = completed[key] ?? 0;
    const rem = (remaining as Record<string, number>)[key] ?? 0;
    const planQty = p.d0_plan_qty;
    labels.push(`${p.plan_prod_key}/${p.oper_id}`);
    done.push(fin);
    wipLeft.push(rem);
    planQtys.push(planQty);
    planGap.push(Math.max(planQty - fin - rem, 0));
  });

  const shortLbls = abbreviateProdOperLabels(labels);

  return {
    data: [
      {
        type: "bar",
        orientation: "h",
        name: "완료",
        x: done,
        y: shortLbls,
        customdata: labels.map((l, i) => `${l}<br>완료: ${done[i]}매 / 계획: ${planQtys[i]}매`),
        hovertemplate: "%{customdata}<extra></extra>",
        marker: { color: "#55A868" },
      } as Data,
      {
        type: "bar",
        orientation: "h",
        name: "잔여 재공",
        x: wipLeft,
        y: shortLbls,
        customdata: labels.map((l, i) => `${l}<br>잔여 재공: ${wipLeft[i]}매`),
        hovertemplate: "%{customdata}<extra></extra>",
        marker: { color: "#DD8452" },
      } as Data,
      {
        type: "bar",
        orientation: "h",
        name: "계획 미달",
        x: planGap,
        y: shortLbls,
        customdata: labels.map((l, i) => `${l}<br>계획 미달: ${planGap[i]}매`),
        hovertemplate: "%{customdata}<extra></extra>",
        marker: { color: "#C44E52", opacity: 0.65 },
      } as Data,
      {
        type: "scatter",
        mode: "markers",
        name: "계획 목표",
        x: planQtys,
        y: shortLbls,
        customdata: labels.map((l, i) => `${l}<br>계획: ${planQtys[i]}매`),
        hovertemplate: "%{customdata}<extra></extra>",
        marker: {
          symbol: "line-ns",
          size: 18,
          color: "#4C72B0",
          line: { width: 2.5, color: "#4C72B0" },
        },
      } as Data,
    ],
    layout: {
      title: { text: "재공 처리 현황 (완료 / 잔여 재공 / 계획 미달)", font: { size: 13 } },
      barmode: "stack",
      xaxis: { title: { text: "수량 (매)" }, automargin: true },
      height: Math.max(300, 40 * Math.max(plan.length, 4)),
      margin: { l: 80, r: 24, t: 56, b: 64 },
      ...SHARED_DARK,
      legend: { orientation: "h", y: -0.22, x: 0.5, xanchor: "center" },
    },
  };
}
