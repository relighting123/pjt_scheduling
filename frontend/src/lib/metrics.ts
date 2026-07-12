import type { ConversionPlan, DowntimePlan, InferenceResult, PlanRecord, ScheduleRecord, SimEvent } from "../types";

/** 무제한 다운(down_end_min 없음)의 통계용 종료 시각(분). */
function downEffectiveEnd(p: DowntimePlan, simEndMin: number): number {
  return p.down_end_min ?? Math.max(simEndMin, p.down_start_min);
}

export interface EqpUtil {
  eqp_id: string;
  model?: string;
  busyMin: number;
  makespanMin: number;
  utilPct: number;
}

export interface TatRow {
  prod: string;
  count: number;
  avgMin: number;
  minMin: number;
  maxMin: number;
}

export interface AchievementRow {
  key: string;
  prod: string;
  oper: string;
  planQty: number;
  targetQty: number;
  doneQty: number;
  pct: number;
  targetPct: number;
}

export interface EqpScheduleSummary {
  eqp_id: string;
  model?: string;
  firstStart: number | null;
  lastEnd: number | null;
  makespanMin: number;
  jobCount: number;
  busyMin: number;
  convMin: number;
  downMin: number;
  idleMin: number;
  utilPct: number;
  idlePct: number;
  outputQty: number;
  operSwitches: number;
  prodSwitches: number;
}

export function buildEqpModelMap(events: SimEvent[]): Record<string, string> {
  const map: Record<string, string> = {};
  events.forEach((e) => {
    if (e.eqp_id && e.eqp_model) map[e.eqp_id] = e.eqp_model;
  });
  return map;
}

export function computeEqpUtil(
  schedule: ScheduleRecord[],
  eqpIds: string[],
  simEndMin: number,
  modelMap: Record<string, string> = {},
  conversionPlans: ConversionPlan[] = [],
  downtimePlans: DowntimePlan[] = [],
): EqpUtil[] {
  const byEqp: Record<string, ScheduleRecord[]> = {};
  eqpIds.forEach((id) => { byEqp[id] = []; });
  schedule.forEach((r) => { (byEqp[r.EQP_ID] ??= []).push(r); });

  return eqpIds.map((id) => {
    const recs = byEqp[id];
    const convs = conversionPlans.filter((p) => p.eqp_id === id);
    const downs = downtimePlans.filter((p) => p.eqp_id === id);
    const busy = recs.reduce((s, r) => s + (r.END_TM - r.START_TM), 0);

    const firstStart = recs.length > 0 ? Math.min(...recs.map((r) => r.START_TM)) : null;
    const allEnds = [
      ...recs.map((r) => r.END_TM),
      ...convs.map((p) => p.conv_end_min),
      ...downs.map((p) => downEffectiveEnd(p, simEndMin)),
    ];
    const lastEnd = allEnds.length > 0 ? Math.max(...allEnds) : null;
    const makespanMin = firstStart !== null && lastEnd !== null ? lastEnd - firstStart : 0;

    return {
      eqp_id: id,
      model: modelMap[id],
      busyMin: busy,
      makespanMin,
      utilPct: makespanMin > 0 ? Math.round((busy / makespanMin) * 1000) / 10 : 0,
    };
  }).sort((a, b) => b.utilPct - a.utilPct);
}

export function computeEqpScheduleSummary(
  schedule: ScheduleRecord[],
  eqpIds: string[],
  simEndMin: number,
  modelMap: Record<string, string> = {},
  conversionPlans: ConversionPlan[] = [],
  downtimePlans: DowntimePlan[] = [],
): EqpScheduleSummary[] {
  return eqpIds.map((eqp_id) => {
    const recs = schedule
      .filter((r) => r.EQP_ID === eqp_id)
      .sort((a, b) => a.START_TM - b.START_TM);
    const convs = conversionPlans.filter((p) => p.eqp_id === eqp_id);
    const downs = downtimePlans.filter((p) => p.eqp_id === eqp_id);

    const busyMin = recs.reduce((s, r) => s + (r.END_TM - r.START_TM), 0);
    const convMin = convs.reduce((s, p) => s + (p.conv_end_min - p.conv_start_min), 0);
    const downMin = downs.reduce((s, p) => s + (downEffectiveEnd(p, simEndMin) - p.down_start_min), 0);
    const outputQty = recs.reduce((s, r) => s + (r.WF_QTY ?? 25), 0);

    let operSwitches = 0;
    let prodSwitches = 0;
    for (let i = 1; i < recs.length; i++) {
      if ((recs[i].OPER_ID ?? "") !== (recs[i - 1].OPER_ID ?? "")) operSwitches++;
      if (recs[i].PLAN_PROD_ATTR_VAL !== recs[i - 1].PLAN_PROD_ATTR_VAL) prodSwitches++;
    }

    const convEnds = convs.map((p) => p.conv_end_min);
    const downEnds = downs.map((p) => downEffectiveEnd(p, simEndMin));
    const schedEnds = recs.map((r) => r.END_TM);
    const lastEnd = schedEnds.length || convEnds.length || downEnds.length
      ? Math.max(0, ...schedEnds, ...convEnds, ...downEnds)
      : null;
    const firstStart = recs.length ? recs[0].START_TM : null;

    const makespanMin = firstStart !== null && lastEnd !== null ? lastEnd - firstStart : 0;
    // 유휴 = makespan 내에서 가동·전환·다운 어디에도 속하지 않는 구간 (재공 빈 시간)
    const idleMin = Math.max(0, makespanMin - busyMin - convMin - downMin);
    // 가동률 = busy / makespan (전환·다운·빈 시간은 비가동으로 처리)
    const utilPct = makespanMin > 0 ? Math.round((busyMin / makespanMin) * 1000) / 10 : 0;
    // 유휴율 = (전환 + 다운 + 빈 시간) / makespan
    const idlePct = makespanMin > 0 ? Math.round(((makespanMin - busyMin) / makespanMin) * 1000) / 10 : 0;

    return {
      eqp_id,
      model: modelMap[eqp_id],
      firstStart,
      lastEnd,
      makespanMin,
      jobCount: recs.length,
      busyMin,
      convMin,
      downMin,
      idleMin,
      utilPct,
      idlePct,
      outputQty,
      operSwitches,
      prodSwitches,
    };
  }).sort((a, b) => a.eqp_id.localeCompare(b.eqp_id));
}

export function computeTAT(schedule: ScheduleRecord[]): TatRow[] {
  const lotTimes: Record<string, { start: number; end: number; prod: string }> = {};
  schedule.forEach((r) => {
    const prev = lotTimes[r.LOT_ID];
    if (!prev) {
      lotTimes[r.LOT_ID] = { start: r.START_TM, end: r.END_TM, prod: r.PLAN_PROD_ATTR_VAL };
    } else {
      lotTimes[r.LOT_ID].start = Math.min(prev.start, r.START_TM);
      lotTimes[r.LOT_ID].end = Math.max(prev.end, r.END_TM);
    }
  });

  const prodTats: Record<string, number[]> = {};
  Object.values(lotTimes).forEach(({ start, end, prod }) => {
    (prodTats[prod] ??= []).push(end - start);
  });

  return Object.entries(prodTats).map(([prod, tats]) => ({
    prod,
    count: tats.length,
    avgMin: Math.round(tats.reduce((s, t) => s + t, 0) / tats.length),
    minMin: Math.min(...tats),
    maxMin: Math.max(...tats),
  })).sort((a, b) => a.prod.localeCompare(b.prod));
}

function achievementKeyPart(value: string | undefined): string {
  return String(value ?? "").trim();
}

function achievementKey(prod: string, oper: string | undefined): string {
  return `${achievementKeyPart(prod)}|${achievementKeyPart(oper)}`;
}

function sortAchievementRows(
  rows: AchievementRow[],
  prodKeys: string[],
  operIds: string[],
): AchievementRow[] {
  const prodIdx = Object.fromEntries(prodKeys.map((k, i) => [k, i]));
  const operIdx = Object.fromEntries(operIds.map((k, i) => [k, i]));
  return [...rows].sort((a, b) => {
    const pa = prodIdx[a.prod] ?? prodKeys.length;
    const pb = prodIdx[b.prod] ?? prodKeys.length;
    if (pa !== pb) return pa - pb;
    const oa = operIdx[a.oper] ?? operIds.length;
    const ob = operIdx[b.oper] ?? operIds.length;
    if (oa !== ob) return oa - ob;
    return a.prod.localeCompare(b.prod) || a.oper.localeCompare(b.oper);
  });
}

export function computeAchievement(
  schedule: ScheduleRecord[],
  plan: PlanRecord[],
  order?: { prodKeys?: string[]; operIds?: string[] },
): AchievementRow[] {
  const doneMap: Record<string, number> = {};
  schedule.forEach((r) => {
    const key = achievementKey(r.PLAN_PROD_ATTR_VAL, r.OPER_ID);
    doneMap[key] = (doneMap[key] ?? 0) + (r.WF_QTY ?? 25);
  });

  // Accumulate plan qty (D0) and target qty (D1: 재공 고려 타겟)
  const qtyMap = new Map<string, { prod: string; oper: string; planQty: number; targetQty: number }>();
  for (const p of plan) {
    const prod = achievementKeyPart(p.PLAN_PROD_ATTR_VAL);
    const oper = achievementKeyPart(p.oper_id);
    const key = achievementKey(prod, oper);
    const prev = qtyMap.get(key);
    qtyMap.set(key, {
      prod,
      oper,
      planQty: (prev?.planQty ?? 0) + p.d0_plan_qty,
      targetQty: (prev?.targetQty ?? 0) + (p.d1_plan_qty ?? p.d0_plan_qty),
    });
  }

  const rows: AchievementRow[] = [];
  for (const [key, { prod, oper, planQty, targetQty }] of qtyMap) {
    const done = doneMap[key] ?? 0;
    rows.push({
      key,
      prod,
      oper,
      planQty,
      targetQty,
      doneQty: done,
      pct: Math.round((done / Math.max(planQty, 1)) * 1000) / 10,
      targetPct: Math.round((done / Math.max(targetQty, 1)) * 1000) / 10,
    });
  }

  const prodKeys = order?.prodKeys ?? [];
  const operIds = order?.operIds ?? [];
  if (prodKeys.length || operIds.length) {
    return sortAchievementRows(rows, prodKeys, operIds);
  }
  return rows.sort((a, b) => a.prod.localeCompare(b.prod) || a.oper.localeCompare(b.oper));
}

export function countToolSwitches(
  schedule: ScheduleRecord[],
  convPlans: ConversionPlan[],
): number {
  if (convPlans.length > 0) return convPlans.length;
  // Fallback 1: 행마다 백엔드가 이미 판정한 CONVERSION 플래그가 있으면 그대로 집계
  // (_would_need_conversion 결과 그대로 — LOT_CD 또는 TEMP 중 하나라도 바뀌면 true)
  if (schedule.some((r) => r.CONVERSION !== undefined)) {
    return schedule.filter((r) => r.CONVERSION).length;
  }
  // Fallback 2: EQP_ID 기준으로 직전 LOT_CD 또는 TEMP 중 하나라도 바뀌면 tool 교체로 집계
  const eqpSeq: Record<string, ScheduleRecord[]> = {};
  schedule.forEach((r) => {
    (eqpSeq[r.EQP_ID] ??= []).push(r);
  });
  let count = 0;
  Object.values(eqpSeq).forEach((recs) => {
    recs.sort((a, b) => a.START_TM - b.START_TM);
    for (let i = 1; i < recs.length; i++) {
      const prev = recs[i - 1];
      const cur = recs[i];
      if (cur.LOT_CD == null && cur.TEMP == null) continue;
      if (cur.LOT_CD !== prev.LOT_CD || (cur.TEMP ?? "") !== (prev.TEMP ?? "")) count++;
    }
  });
  return count;
}

export function computeAvgUtil(utils: EqpUtil[]): number {
  const active = utils.filter((u) => u.makespanMin > 0);
  if (!active.length) return 0;
  return Math.round((active.reduce((s, u) => s + u.utilPct, 0) / active.length) * 10) / 10;
}

export function computeAvgIdle(rows: EqpScheduleSummary[]): number {
  if (!rows.length) return 0;
  return Math.round((rows.reduce((s, r) => s + r.idlePct, 0) / rows.length) * 10) / 10;
}

export function computeAvgAchievement(rows: AchievementRow[]): number {
  if (!rows.length) return 0;
  return Math.round((rows.reduce((s, r) => s + Math.min(r.pct, 100), 0) / rows.length) * 10) / 10;
}

export function computeAvgTargetAchievement(rows: AchievementRow[]): number {
  if (!rows.length) return 0;
  return Math.round((rows.reduce((s, r) => s + Math.min(r.targetPct, 100), 0) / rows.length) * 10) / 10;
}

export interface InferenceKpi {
  makespan: number;
  avgUtilPct: number;
  avgIdlePct: number;
  operSwitches: number;
  prodSwitches: number;
  toolSwitches: number;
  /** 계획 달성률: 실제 계획(D0) 기준 */
  avgAchPct: number;
  /** 타겟 달성률: 재공 고려 타겟(D1) 기준 */
  avgTargetAchPct: number;
}

export function computeInferenceKpi(
  result: InferenceResult,
  modelMap: Record<string, string> = {},
): InferenceKpi {
  const sched = result.schedule;
  const convPlans = result.conversion_plans ?? [];
  const makespan = sched.length ? Math.max(...sched.map((r) => r.END_TM)) : 0;
  const utils = computeEqpUtil(sched, result.eqp_ids, result.sim_end_minutes, modelMap, convPlans);
  const eqpSummary = computeEqpScheduleSummary(
    sched,
    result.eqp_ids,
    result.sim_end_minutes,
    modelMap,
    convPlans,
  );
  const ach = computeAchievement(sched, result.plan, {
    prodKeys: result.prod_keys,
    operIds: result.oper_ids,
  });
  const toolSw = countToolSwitches(sched, convPlans);
  return {
    makespan,
    avgUtilPct: computeAvgUtil(utils),
    avgIdlePct: computeAvgIdle(eqpSummary),
    operSwitches: result.stats.oper_switches,
    prodSwitches: result.stats.prod_switches,
    toolSwitches: toolSw,
    avgAchPct: computeAvgAchievement(ach),
    avgTargetAchPct: computeAvgTargetAchievement(ach),
  };
}
