import type { ConversionPlan, InferenceResult, PlanRecord, ScheduleRecord, SimEvent } from "../types";

export interface EqpUtil {
  eqp_id: string;
  model?: string;
  busyMin: number;
  totalMin: number;
  utilPct: number;
}

export interface ModelUtil {
  model: string;
  eqpCount: number;
  avgUtilPct: number;
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
  doneQty: number;
  pct: number;
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
): EqpUtil[] {
  const busyMap: Record<string, number> = {};
  schedule.forEach((r) => {
    busyMap[r.EQP_ID] = (busyMap[r.EQP_ID] ?? 0) + (r.END_TM - r.START_TM);
  });
  return eqpIds.map((id) => {
    const busy = busyMap[id] ?? 0;
    return {
      eqp_id: id,
      model: modelMap[id],
      busyMin: busy,
      totalMin: simEndMin,
      utilPct: simEndMin > 0 ? Math.round((busy / simEndMin) * 1000) / 10 : 0,
    };
  }).sort((a, b) => b.utilPct - a.utilPct);
}

export function computeModelUtil(utils: EqpUtil[]): ModelUtil[] {
  const grouped: Record<string, EqpUtil[]> = {};
  utils.forEach((u) => {
    const key = u.model ?? "Unknown";
    (grouped[key] ??= []).push(u);
  });
  return Object.entries(grouped).map(([model, group]) => ({
    model,
    eqpCount: group.length,
    avgUtilPct: Math.round((group.reduce((s, u) => s + u.utilPct, 0) / group.length) * 10) / 10,
  })).sort((a, b) => b.avgUtilPct - a.avgUtilPct);
}

export function computeTAT(schedule: ScheduleRecord[]): TatRow[] {
  const lotTimes: Record<string, { start: number; end: number; prod: string }> = {};
  schedule.forEach((r) => {
    const prev = lotTimes[r.LOT_ID];
    if (!prev) {
      lotTimes[r.LOT_ID] = { start: r.START_TM, end: r.END_TM, prod: r.PLAN_PROD_KEY };
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

export function computeAchievement(
  schedule: ScheduleRecord[],
  plan: PlanRecord[],
): AchievementRow[] {
  const doneMap: Record<string, number> = {};
  schedule.forEach((r) => {
    const key = `${r.PLAN_PROD_KEY}|${r.OPER_ID ?? ""}`;
    doneMap[key] = (doneMap[key] ?? 0) + (r.WF_QTY ?? 25);
  });

  return plan.map((p) => {
    const key = `${p.plan_prod_key}|${p.oper_id}`;
    const done = doneMap[key] ?? 0;
    const pct = Math.round((done / Math.max(p.d0_plan_qty, 1)) * 1000) / 10;
    return {
      key,
      prod: p.plan_prod_key,
      oper: p.oper_id,
      planQty: p.d0_plan_qty,
      doneQty: done,
      pct,
    };
  }).sort((a, b) => a.prod.localeCompare(b.prod) || a.oper.localeCompare(b.oper));
}

export function countToolSwitches(
  schedule: ScheduleRecord[],
  convPlans: ConversionPlan[],
): number {
  if (convPlans.length > 0) return convPlans.length;
  // Fallback: count ST changes per EQP
  const eqpSeq: Record<string, ScheduleRecord[]> = {};
  schedule.forEach((r) => {
    (eqpSeq[r.EQP_ID] ??= []).push(r);
  });
  let count = 0;
  Object.values(eqpSeq).forEach((recs) => {
    recs.sort((a, b) => a.START_TM - b.START_TM);
    for (let i = 1; i < recs.length; i++) {
      if (recs[i].ST && recs[i - 1].ST && recs[i].ST !== recs[i - 1].ST) count++;
    }
  });
  return count;
}

export function computeAvgUtil(utils: EqpUtil[]): number {
  if (!utils.length) return 0;
  return Math.round((utils.reduce((s, u) => s + u.utilPct, 0) / utils.length) * 10) / 10;
}

export function computeAvgAchievement(rows: AchievementRow[]): number {
  if (!rows.length) return 0;
  return Math.round((rows.reduce((s, r) => s + Math.min(r.pct, 100), 0) / rows.length) * 10) / 10;
}

export interface InferenceKpi {
  makespan: number;
  avgUtilPct: number;
  operSwitches: number;
  prodSwitches: number;
  toolSwitches: number;
  avgAchPct: number;
}

export function computeInferenceKpi(
  result: InferenceResult,
  modelMap: Record<string, string> = {},
): InferenceKpi {
  const sched = result.schedule;
  const makespan = sched.length ? Math.max(...sched.map((r) => r.END_TM)) : 0;
  const utils = computeEqpUtil(sched, result.eqp_ids, result.sim_end_minutes, modelMap);
  const ach = computeAchievement(sched, result.plan);
  const toolSw = countToolSwitches(sched, result.conversion_plans ?? []);
  return {
    makespan,
    avgUtilPct: computeAvgUtil(utils),
    operSwitches: result.stats.oper_switches,
    prodSwitches: result.stats.prod_switches,
    toolSwitches: toolSw,
    avgAchPct: computeAvgAchievement(ach),
  };
}
