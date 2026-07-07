// output.json(RTS 적재 JSON) / result_full.json 을 백엔드 없이 UI 간트용
// InferenceResult 로 변환한다. 백엔드 api/server.py 의 _result_from_rts_output 과
// 동일한 복원 로직을 클라이언트에서 수행한다.
import type {
  ConversionPlan,
  InferenceResult,
  PlanRecord,
  ScheduleRecord,
} from "../types";

/** YYYYMMDDHHMMSS(14) / YYYYMMDDHHMM(12) → Date (실패 시 null). */
function parseTimekey(value: string): Date | null {
  const s = String(value ?? "");
  if (s.length !== 14 && s.length !== 12) return null;
  const year = Number(s.slice(0, 4));
  const month = Number(s.slice(4, 6));
  const day = Number(s.slice(6, 8));
  const hour = Number(s.slice(8, 10));
  const minute = Number(s.slice(10, 12));
  const second = s.length === 14 ? Number(s.slice(12, 14)) : 0;
  if ([year, month, day, hour, minute, second].some((n) => Number.isNaN(n))) {
    return null;
  }
  // 차이 계산만 하므로 UTC 기준으로 만들어 DST 영향을 제거한다.
  return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

/** timekey → base 로부터의 경과 분 (백엔드 _minutes_from_timekey 대응). */
function minutesFromTimekey(value: string, base: Date): number {
  const d = parseTimekey(value);
  if (!d) return 0;
  return Math.floor((d.getTime() - base.getTime()) / 60000);
}

function sortedUnique(values: string[]): string[] {
  return [...new Set(values.filter((v) => v !== "" && v != null))].sort();
}

/** RTS output.json(RTS_RSLT_INF 포함) → InferenceResult. */
function resultFromRtsOutput(payload: Record<string, unknown>): InferenceResult {
  const rsltRows = (payload.RTS_RSLT_INF as Record<string, unknown>[]) ?? [];
  const convRows = (payload.RTS_EQPCONVPLAN_INF as Record<string, unknown>[]) ?? [];

  // env_data 가 없으므로 가장 이른 START_TIME 을 base(=0분) 로 사용한다.
  let base: Date | null = null;
  for (const row of rsltRows) {
    const d = parseTimekey(String(row.START_TIME ?? ""));
    if (d && (base === null || d.getTime() < base.getTime())) base = d;
  }
  if (!base) base = new Date(0);

  const schedule: ScheduleRecord[] = rsltRows.map((row) => {
    const startTm = minutesFromTimekey(String(row.START_TIME ?? ""), base as Date);
    const endTm = minutesFromTimekey(String(row.END_TIME ?? ""), base as Date);
    return {
      EQP_ID: String(row.EQP_ID ?? ""),
      LOT_ID: String(row.LOT_ID ?? ""),
      CARRIER_ID: String(row.CARRIER_ID ?? ""),
      PLAN_PROD_ATTR_VAL: String(row.PLAN_PROD_ATTR_VAL ?? ""),
      OPER_ID: String(row.OPER_ID ?? ""),
      EQP_MODEL: String(row.EQP_MODEL_CD ?? ""),
      SEQ: Number(row.SEQ_NO ?? 0) || 0,
      START_TM: startTm,
      END_TM: endTm,
      PROC_TIME: Math.max(endTm - startTm, 0),
      WF_QTY: Number(row.PRODUCE_QTY ?? 0) || 0,
      LOT_CD: String(row.LOT_CD ?? ""),
      TEMP: String(row.TEMPER_VAL ?? ""),
      START_TM_STR: String(row.START_TIME ?? ""),
      END_TM_STR: String(row.END_TIME ?? ""),
    } as ScheduleRecord;
  });
  schedule.sort(
    (a, b) =>
      a.START_TM - b.START_TM ||
      a.EQP_ID.localeCompare(b.EQP_ID) ||
      (a.SEQ ?? 0) - (b.SEQ ?? 0) ||
      a.LOT_ID.localeCompare(b.LOT_ID),
  );

  const conversionPlans: ConversionPlan[] = convRows.map((row) => {
    const convStart = minutesFromTimekey(String(row.CONV_START_TM ?? ""), base as Date);
    const convEnd = minutesFromTimekey(String(row.CONV_END_TM ?? ""), base as Date);
    return {
      eqp_id: String(row.EQP_ID ?? ""),
      eqp_model_cd: String(row.EQP_MODEL_CD ?? ""),
      oper_id: String(row.OPER_ID ?? ""),
      PLAN_PROD_ATTR_VAL: String(row.PLAN_PROD_ATTR_VAL ?? ""),
      from_lot_cd: String(row.LOT_CD ?? ""),
      from_temp: String(row.TEMPER_VAL ?? ""),
      to_lot_cd: String(row.TO_LOT_CD ?? ""),
      to_temp: String(row.TO_TEMPER_VAL ?? ""),
      conv_start_min: convStart,
      conv_end_min: convEnd,
      conv_time: Number(row.CONV_TIME ?? 0) || Math.max(convEnd - convStart, 0),
    } as unknown as ConversionPlan;
  });

  const completed: Record<string, number> = {};
  for (const rec of schedule) {
    const key = `${rec.PLAN_PROD_ATTR_VAL}|${rec.OPER_ID ?? ""}`;
    completed[key] = (completed[key] ?? 0) + (rec.WF_QTY ?? 0);
  }

  const meta = (payload.meta as Record<string, unknown>) ?? {};
  const simEnd = schedule.reduce((m, r) => Math.max(m, r.END_TM), 0);

  return {
    schedule,
    history: [],
    event_log: [],
    decision_log: [],
    conversion_plans: conversionPlans,
    stats: {
      idle_total: 0,
      oper_switches: 0,
      prod_switches: 0,
      completed_qty: completed,
      source_file: "output.json",
    },
    plan: [],
    prod_keys: sortedUnique(schedule.map((r) => r.PLAN_PROD_ATTR_VAL)),
    oper_ids: sortedUnique(schedule.map((r) => r.OPER_ID ?? "")),
    eqp_ids: sortedUnique(schedule.map((r) => r.EQP_ID)),
    sim_end_minutes: simEnd,
    algorithm: String(meta.ALGORITHM ?? "saved"),
  };
}

/** result_full.json(이미 직렬화된 결과) → InferenceResult (누락 필드 보강). */
function resultFromFull(payload: Record<string, unknown>): InferenceResult {
  const schedule = (payload.schedule as ScheduleRecord[]) ?? [];
  const stats = (payload.stats as Record<string, unknown>) ?? {};
  const plan = (payload.plan as PlanRecord[]) ?? [];

  const prodKeys =
    (payload.prod_keys as string[])?.length
      ? (payload.prod_keys as string[])
      : sortedUnique(schedule.map((r) => r.PLAN_PROD_ATTR_VAL));
  const operIds =
    (payload.oper_ids as string[])?.length
      ? (payload.oper_ids as string[])
      : sortedUnique(schedule.map((r) => r.OPER_ID ?? ""));
  const eqpIds =
    (payload.eqp_ids as string[])?.length
      ? (payload.eqp_ids as string[])
      : sortedUnique(schedule.map((r) => r.EQP_ID));
  const simEnd =
    Number(payload.sim_end_minutes ?? stats.sim_end_minutes ?? 0) ||
    schedule.reduce((m, r) => Math.max(m, r.END_TM), 0);

  return {
    schedule,
    history: (payload.history as InferenceResult["history"]) ?? [],
    event_log: (payload.event_log as InferenceResult["event_log"]) ?? [],
    decision_log: (payload.decision_log as InferenceResult["decision_log"]) ?? [],
    conversion_plans:
      (payload.conversion_plans as InferenceResult["conversion_plans"]) ?? [],
    stats: {
      idle_total: Number(stats.idle_total ?? 0),
      oper_switches: Number(stats.oper_switches ?? 0),
      prod_switches: Number(stats.prod_switches ?? 0),
      ...stats,
      completed_qty: (stats.completed_qty as Record<string, number>) ?? {},
      source_file: String(stats.source_file ?? "result_full.json"),
    },
    plan,
    prod_keys: prodKeys,
    oper_ids: operIds,
    eqp_ids: eqpIds,
    sim_end_minutes: simEnd,
    algorithm: (payload.algorithm as string) ?? "saved",
  };
}

/**
 * 업로드한 JSON 을 InferenceResult 로 변환한다.
 * - RTS output.json (RTS_RSLT_INF 포함)
 * - result_full.json (schedule 최상위 키 포함)
 */
export function parseResultFile(json: unknown): InferenceResult {
  if (!json || typeof json !== "object") {
    throw new Error("JSON 형식이 올바르지 않습니다.");
  }
  const payload = json as Record<string, unknown>;
  if (Array.isArray(payload.RTS_RSLT_INF)) {
    return resultFromRtsOutput(payload);
  }
  if (Array.isArray(payload.schedule)) {
    return resultFromFull(payload);
  }
  throw new Error(
    "인식할 수 없는 결과 파일입니다. output.json(RTS_RSLT_INF) 또는 result_full.json(schedule) 을 선택하세요.",
  );
}

/** File 객체를 읽어 InferenceResult 로 파싱한다. */
export async function loadResultFromFile(file: File): Promise<InferenceResult> {
  const text = await file.text();
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("JSON 파싱에 실패했습니다. 올바른 결과 파일인지 확인하세요.");
  }
  return parseResultFile(json);
}
