import type { SimEvent, SimEventKind } from "../types";

/** Python simulation/events.py 와 동일한 이벤트 코드 */
export const SIM_EVENT = {
  PROCESS_END: "PROCESS_END",
  MOVE_OUT: "MOVE_OUT",
  TOOL_RELEASE: "TOOL_RELEASE",
  WIP_INJECT: "WIP_INJECT",
  JOB_START: "JOB_START",
  CONV_START: "CONV_START",
  CONV_END: "CONV_END",
  TOOL_OCCUPY: "TOOL_OCCUPY",
  JOB_ASSIGNED: "JOB_ASSIGNED",
} as const satisfies Record<string, SimEventKind>;

export const SIM_EVENT_KINDS = Object.values(SIM_EVENT);

export const SIM_EVENT_LABEL: Record<SimEventKind, string> = {
  PROCESS_END: "PROCESS_END",
  MOVE_OUT: "MOVE_OUT",
  TOOL_RELEASE: "TOOL_RELEASE",
  WIP_INJECT: "WIP_INJECT",
  JOB_START: "JOB_START",
  CONV_START: "CONV_START",
  CONV_END: "CONV_END",
  TOOL_OCCUPY: "TOOL_OCCUPY",
  JOB_ASSIGNED: "JOB_ASSIGNED",
};

export const SIM_EVENT_LABEL_KO: Record<SimEventKind, string> = {
  PROCESS_END: "공정 완료",
  MOVE_OUT: "Move Out",
  TOOL_RELEASE: "Tool 반환",
  WIP_INJECT: "WIP 유입",
  JOB_START: "Job Start",
  CONV_START: "Conv Start",
  CONV_END: "Conv End",
  TOOL_OCCUPY: "Tool 점유",
  JOB_ASSIGNED: "Job 배정",
};

export const SIM_EVENT_CLASS: Record<SimEventKind, string> = {
  PROCESS_END: "evt-process",
  MOVE_OUT: "evt-move",
  TOOL_RELEASE: "evt-tool",
  WIP_INJECT: "evt-wip",
  JOB_START: "evt-decision",
  CONV_START: "evt-conv",
  CONV_END: "evt-conv",
  TOOL_OCCUPY: "evt-tool",
  JOB_ASSIGNED: "evt-assign",
};

/** 이전 event_log (snake_case) → 대문자 코드 */
const LEGACY_EVENT_KIND: Record<string, SimEventKind> = {
  process_end: "PROCESS_END",
  move_out: "MOVE_OUT",
  tool_release: "TOOL_RELEASE",
  wip_inject: "WIP_INJECT",
  job_start: "JOB_START",
  conv_start: "CONV_START",
  conv_end: "CONV_END",
  tool_occupy: "TOOL_OCCUPY",
  job_assigned: "JOB_ASSIGNED",
};

export function normalizeSimEventKind(kind: string): SimEventKind | string {
  if (kind in SIM_EVENT_LABEL) return kind as SimEventKind;
  return LEGACY_EVENT_KIND[kind] ?? kind;
}

export function simEventLabel(kind: string, ko = false): string {
  const normalized = normalizeSimEventKind(kind);
  const map = ko ? SIM_EVENT_LABEL_KO : SIM_EVENT_LABEL;
  return map[normalized as SimEventKind] ?? String(normalized);
}

export function formatSimEventDetail(ev: SimEvent): string {
  const parts: string[] = [];
  if (ev.eqp_id) parts.push(ev.eqp_id);
  if (ev.lot_id) parts.push(ev.lot_id);
  if (ev.lot_cd) parts.push(`LOT_CD=${ev.lot_cd}`);
  if (ev.plan_prod_key) parts.push(ev.plan_prod_key);
  if (ev.oper_id) parts.push(ev.oper_id);
  if (ev.from_lot_cd && ev.to_lot_cd) {
    parts.push(`${ev.from_lot_cd} → ${ev.to_lot_cd}`);
  } else if (ev.from_lot_cd) {
    parts.push(`from ${ev.from_lot_cd}`);
  }
  if (ev.eqp_model) parts.push(`model ${ev.eqp_model}`);
  return parts.join(" · ");
}
