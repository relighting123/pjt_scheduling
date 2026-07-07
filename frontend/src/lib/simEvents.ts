import type { SimEvent, SimEventKind } from "../types";

/** Python simulation/events.py 와 동일한 이벤트 코드 */
export const SIM_EVENT = {
  MOVE_OUT: "MOVE_OUT",
  IDLE: "IDLE",
  JOB_ASSIGNED: "JOB_ASSIGNED",
  CONV_ASSIGNED: "CONV_ASSIGNED",
} as const satisfies Record<string, SimEventKind>;

export const SIM_EVENT_LABEL: Record<SimEventKind, string> = {
  MOVE_OUT: "MOVE_OUT",
  IDLE: "IDLE",
  JOB_ASSIGNED: "JOB_ASSIGNED",
  CONV_ASSIGNED: "CONV_ASSIGNED",
  TOOL_RELEASE: "TOOL_RELEASE",
  WIP_INJECT: "WIP_INJECT",
  TOOL_OCCUPY: "TOOL_OCCUPY",
  PROCESS_END: "PROCESS_END",
  IDLE_DECISION: "IDLE_DECISION",
  CONV_START: "CONV_START",
  CONV_END: "CONV_END",
};

export const SIM_EVENT_LABEL_KO: Record<SimEventKind, string> = {
  MOVE_OUT: "작업 완료 · 배출",
  IDLE: "장비 부하 발생",
  JOB_ASSIGNED: "작업 시작",
  CONV_ASSIGNED: "변환 배정",
  TOOL_RELEASE: "툴 반환",
  WIP_INJECT: "재공 유입",
  TOOL_OCCUPY: "툴 점유",
  PROCESS_END: "공정 완료",
  IDLE_DECISION: "유휴 · 배정 결정",
  CONV_START: "변환 시작",
  CONV_END: "변환 완료",
};

export const SIM_EVENT_CLASS: Record<SimEventKind, string> = {
  MOVE_OUT: "evt-move",
  IDLE: "evt-decision",
  JOB_ASSIGNED: "evt-assign",
  CONV_ASSIGNED: "evt-conv",
  TOOL_RELEASE: "evt-tool",
  WIP_INJECT: "evt-wip",
  TOOL_OCCUPY: "evt-tool",
  PROCESS_END: "evt-process",
  IDLE_DECISION: "evt-decision",
  CONV_START: "evt-conv",
  CONV_END: "evt-conv",
};

/** 이전 event_log (snake_case / 구 코드) → 대문자 코드 */
const LEGACY_EVENT_KIND: Record<string, SimEventKind> = {
  process_end: "PROCESS_END",
  move_out: "MOVE_OUT",
  tool_release: "MOVE_OUT",
  TOOL_RELEASE: "MOVE_OUT",
  wip_inject: "MOVE_OUT",
  WIP_INJECT: "MOVE_OUT",
  idle: "IDLE",
  job_start: "IDLE",
  JOB_START: "IDLE",
  IDLE_DECISION: "IDLE",
  idle_decision: "IDLE",
  conv_assigned: "CONV_ASSIGNED",
  conv_start: "CONV_ASSIGNED",
  CONV_START: "CONV_ASSIGNED",
  conv_end: "CONV_END",
  tool_occupy: "JOB_ASSIGNED",
  TOOL_OCCUPY: "JOB_ASSIGNED",
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
  if (ev.eqp_status) parts.push(ev.eqp_status);
  if (ev.lot_cd) parts.push(`LOT_CD=${ev.lot_cd}`);
  if (ev.next_oper_id) {
    parts.push(
      ev.next_plan_prod_key
        ? `→ ${ev.next_plan_prod_key}/${ev.next_oper_id}`
        : `→ ${ev.next_oper_id}`,
    );
  }
  if (ev.from_lot_cd && ev.to_lot_cd) {
    parts.push(`${ev.from_lot_cd} → ${ev.to_lot_cd}`);
  } else if (ev.from_lot_cd) {
    parts.push(`from ${ev.from_lot_cd}`);
  }
  if (ev.conv_duration_min != null) parts.push(`${ev.conv_duration_min}분`);
  if (ev.tool_from_delta != null && ev.tool_to_delta != null) {
    parts.push(`tool +${ev.tool_from_delta}/${ev.tool_to_delta}`);
  } else if (ev.tool_to_delta != null) {
    parts.push(`tool ${ev.tool_to_delta}`);
  }
  if (ev.eqp_model) parts.push(`model ${ev.eqp_model}`);
  return parts.join(" · ");
}

/** Dedicated columns(EQP/LOT/PPK/OPER)에 없는 보조 정보만 */
export function formatSimEventExtra(ev: SimEvent): string {
  return formatSimEventDetail(ev) || "—";
}
