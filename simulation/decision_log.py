"""추론 step별 EQP/PPK/OPER 결정 및 미할당 사유 진단."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from simulation.simulator import SchedulingSimulator

REASON_LABELS: Dict[str, str] = {
    "assigned": "배정 완료",
    "action_corrected": "요청 action 보정 후 배정",
    "assign_failed": "배정 시도 실패",
    "no_feasible": "idle이나 feasible (PPK,OPER) 없음",
    "no_idle_eqp": "결정 대기 EQP 없음 (시간 전진)",
    "eqp_not_idle": "EQP가 idle 상태가 아님",
    "sim_done": "시뮬레이션 종료",
    "no_wip": "재공(WIP) 없음",
    "wip_not_ready": "재공 oper_in_time 미도래",
    "no_arrange": "EQP arrange/가공 불가",
    "tool_cap_blocked": "tool cap 차단",
    "lot_select_failed": "LOT 자동 선택 실패",
}


def _lot_cd_temp(sim: "SchedulingSimulator", lot_id: str, ppk: str, oper_id: str) -> Tuple[str, str]:
    lot = sim.lot_pool.get(lot_id)
    return sim._lot_cd_temp(lot_id, lot, ppk=ppk, oper_id=oper_id)


def _bucket_block_reason(
    sim: "SchedulingSimulator",
    eqp_id: str,
    ppk: str,
    oper_id: str,
    wip: Optional[dict],
) -> Tuple[str, str]:
    if not wip or wip.get("wip_qty", 0) <= 0:
        return "no_wip", f"{ppk}/{oper_id} 재공 0"

    if not sim._eqp_can_process(eqp_id, ppk, oper_id):
        model = sim._eqp_model_map[eqp_id]
        if sim._abstract_row_for(eqp_id, ppk, oper_id) is None:
            return "no_arrange", f"EQP MODEL {model}에 {ppk}/{oper_id} arrange 없음"
        return "no_arrange", f"{eqp_id}가 {ppk}/{oper_id} 가공 불가"

    ready_lots: List[str] = []
    earliest_not_ready: Optional[int] = None
    for lid in list(wip.get("lot_ids", [])):
        oper_in_time = sim._wip_lot_meta.get(lid, {}).get("oper_in_time", 0)
        if sim._lot_ready(lid, oper_in_time):
            ready_lots.append(lid)
        elif earliest_not_ready is None or oper_in_time < earliest_not_ready:
            earliest_not_ready = oper_in_time

    if not ready_lots:
        if earliest_not_ready is not None:
            return "wip_not_ready", f"oper_in_time 대기 (가장 이른 {earliest_not_ready}분)"
        return "wip_not_ready", "투입 가능 LOT 없음"

    for lid in ready_lots:
        lot_cd, temp = _lot_cd_temp(sim, lid, ppk, oper_id)
        if not sim._tool_cap_blocks(eqp_id, lot_cd, temp):
            return "unknown", "후보 LOT 있으나 feasible에서 제외됨"

    return "tool_cap_blocked", "tool cap으로 모든 LOT 차단"


def diagnose_eqp(sim: "SchedulingSimulator", eqp_id: str) -> dict:
    """EQP 단위 feasible 옵션 및 차단 bucket 진단."""
    eqp = sim.eqps.get(eqp_id)
    if eqp is None:
        return {"eqp_id": eqp_id, "error": "unknown_eqp"}

    feasible_flats = sim.get_feasible_ppk_oper(eqp_id)
    feasible_options: List[dict] = []
    for flat in feasible_flats:
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        lots = [
            lot for lot in sim.available_lots(eqp_id)
            if lot["PLAN_PROD_ATTR_VAL"] == ppk and lot["oper_id"] == oper_id
        ]
        lot_id = sim._auto_select_lot(eqp_id, lots)
        feasible_options.append({
            "flat": flat,
            "ppk": ppk,
            "oper_id": oper_id,
            "lot_id": lot_id,
        })

    feasible_set = set(feasible_flats)
    blocked_buckets: List[dict] = []
    seen: set = set()
    model = sim._eqp_model_map[eqp_id]

    for tmpl in sim._abstract_template:
        if tmpl["eqp_model"] != model:
            continue
        ppk = tmpl["PLAN_PROD_ATTR_VAL"]
        oper_id = tmpl["oper_id"]
        bucket = (ppk, oper_id)
        if bucket in seen:
            continue
        seen.add(bucket)
        flat = sim.ppk_oper_flat_index(oper_id, ppk)
        if flat in feasible_set:
            continue
        wip = sim._wip_for(ppk, oper_id)
        reason, detail = _bucket_block_reason(sim, eqp_id, ppk, oper_id, wip)
        blocked_buckets.append({
            "ppk": ppk,
            "oper_id": oper_id,
            "reason": reason,
            "detail": detail,
            "wip_qty": int(wip.get("wip_qty", 0)) if wip else 0,
        })

    for (ppk, oper_id), wip in sim._wip_pool.items():
        bucket = (ppk, oper_id)
        if bucket in seen:
            continue
        if wip.get("wip_qty", 0) <= 0:
            continue
        seen.add(bucket)
        flat = sim.ppk_oper_flat_index(oper_id, ppk)
        if flat in feasible_set:
            continue
        reason, detail = _bucket_block_reason(sim, eqp_id, ppk, oper_id, wip)
        blocked_buckets.append({
            "ppk": ppk,
            "oper_id": oper_id,
            "reason": reason,
            "detail": detail,
            "wip_qty": int(wip.get("wip_qty", 0)),
        })

    summary = REASON_LABELS["no_feasible"]
    if eqp.status != "idle":
        summary = f"EQP 상태: {eqp.status}"
    elif feasible_options:
        summary = f"feasible {len(feasible_options)}건"
    elif blocked_buckets:
        top = blocked_buckets[0]
        summary = f"{top['ppk']}/{top['oper_id']}: {top['detail']}"

    return {
        "eqp_id": eqp_id,
        "eqp_status": eqp.status,
        "eqp_model": model,
        "feasible_options": feasible_options,
        "blocked_buckets": blocked_buckets,
        "summary": summary,
    }


def diagnose_assign_failure(
    sim: "SchedulingSimulator",
    eqp_id: str,
    ppk: str,
    oper_id: str,
) -> Tuple[str, str]:
    lots = [
        lot for lot in sim.available_lots(eqp_id)
        if lot["PLAN_PROD_ATTR_VAL"] == ppk and lot["oper_id"] == oper_id
    ]
    lot_id = sim._auto_select_lot(eqp_id, lots)
    if lot_id is None:
        return "lot_select_failed", "LOT 자동 선택 실패"

    lot_cd, temp = _lot_cd_temp(sim, lot_id, ppk, oper_id)
    if sim._tool_cap_blocks(eqp_id, lot_cd, temp):
        return "tool_cap_blocked", f"LOT {lot_id} tool cap 차단 ({lot_cd})"

    wip = sim._wip_for(ppk, oper_id)
    if not wip or wip.get("wip_qty", 0) <= 0:
        return "no_wip", f"{ppk}/{oper_id} 재공 소진"

    return "assign_failed", "배정 실행 실패"


def build_step_decision_entry(
    *,
    step: int,
    sim_time: int,
    sim_time_after: int,
    eqp_id: Optional[str],
    action_flat: int,
    resolved_flat: Optional[int],
    reward: float,
    sim: "SchedulingSimulator",
    terminated: bool,
) -> dict:
    """SchedulingEnv.step() 한 번에 대한 결정 로그 엔트리."""
    oper_count = len(sim._env_data.get("oper_ids", []))
    prod_count = len(sim._env_data.get("prod_keys", []))
    n_flat = max(oper_count * prod_count, 1)

    def flat_to_ppk_oper(flat: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
        if flat is None:
            return None, None
        try:
            return sim.ppk_oper_from_flat(int(flat))
        except Exception:
            return None, None

    req_ppk, req_oper = flat_to_ppk_oper(action_flat % n_flat if n_flat else action_flat)
    res_ppk, res_oper = flat_to_ppk_oper(resolved_flat)

    time_advanced = sim_time_after != sim_time
    diagnosis = diagnose_eqp(sim, eqp_id) if eqp_id else None
    feasible = diagnosis["feasible_options"] if diagnosis else []
    selected = getattr(sim, "_last_decision_assignment", None)
    selected_matches_eqp = bool(
        selected
        and eqp_id is not None
        and selected.get("eqp_id") == eqp_id
    )

    action_corrected = (
        resolved_flat is not None
        and feasible
        and (action_flat % n_flat) != resolved_flat
        and any(opt["flat"] == resolved_flat for opt in feasible)
    )

    assigned_lot_id: Optional[str] = None
    if selected_matches_eqp:
        assigned_lot_id = selected.get("lot_id")
    elif sim._last_assigned and sim._last_assigned.get("eqp_id") == eqp_id:
        assigned_lot_id = sim._last_assigned.get("lot_id")

    selected_eqp_id = selected.get("eqp_id") if selected_matches_eqp else eqp_id
    selected_ppk = selected.get("PLAN_PROD_ATTR_VAL") if selected_matches_eqp else res_ppk
    selected_oper = selected.get("oper_id") if selected_matches_eqp else res_oper

    status = "no_idle_eqp"
    reason = REASON_LABELS["no_idle_eqp"]
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None

    # 이번 step에 배정이 성사됐으면(=selected_matches_eqp) 사후 진단(EQP busy 등)보다
    # '배정' 사유를 우선 표기한다. 진단은 배정 직후 호출돼 EQP가 busy로 보이기 때문.
    if terminated and eqp_id is None:
        status = "sim_done"
        reason = REASON_LABELS["sim_done"]
    elif eqp_id is None:
        status = "no_idle_eqp"
        reason = REASON_LABELS["no_idle_eqp"]
    elif selected_matches_eqp and action_corrected:
        status = "action_corrected"
        reason = (
            f"요청 {req_ppk}/{req_oper} → 보정 {selected_ppk}/{selected_oper}"
            + (f" · LOT {assigned_lot_id}" if assigned_lot_id else "")
        )
    elif selected_matches_eqp:
        status = "assigned"
        reason = (
            f"{eqp_id} ← {selected_ppk}/{selected_oper}"
            + (f" · LOT {assigned_lot_id}" if assigned_lot_id else "")
        )
    elif diagnosis and diagnosis.get("eqp_status") != "idle":
        status = "eqp_not_idle"
        reason = f"EQP {eqp_id} 상태: {diagnosis['eqp_status']}"
    elif not feasible:
        status = "no_feasible"
        reason = diagnosis["summary"] if diagnosis else REASON_LABELS["no_feasible"]
        if diagnosis and diagnosis.get("blocked_buckets"):
            top = diagnosis["blocked_buckets"][0]
            failure_code = top["reason"]
            failure_detail = top["detail"]
    elif reward < 0:
        status = "assign_failed"
        failure_code, failure_detail = diagnose_assign_failure(sim, eqp_id, res_ppk or "", res_oper or "")
        reason = failure_detail
    else:
        status = "assigned"
        reason = (
            f"{eqp_id} ← {res_ppk}/{res_oper}"
            + (f" · LOT {assigned_lot_id}" if assigned_lot_id else "")
        )

    entry: Dict[str, Any] = {
        "step": step,
        "sim_time": sim_time,
        "sim_time_after": sim_time_after,
        "time_advanced": time_advanced,
        "eqp_id": eqp_id,
        "action_requested_flat": action_flat,
        "action_requested_ppk": req_ppk,
        "action_requested_oper": req_oper,
        "resolved_flat": resolved_flat,
        "resolved_ppk": res_ppk,
        "resolved_oper": res_oper,
        "selected_eqp_id": selected_eqp_id,
        "selected_ppk": selected_ppk,
        "selected_oper_id": selected_oper,
        "selected_lot_id": assigned_lot_id,
        "selection_reason": reason,
        "action_corrected": action_corrected,
        "status": status,
        "reason": reason,
        "reward": round(float(reward), 4),
        "reward_breakdown": (
            dict(selected.get("reward_breakdown", {})) if selected_matches_eqp else {}
        ),
        "assigned_lot_id": assigned_lot_id,
        # 배정 성사 step의 사후 진단(EQP busy 기준 feasible/blocked)은 노이즈라 비움
        "feasible_options": [] if selected_matches_eqp else feasible,
        "blocked_buckets": (
            [] if selected_matches_eqp
            else (diagnosis.get("blocked_buckets", []) if diagnosis else [])
        ),
    }
    if failure_code:
        entry["failure_code"] = failure_code
        entry["failure_detail"] = failure_detail
    return entry
