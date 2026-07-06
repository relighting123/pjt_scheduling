"""
validation/output_checks.py – 스케줄링 결과(출력) 정합성 검증

run_inference() 결과의 schedule이 입력 데이터(env_data)와 정합한지 사후 검증한다.

검사 항목:
  1. 장비 투입 가능성 – 배정된 (LOT, EQP)가 discrete/abstract arrange 데이터상 허용되는 조합인지
  2. 처리시간 일치 – (END_TM - START_TM)이 입력 데이터(ST × WF_QTY) 기대값과 일치하는지
  3. 배정 완전성 – 입력에 존재하는 모든 LOT이 결과에 배정되었거나, 남은 재공 통계로
     설명되는지 (설명되지 않으면 배정 누락으로 간주)
  4. 강제 배정 – LOT_STAT_CD!=WAIT LOT이 지정된 EQP에, 입력 순서대로 배정됐는지

`validation/runner.py::run_validation()`(test 데이터셋 기준 모델 성능 검증)과는 별개의 기능이다.
"""
from typing import Dict, List, Optional

from utils.helpers import effective_proc_time


def _eqp_can_process(env_data: dict, eqp_id: str, ppk: str, oper_id: str) -> bool:
    """simulation/simulator.py::SchedulingSimulator._eqp_can_process 와 동일한 판정 로직."""
    model = env_data.get("eqp_model_map", {}).get(eqp_id)
    if (ppk, oper_id, model) in env_data.get("abstract_arrange_map", {}):
        return True
    return oper_id in env_data.get("eqp_oper_cap", {}).get(eqp_id, [])


def _expected_proc_time(
    env_data: dict, lot_id: str, eqp_id: str, oper_id: str, ppk: str, wf_qty: int,
) -> Optional[int]:
    """(LOT, EQP, OPER) 조합의 기대 처리시간(분). 근거 데이터가 없으면 None."""
    proc_time_matrix = env_data.get("proc_time_matrix", {})
    st = proc_time_matrix.get((lot_id, eqp_id, oper_id))
    if st is None:
        model = env_data.get("eqp_model_map", {}).get(eqp_id)
        st = env_data.get("abstract_arrange_map", {}).get((ppk, oper_id, model))
    if st is None:
        return None
    return effective_proc_time(st, wf_qty)


def check_eligibility(schedule: List[dict], env_data: dict) -> List[dict]:
    """배정된 (LOT, EQP)가 투입 불가능한 조합인 경우를 찾는다."""
    violations = []
    for row in schedule:
        eqp_id, ppk, oper_id, lot_id = (
            row["EQP_ID"], row["PLAN_PROD_ATTR_VAL"], row["OPER_ID"], row["LOT_ID"],
        )
        if not _eqp_can_process(env_data, eqp_id, ppk, oper_id):
            violations.append({
                "lot_id":        lot_id,
                "eqp_id":        eqp_id,
                "plan_prod_key": ppk,
                "oper_id":       oper_id,
                "reason":        "장비 투입 불가 (discrete/abstract arrange에 해당 조합 없음)",
            })
    return violations


def check_processing_time(
    schedule: List[dict], env_data: dict, *, tolerance_minutes: int = 0,
) -> List[dict]:
    """실제 처리시간(END_TM-START_TM)이 입력 데이터 기대값과 다른 경우를 찾는다."""
    mismatches = []
    for row in schedule:
        lot_id, eqp_id, oper_id, ppk = (
            row["LOT_ID"], row["EQP_ID"], row["OPER_ID"], row["PLAN_PROD_ATTR_VAL"],
        )
        wf_qty = row.get("WF_QTY", 0)
        actual = row["END_TM"] - row["START_TM"]
        expected = _expected_proc_time(env_data, lot_id, eqp_id, oper_id, ppk, wf_qty)
        if expected is None:
            # 근거 데이터 자체가 없는 경우는 check_eligibility()가 별도로 잡는다.
            continue
        if abs(actual - expected) > tolerance_minutes:
            mismatches.append({
                "lot_id":             lot_id,
                "eqp_id":             eqp_id,
                "oper_id":            oper_id,
                "wf_qty":             wf_qty,
                "expected_proc_time": expected,
                "actual_proc_time":   actual,
                "st_used":            row.get("ST"),
            })
    return mismatches


def check_forced_placement(schedule: List[dict], env_data: dict) -> List[dict]:
    """LOT_STAT_CD!=WAIT LOT이 지정된 EQP에, 지정된 순서대로 배정됐는지 확인."""
    violations = []
    forced_queue = env_data.get("eqp_forced_queue", {})
    if not forced_queue:
        return violations

    home_eqp_by_lot = {
        lot_id: eqp_id for eqp_id, lot_ids in forced_queue.items() for lot_id in lot_ids
    }
    actual_eqp_by_lot = {row["LOT_ID"]: row["EQP_ID"] for row in schedule}
    for lot_id, home_eqp in home_eqp_by_lot.items():
        actual_eqp = actual_eqp_by_lot.get(lot_id)
        if actual_eqp is not None and actual_eqp != home_eqp:
            violations.append({
                "lot_id":      lot_id,
                "expected_eqp": home_eqp,
                "actual_eqp":  actual_eqp,
                "reason":      "LOT_STAT_CD 강제 LOT이 지정된 EQP가 아닌 곳에 배정됨",
            })

    for eqp_id, expected_order in forced_queue.items():
        eqp_rows = sorted(
            (r for r in schedule if r["EQP_ID"] == eqp_id),
            key=lambda r: r["START_TM"],
        )
        actual_order = [r["LOT_ID"] for r in eqp_rows if r["LOT_ID"] in expected_order]
        if actual_order != expected_order[:len(actual_order)]:
            violations.append({
                "eqp_id":         eqp_id,
                "expected_order": expected_order,
                "actual_order":   actual_order,
                "reason":         "LOT_STAT_CD 강제 배정 순서가 입력 순서와 다름",
            })
    return violations


def check_completeness(schedule: List[dict], env_data: dict, stats: dict) -> List[dict]:
    """입력 LOT 중 결과에 배정되지 않았고, 잔여 재공 통계로도 설명되지 않는 LOT을 찾는다."""
    scheduled_lot_ids = {row["LOT_ID"] for row in schedule}
    remaining_current_wip = stats.get("remaining_current_wip") or {}
    remaining_wip = stats.get("remaining_wip") or {}

    missing = []
    for lot in env_data.get("lots", []):
        lot_id = lot["lot_id"]
        if lot_id in scheduled_lot_ids:
            continue
        key = f"{lot['plan_prod_key']}|{lot['oper_id']}"
        if remaining_current_wip.get(key, 0) > 0 or remaining_wip.get(key, 0) > 0:
            # 시뮬레이션 종료 시점까지 대기 중인 재공으로 설명됨 (누락 아님)
            continue
        missing.append({
            "lot_id":        lot_id,
            "plan_prod_key": lot["plan_prod_key"],
            "oper_id":       lot["oper_id"],
            "wf_qty":        lot.get("wf_qty"),
            "reason":        "결과 및 잔여 재공 통계 어디에도 없음 (배정 누락 의심)",
        })
    return missing


def validate_schedule_output(
    result: dict, env_data: dict, *, tolerance_minutes: int = 0,
) -> dict:
    """
    추론 결과(result) 전체를 대상으로 위 3개 검사를 모두 수행한다.

    Input:
        result: run_inference() 반환값 ({"schedule", "stats", ...})
        env_data: preprocess() 반환값
        tolerance_minutes: 처리시간 비교 허용 오차(분)
    Output:
        {
          "ok": bool,
          "eligibility_violations": [...],
          "proc_time_mismatches":   [...],
          "unassigned_lots":        [...],
          "summary": {...},
        }
    """
    schedule = result.get("schedule", [])
    stats = result.get("stats", {})

    eligibility_violations = check_eligibility(schedule, env_data)
    proc_time_mismatches = check_processing_time(
        schedule, env_data, tolerance_minutes=tolerance_minutes,
    )
    unassigned_lots = check_completeness(schedule, env_data, stats)
    forced_placement_violations = check_forced_placement(schedule, env_data)

    return {
        "ok": not (
            eligibility_violations or proc_time_mismatches or unassigned_lots
            or forced_placement_violations
        ),
        "eligibility_violations": eligibility_violations,
        "proc_time_mismatches":   proc_time_mismatches,
        "unassigned_lots":        unassigned_lots,
        "forced_placement_violations": forced_placement_violations,
        "summary": {
            "total_scheduled":            len(schedule),
            "eligibility_violation_count": len(eligibility_violations),
            "proc_time_mismatch_count":    len(proc_time_mismatches),
            "unassigned_lot_count":        len(unassigned_lots),
            "forced_placement_violation_count": len(forced_placement_violations),
        },
    }
