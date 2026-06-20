"""
validation/pacing_metrics.py – 계획 직선 대비 누적 생산 편차 (takt pacing) 메트릭
"""
from typing import Dict, List, Tuple

Key = Tuple[str, str]


def _plan_map(plan: List[dict]) -> Dict[Key, int]:
    out: Dict[Key, int] = {}
    for p in plan:
        key = (p["plan_prod_key"] if "plan_prod_key" in p else p["PLAN_PROD_KEY"],
               p["oper_id"] if "oper_id" in p else p["OPER_ID"])
        out[key] = int(p.get("d0_plan_qty", p.get("D0_PLAN_QTY", 0)))
    return out


def cumulative_qty_at(
    schedule: List[dict],
    ppk: str,
    oper_id: str,
    t_end: int,
) -> int:
    total = 0
    for rec in schedule:
        if rec.get("PLAN_PROD_KEY") != ppk or rec.get("OPER_ID") != oper_id:
            continue
        if rec.get("START_TM", 0) <= t_end:
            total += int(rec.get("WF_QTY", 25))
    return total


def ideal_cumulative(plan_qty: int, t: int, horizon: int) -> float:
    if plan_qty <= 0 or horizon <= 0:
        return 0.0
    return plan_qty * min(max(t, 0), horizon) / horizon


def pacing_metrics(
    schedule: List[dict],
    plan: List[dict],
    horizon: int = 1320,
    step_minutes: int = 60,
) -> dict:
    """
    (PPK, OPER)별 계획 직선 대비 누적 편차.

    Returns:
        mae: 시간 격자 전체 평균 |actual - ideal|
        max_dev: 최대 절대 편차
        final_gap: 종료 시점 누적 달성률 평균 (1.0 = 계획 100%)
        by_key: {(ppk, oper): {mae, max_dev, final_actual, plan_qty, achievement}}
    """
    pmap = _plan_map(plan)
    if not pmap:
        return {"mae": 0.0, "max_dev": 0.0, "final_gap": 0.0, "by_key": {}}

    checkpoints = list(range(0, horizon + 1, step_minutes))
    if checkpoints[-1] != horizon:
        checkpoints.append(horizon)

    by_key: Dict[Key, dict] = {}
    all_errors: List[float] = []
    achievements: List[float] = []

    for key, plan_qty in pmap.items():
        ppk, oper_id = key
        errors: List[float] = []
        for t in checkpoints:
            ideal = ideal_cumulative(plan_qty, t, horizon)
            actual = cumulative_qty_at(schedule, ppk, oper_id, t)
            errors.append(abs(ideal - actual))
        final_actual = cumulative_qty_at(schedule, ppk, oper_id, horizon)
        ach = final_actual / max(plan_qty, 1)
        by_key[key] = {
            "mae": sum(errors) / len(errors),
            "max_dev": max(errors) if errors else 0.0,
            "final_actual": final_actual,
            "plan_qty": plan_qty,
            "achievement": ach,
        }
        all_errors.extend(errors)
        achievements.append(ach)

    return {
        "mae": sum(all_errors) / max(len(all_errors), 1),
        "max_dev": max(all_errors) if all_errors else 0.0,
        "final_gap": sum(achievements) / max(len(achievements), 1),
        "by_key": {
            f"{k[0]}|{k[1]}": v for k, v in by_key.items()
        },
    }


def compare_algorithms(results: List[dict]) -> List[dict]:
    """run_takt_suite 결과 리스트를 MAE 기준 정렬."""
    return sorted(results, key=lambda r: r["metrics"]["mae"])
