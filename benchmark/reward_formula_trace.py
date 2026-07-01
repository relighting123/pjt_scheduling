"""PPT용 보상 항목별 세부 산식 생성."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.simulator import SchedulingSimulator

REWARD_LABELS = {
    "same_setup": "동일 셋업",
    "pacing": "페이싱",
    "plan_hit": "계획 달성",
    "flow_balance": "흐름 균형",
    "idle": "유휴",
    "conversion": "전환",
    "avoidable_conversion": "회피가능 전환",
    "bulk_block_bonus": "블록 보너스",
    "dedication_misuse": "전용 오용",
    "redundant_cover": "중복 커버",
}


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


def build_reward_formula_details(
    sim: "SchedulingSimulator",
    *,
    ppk: str,
    oper_id: str,
    eqp_id: str,
    wf_qty: int,
    t: int,
    breakdown: Dict[str, float],
    block_start: bool,
    block_size: int,
    eqp_prev_prod: Optional[str],
    eqp_prev_oper: Optional[str],
    done_before: int,
) -> List[Dict[str, Any]]:
    """breakdown에 있는 항목마다 세부 산식·대입값을 생성."""
    cfg = sim._reward_cfg
    target = max(sim._achievable_qty(ppk, oper_id), 1)
    horizon = max(sim.soft_cutoff, 1)
    ideal = target * min(max(t, 0), horizon) / horizon
    cover = 0.0
    if cfg.pacing_coverage_scale > 0:
        cover = cfg.pacing_coverage_scale * sim._bucket_projected_cover(
            ppk, oper_id, exclude_eqp=eqp_id,
        )
    eff_before = done_before + cover
    eff_after = eff_before + wf_qty
    err_before = abs(ideal - eff_before)
    err_after = abs(ideal - eff_after)
    gap_before = max(target - done_before, 0)
    gap_after = max(target - done_before - wf_qty, 0)

    details: List[Dict[str, Any]] = []

    def add(
        key: str,
        formula: str,
        substitution: str,
        result: str,
        value: float,
        vars_: Optional[Dict[str, Any]] = None,
    ) -> None:
        if key not in breakdown or abs(breakdown[key]) < 0.005:
            return
        details.append({
            "key": key,
            "label": REWARD_LABELS.get(key, key),
            "value": _r(value),
            "formula": formula,
            "substitution": substitution,
            "result": result,
            "vars": vars_ or {},
        })

    if "pacing" in breakdown:
        w = cfg.w_pacing
        val = breakdown["pacing"]
        ideal_expr = f"target·min(t,horizon)/horizon = {target}·{min(max(t, 0), horizon)}/{horizon}"
        add(
            "pacing",
            "w_pacing · (|ideal − eff_before| − |ideal − eff_after|) / target",
            (
                f"ideal: {ideal_expr} = {_r(ideal)} · "
                f"eff: {_r(eff_before)}→{_r(eff_after)} (done={done_before}, cover={_r(cover)}, wf={wf_qty}) · "
                f"{w}·({_r(err_before)}−{_r(err_after)})/{target}"
            ),
            f"= {val:+.2f}",
            val,
            {
                "w_pacing": w,
                "ideal": _r(ideal),
                "eff_before": _r(eff_before),
                "eff_after": _r(eff_after),
                "target": target,
                "t": t,
                "horizon": horizon,
                "done_before": done_before,
                "cover": _r(cover),
                "wf_qty": wf_qty,
            },
        )

    if "plan_hit" in breakdown:
        w = cfg.w_plan_hit
        val = breakdown["plan_hit"]
        add(
            "plan_hit",
            "w_plan_hit · (gap_before − gap_after) / target   ;  gap = max(target − done, 0)",
            (
                f"gap: {gap_before}→{gap_after} (done {done_before}→{done_before + wf_qty}, target={target}) · "
                f"{w}·({gap_before}−{gap_after})/{target}"
            ),
            f"= {val:+.2f}",
            val,
            {
                "w_plan_hit": w,
                "gap_before": gap_before,
                "gap_after": gap_after,
                "target": target,
                "done_before": done_before,
                "wf_qty": wf_qty,
            },
        )

    if "same_setup" in breakdown:
        w = cfg.w_same_setup
        val = breakdown["same_setup"]
        same = (eqp_prev_oper == oper_id and eqp_prev_prod == ppk)
        add(
            "same_setup",
            "w_same_setup · 1[동일 제품·공정 & 재공 잔존]",
            f"{w} · 1[{eqp_prev_prod}→{ppk}, {eqp_prev_oper}→{oper_id}] = {w if same else 0}",
            f"= {val:+.2f}",
            val,
            {
                "w_same_setup": w,
                "prev_prod": eqp_prev_prod,
                "prev_oper": eqp_prev_oper,
                "same_setup": same,
            },
        )

    if "bulk_block_bonus" in breakdown and block_start:
        w = cfg.w_bulk_block_bonus
        budget = max(sim._takt_budget_carriers(ppk, oper_id), 1)
        ratio = min(block_size / budget, 1.0)
        val = breakdown["bulk_block_bonus"]
        add(
            "bulk_block_bonus",
            "w_bulk · min(N / takt예산, 1)",
            f"{w} · min({block_size} / {budget}, 1) = {w} · {_r(ratio, 3)}",
            f"= {val:+.2f}",
            val,
            {"w_bulk": w, "N": block_size, "budget": budget, "ratio": _r(ratio, 3)},
        )

    if "dedication_misuse" in breakdown:
        w = cfg.w_dedication_misuse
        val = breakdown["dedication_misuse"]
        add(
            "dedication_misuse",
            "w_dedication_misuse · 1[더 전용 idle 설비 존재]",
            f"{w} · 1[전용 오용]",
            f"= {val:+.2f}",
            val,
            {"w_dedication_misuse": w},
        )

    if "redundant_cover" in breakdown:
        w = cfg.w_redundant_cover
        done = done_before  # 블록 shaping 시점 ≈ 배정 직전 done
        need = max(target - done, 1)
        cover_r = sim._bucket_projected_cover(ppk, oper_id, exclude_eqp=eqp_id)
        ratio = min(cover_r / need, 2.0)
        val = breakdown["redundant_cover"]
        add(
            "redundant_cover",
            "w_redundant · min(cover / need, 2)",
            f"{w} · min({_r(cover_r, 2)} / {need}, 2) = {w} · {_r(ratio, 3)}",
            f"= {val:+.2f}",
            val,
            {"w_redundant": w, "cover": _r(cover_r, 2), "need": need},
        )

    if "conversion" in breakdown:
        w = cfg.w_conversion
        val = breakdown["conversion"]
        add(
            "conversion",
            "w_conversion · 1[LOT_CD/TEMP 전환]",
            f"{w} · 1[셋업 변경]",
            f"= {val:+.2f}",
            val,
            {"w_conversion": w},
        )

    if "avoidable_conversion" in breakdown:
        w = cfg.w_avoidable_conversion
        val = breakdown["avoidable_conversion"]
        add(
            "avoidable_conversion",
            "w_avoidable · avoidable_frac",
            f"{w} · α",
            f"= {val:+.2f}",
            val,
            {"w_avoidable": w},
        )

    # breakdown 순서 유지
    order = list(breakdown.keys())
    details.sort(key=lambda d: order.index(d["key"]) if d["key"] in order else 99)
    return details
