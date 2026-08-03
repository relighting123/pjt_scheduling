"""
simulation/selection_reason.py – carrier 배정 시 (PPK,OPER) 선택 사유 / carrier 선택 사유 산출.

RTS_RSLT_MAS/HIS에 적재되는 각 배정 행(EQP×carrier)마다 "왜 이 (PPK,OPER) 버킷을
골랐는지"와 "왜 그 버킷 안에서 이 carrier를 골랐는지"를 사람이 읽을 수 있는 코드+
문구로 남기기 위한 순수 함수 모음. simulator.py가 배정을 확정하는 시점에 호출한다.

- CARRIER 선택 사유: `_auto_select_lot()`의 정렬 기준(sort_key)과 동일한 우선순위를
  후보 목록에 재적용해, 몇 번째 기준에서 승자가 갈렸는지를 역산한다(결정적).
- PPK/OPER 선택 사유: 정책(RL/휴리스틱)이 버킷을 고른 "이유" 자체는 정책 내부
  구현마다 달라 시뮬레이터가 알 수 없으므로, 대신 그 배정에 실제로 반영된
  리워드 항목 분해(same_setup/pacing/flow_balance/conversion/plan_hit)와 버킷의
  계획/재공 현황을 근거로 설명 문구를 만든다. RL 벌크 블록 shaping(bulk_block_bonus
  등)은 assign_ppk_oper() 반환 후 별도 호출(bulk_decision_shaping)로 사후 반영되므로
  여기 포함되지 않는다 — 배정 시점에 확정된 핵심 항목만 다룬다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from simulation.simulator import SchedulingSimulator

CARRIER_REASON_LABELS: Dict[str, str] = {
    "INITIAL_WIP": "초기 재공(이미 투입되어 있던 재공) 우선 선택",
    "DISCRETE_ACTUAL": "실적(discrete) 데이터가 있는 carrier 우선 선택",
    "LOT_CONTINUATION": "동일 LOT의 다른 carrier가 이미 처리 중 — LOT 완결을 위해 우선 선택",
    "EARLIEST_ST": "예상 종료 시각(전환+장수×ST)이 가장 이른 carrier 선택",
    "NO_CONVERSION": "전환(conversion)이 필요 없는 carrier 우선 선택",
    "OPER_IN_TIME": "공정 투입 시각이 가장 이른 carrier 우선 선택(FIFO)",
    "TIE_BREAK": "위 기준이 모두 동률이라 carrier_id/lot_id 순으로 결정적 선택",
    "SINGLE_CANDIDATE": "해당 (PPK,OPER) 버킷에 투입 가능한 carrier가 1건뿐이라 선택",
    "FIXED_QUEUE_EXTERNAL": "eqp_queue_init 외부 확정 큐 — 알고리즘 선택이 아닌 기존 진행 사실 반영",
    "NONE": "선택 근거 없음",
}

# _auto_select_lot()의 sort_key 필드 순서와 1:1 대응.
_CARRIER_SORT_FIELD_CODES = [
    "INITIAL_WIP",
    "DISCRETE_ACTUAL",
    "LOT_CONTINUATION",
    "EARLIEST_ST",
    "NO_CONVERSION",
    "OPER_IN_TIME",
]


def carrier_select_reason(
    sim: "SchedulingSimulator", eqp_id: str, candidates: List[dict],
) -> Tuple[str, str]:
    """버킷 내 carrier 자동 선택(`_auto_select_lot`)과 동일한 기준으로,
    어느 기준에서 승자가 갈렸는지를 (사유코드, 문구)로 반환한다."""
    eligible = [c for c in candidates if sim._lot_conv_discrete_eligible(eqp_id, c)]
    if not eligible:
        return "NONE", CARRIER_REASON_LABELS["NONE"]
    if len(eligible) == 1:
        return "SINGLE_CANDIDATE", CARRIER_REASON_LABELS["SINGLE_CANDIDATE"]

    ppk = eligible[0].get("PLAN_PROD_ATTR_VAL", "")
    oper_id = eligible[0].get("oper_id", "")
    active_logical_lots = sim._active_logical_lot_ids(ppk, oper_id)

    scored: List[Tuple[tuple, str]] = []
    for c in eligible:
        end, needs_conv, carrier, lot_id = sim.earliest_st_combo_score(eqp_id, c)
        logical_id = sim._logical_lot_id(c)
        lot_in_progress = logical_id in active_logical_lots
        key = (
            not c.get("is_initial_wip", False),
            int(c.get("is_abstract", True)),
            not lot_in_progress,
            end,
            needs_conv,
            int(c.get("oper_in_time", 0)),
            carrier,
            lot_id,
        )
        scored.append((key, lot_id))

    remaining = scored
    for i, code in enumerate(_CARRIER_SORT_FIELD_CODES):
        if len(remaining) <= 1:
            break
        min_val = min(item[0][i] for item in remaining)
        filtered = [item for item in remaining if item[0][i] == min_val]
        if len(filtered) < len(remaining) and len(filtered) == 1:
            return code, CARRIER_REASON_LABELS[code]
        remaining = filtered

    return "TIE_BREAK", CARRIER_REASON_LABELS["TIE_BREAK"]


GLOBAL_EARLIEST_ST_REASON: Tuple[str, str] = (
    "GLOBAL_EARLIEST_ST",
    "버킷 정책 없이 전체 idle 장비×재공 조합 중 예상 종료 시각이 가장 이른 "
    "조합을 (EQP,PPK,OPER,carrier) 동시 선택(earliest_st 알고리즘)",
)

PPK_OPER_TERM_LABELS: Dict[str, str] = {
    "same_setup": "동일 셋업(전환 없음) 유지",
    "pacing": "생산 페이싱(pacing) 목표 부합",
    "flow_balance": "하류 공정 재공 흐름 균형",
    "plan_hit": "당일(D0) 계획 수량 달성 기여",
    "conversion": "전환(conversion) 발생에 따른 비용 반영",
    "avoidable_conversion": "회피 가능했던 전환에 대한 추가 비용 반영",
    "idle": "전환 대기(idle) 비용 반영",
}


def ppk_oper_select_reason(
    sim: "SchedulingSimulator",
    eqp_id: str,
    ppk: str,
    oper_id: str,
    terms: Dict[str, float],
    *,
    wip_qty: int,
    mode: str = "BUCKET",
) -> Tuple[str, str]:
    """배정에 실제 반영된 리워드 항목 + 버킷 현황을 근거로 (PPK,OPER) 선택 사유를 만든다."""
    if mode == "GLOBAL_EARLIEST_ST":
        return GLOBAL_EARLIEST_ST_REASON

    plan_meta = sim._env_data.get("plan_meta", {}).get((ppk, oper_id)) or {}
    d0_plan_qty = int(plan_meta.get("d0_plan_qty", 0) or 0)
    done_qty = int(sim.stats.get("completed_qty", {}).get((ppk, oper_id), 0) or 0)

    ranked = sorted(
        ((k, v) for k, v in terms.items() if k in PPK_OPER_TERM_LABELS and v),
        key=lambda kv: -abs(kv[1]),
    )
    if ranked:
        top_code = ranked[0][0]
        factor_txt = ", ".join(
            f"{PPK_OPER_TERM_LABELS[k]}({v:+.2f})" for k, v in ranked
        )
    else:
        top_code = "DEFAULT"
        factor_txt = "추가 리워드 항목 없음(기본 배정 기준)"

    if d0_plan_qty > 0:
        plan_txt = f"D0계획 {d0_plan_qty} / 완료 {done_qty} / 버킷 WIP {wip_qty}건"
    else:
        plan_txt = f"계획 미정 — 보유 WIP {wip_qty}건 소진 목적"

    reason_ctn = f"{eqp_id} ← {ppk}/{oper_id} 선택 근거: {factor_txt} [{plan_txt}]"
    return top_code.upper(), reason_ctn
