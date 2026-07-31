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

# Bulk-Fill PPT·간트에 표시할 전체 보상 항목 (순서 고정)
BULK_REWARD_ORDER = [
    "same_setup",
    "pacing",
    "plan_hit",
    "bulk_block_bonus",
    "conversion",
    "avoidable_conversion",
    "redundant_cover",
    "dedication_misuse",
]

# 항목별 PPT 상세 슬라이드 메타 + 대표 A/B 시나리오
REWARD_TERM_PAGES: List[Dict[str, Any]] = [
    {
        "key": "same_setup",
        "weight": "+1.0",
        "formula": "r = w_same_setup · 1[직전 PPK·OPER 동일 & 재공 잔존]",
        "desc": "셋업 변경 없이 같은 제품·공정을 연속 가공하면 +. 전환 비용 회피를 유도.",
        "scenario_a": {
            "title": "예시 A · 블록 연속",
            "context": "EQP001이 직전과 동일 PPK001·OPER001로 블록 진행",
            "substitution": "1.0 · 1[PPK001→PPK001, OPER001→OPER001] = 1",
            "value": "+1.0",
        },
        "scenario_b": {
            "title": "예시 B · 셋업 변경",
            "context": "PPK001→PPK002로 LOT_CD/TEMP 전환",
            "substitution": "셋업 변경 → 1[동일]=0",
            "value": "0",
        },
        "trace_step": 4,
    },
    {
        "key": "pacing",
        "weight": "+2.5",
        "formula": "r = w_pacing · (|ideal − eff_before| − |ideal − eff_after|) / target",
        "desc": "ideal = target·min(t,horizon)/horizon, eff = done + cover. takt 목표선에 가까워지면 +.",
        "scenario_a": {
            "title": "예시 A · pace 개선",
            "context": "ideal=0.5, eff 0.3→0.5 (target=8)",
            "substitution": "2.5 · (|0.5−0.3| − |0.5−0.5|) / 8 = 2.5 · 0.2/8",
            "value": "+0.06",
        },
        "scenario_b": {
            "title": "예시 B · 과생산",
            "context": "ideal=0.5, eff 0.3→0.8 (takt 초과)",
            "substitution": "2.5 · (0.2 − 0.3) / 8",
            "value": "−Δ",
        },
        "trace_step": 1,
    },
    {
        "key": "plan_hit",
        "weight": "+1.0",
        "formula": "r = w_plan_hit · (gap_before − gap_after) / target   ;  gap = max(target − done, 0)",
        "desc": "달성가능 상한(target) 대비 잔여 gap이 줄면 +. 이미 달성(gap=0)이면 0.",
        "scenario_a": {
            "title": "예시 A · gap 감소",
            "context": "target=8, done 0→1",
            "substitution": "1.0 · (8−7) / 8 = +0.125",
            "value": "+0.13",
        },
        "scenario_b": {
            "title": "예시 B · gap=0",
            "context": "이미 target 달성",
            "substitution": "gap_before=gap_after=0 → 0",
            "value": "0",
        },
        "trace_step": 1,
    },
    {
        "key": "bulk_block_bonus",
        "weight": "+3.0",
        "formula": "r = w_bulk · min(N / takt예산, 1)   [블록 시작 시에만]",
        "desc": "큰 블록으로 커밋할수록 +. N=takt 예산 전량이면 최대.",
        "scenario_a": {
            "title": "예시 A · 블록 시작 N=8",
            "context": "EQP001, PPK001, takt 예산=8",
            "substitution": "3.0 · min(8/8, 1) = +3.0",
            "value": "+3.0",
        },
        "scenario_b": {
            "title": "예시 B · N=1",
            "context": "블록 없이 1캐리어만",
            "substitution": "블록 시작 아님 또는 N=1 → 0",
            "value": "0",
        },
        "trace_step": 1,
    },
    {
        "key": "conversion",
        "weight": "−10.0",
        "formula": "r = w_conversion · 1[LOT_CD/TEMP 셋업 변경]",
        "desc": "LOT_CD 또는 TEMP가 바뀌는 전환 1회마다 고정 패널티.",
        "scenario_a": {
            "title": "예시 A · 전환 없음",
            "context": "동일 셋업 블록 연속",
            "substitution": "1[전환]=0 → 0",
            "value": "0",
        },
        "scenario_b": {
            "title": "예시 B · 전환 발생",
            "context": "PPK001→PPK002 셋업 변경",
            "substitution": "−10.0 · 1 = −10.0",
            "value": "−10.0",
        },
        "trace_step": None,
    },
    {
        "key": "avoidable_conversion",
        "weight": "−8.0",
        "formula": "r = w_avoidable · α   ;  α = 회피가능 비율 (0~1)",
        "desc": "다른 무전환 설비가 커버 가능한데도 전환하면 추가 패널티.",
        "scenario_a": {
            "title": "예시 A · 전환 없음",
            "context": "—",
            "substitution": "전환 없음 → 0",
            "value": "0",
        },
        "scenario_b": {
            "title": "예시 B · 회피가능 전환",
            "context": "다른 설비가 PPK002를 70% 커버 가능",
            "substitution": "−8.0 · 0.7 = −5.6",
            "value": "−5.6",
        },
        "trace_step": None,
    },
    {
        "key": "redundant_cover",
        "weight": "−5.0",
        "formula": "r = w_redundant · min(cover / need, 2)",
        "desc": "다른 셋업 설비가 horizon 내 need를 이미 덮는 버킷을 또 잡으면 −.",
        "scenario_a": {
            "title": "예시 A · cover=0",
            "context": "전담 제품, 다른 설비 커버 없음",
            "substitution": "cover=0 → 0",
            "value": "0",
        },
        "scenario_b": {
            "title": "예시 B · 중복 커버",
            "context": "cover/need ≈ 0.4",
            "substitution": "−5.0 · min(0.4, 2) ≈ −2.0",
            "value": "−2.0",
        },
        "trace_step": 1,
    },
    {
        "key": "dedication_misuse",
        "weight": "−4.0",
        "formula": "r = w_dedication_misuse · 1[더 전용 idle 설비 존재]",
        "desc": "범용 설비가 더 전용적인 idle 설비도 가능한 버킷을 잡으면 −.",
        "scenario_a": {
            "title": "예시 A · 전담 적합",
            "context": "EQP001 전담 PPK001, 더 전용 idle 없음",
            "substitution": "1[전용 오용]=0 → 0",
            "value": "0",
        },
        "scenario_b": {
            "title": "예시 B · 범용 오용",
            "context": "범용 EQP가 전용 EQP 가능 버킷 선점",
            "substitution": "−4.0 · 1 = −4.0",
            "value": "−4.0",
        },
        "trace_step": None,
    },
]


REWARD_ENRICH: Dict[str, Dict[str, Any]] = {
    "same_setup": {
        "plain": "직전과 제품(PPK)·공정(OPER)이 같으면 셋업 변경 없이 +1. 셋업을 이어 붙이라는 신호.",
        "symbols": [("w", "+1.0"), ("1[동일]", "직전 PPK·OPER = 이번 선택")],
        "why": "전환 비용을 피하고 같은 셋업으로 연속 가공하도록 유도.",
    },
    "pacing": {
        "plain": "takt 목표선(ideal)에 실제 진척(eff)이 가까워지면 +, 멀어지면 −.",
        "symbols": [
            ("ideal", "target·min(t,H)/H — 지금까지 만들어야 할 이상량"),
            ("eff", "done + 다른설비 cover — 실제+커버 진척"),
            ("target", "달성가능 상한(재공 한도)"),
        ],
        "why": "계획 속도(takt)에 맞추되, 이미 다른 설비가 덮으면 pace 충족으로 봄.",
    },
    "plan_hit": {
        "plain": "아직 만들어야 할 양(gap)이 줄면 +. 이미 다 만들었으면 0.",
        "symbols": [("gap", "max(target−done, 0)"), ("target", "달성가능 상한")],
        "why": "계획 수량 달성을 직접 장려.",
    },
    "bulk_block_bonus": {
        "plain": "블록 시작 시 N캐리어 연속 처리를 약속(commit)하면 +. 연속 2번째 LOT에는 없음.",
        "symbols": [("N", "커밋 블록 크기"), ("takt 예산", "horizon 내 허용 캐리어 수"), ("w_bulk", "+3.0")],
        "why": "한 설비·한 셋업으로 길게 밀어라 — Bulk-Fill 핵심.",
    },
    "conversion": {
        "plain": "LOT_CD/TEMP 셋업이 바뀌는 전환 1회마다 −10.",
        "symbols": [("w_conv", "−10.0"), ("1[전환]", "셋업 변경 발생")],
        "why": "전환 = 설비 가동 손실. 근본적으로 줄이게 함.",
    },
    "avoidable_conversion": {
        "plain": "굳이 이 설비에서 바꿀 필요 없었는데 전환하면 추가 −8×α.",
        "symbols": [("α", "0~1 회피가능 비율 (다른 설비 커버 등)")],
        "why": "다른 idle 설비에 맡기라는 신호.",
    },
    "redundant_cover": {
        "plain": "다른 설비가 이미 충분히 덮는 제품을 또 잡으면 −.",
        "symbols": [("cover", "다른 EQP horizon 투영 생산"), ("need", "잔여 필요량")],
        "why": "3대가 같은 제품만 몰리는 lockstep 방지.",
    },
    "dedication_misuse": {
        "plain": "범용 설비가 전용 idle 설비도 할 수 있는 일을 가져가면 −4.",
        "symbols": [("breadth", "장비가 할 수 있는 제품 종류 수 — 작을수록 전용")],
        "why": "전용 설비는 전용 제품에, 범용은 범용에.",
    },
}


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


def enriched_reward_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    extra = REWARD_ENRICH.get(meta.get("key", ""), {})
    return {**meta, **extra}


def _term_from_map(by_key: Dict[str, Dict], key: str) -> Dict[str, Any]:
    if key in by_key:
        return by_key[key]
    return {
        "key": key,
        "label": REWARD_LABELS.get(key, key),
        "value": 0.0,
        "formula": "",
        "substitution": "해당 스텝 조건 미충족 → 0",
        "result": "= 0.00",
        "vars": {},
        "active": False,
    }


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
    include_zero: bool = False,
) -> List[Dict[str, Any]]:
    """스텝별 보상 항목 세부 산식. include_zero=True면 8개 항목 전부 반환."""
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

    def emit(
        key: str,
        formula: str,
        substitution: str,
        value: float,
        vars_: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        if not include_zero and not force and abs(value) < 0.005 and key not in breakdown:
            return
        details.append({
            "key": key,
            "label": REWARD_LABELS.get(key, key),
            "value": _r(value),
            "formula": formula,
            "substitution": substitution,
            "result": f"= {value:+.2f}",
            "vars": vars_ or {},
            "active": abs(value) >= 0.005,
        })

    # --- pacing ---
    if cfg.w_pacing > 0 and (include_zero or "pacing" in breakdown):
        w = cfg.w_pacing
        val = breakdown.get("pacing", 0.0)
        ideal_expr = f"target·min(t,H)/H = {target}·{min(max(t, 0), horizon)}/{horizon}"
        emit(
            "pacing",
            "w_pacing · (|ideal − eff_before| − |ideal − eff_after|) / target",
            (
                f"ideal: {ideal_expr} = {_r(ideal)} · "
                f"eff: {_r(eff_before)}→{_r(eff_after)} (done={done_before}, cover={_r(cover)}, wf={wf_qty}) · "
                f"{w}·({_r(err_before)}−{_r(err_after)})/{target}"
            ),
            val,
            {
                "w_pacing": w, "ideal": _r(ideal), "eff_before": _r(eff_before),
                "eff_after": _r(eff_after), "target": target, "t": t, "horizon": horizon,
                "done_before": done_before, "cover": _r(cover), "wf_qty": wf_qty,
            },
        )

    # --- plan_hit ---
    if cfg.w_plan_hit > 0 and (include_zero or "plan_hit" in breakdown):
        w = cfg.w_plan_hit
        val = breakdown.get("plan_hit", 0.0)
        emit(
            "plan_hit",
            "w_plan_hit · (gap_before − gap_after) / target   ;  gap = max(target − done, 0)",
            (
                f"gap: {gap_before}→{gap_after} (done {done_before}→{done_before + wf_qty}, target={target}) · "
                f"{w}·({gap_before}−{gap_after})/{target}"
            ),
            val,
            {"w_plan_hit": w, "gap_before": gap_before, "gap_after": gap_after,
             "target": target, "done_before": done_before, "wf_qty": wf_qty},
        )

    # --- same_setup ---
    if cfg.w_same_setup > 0 and (include_zero or "same_setup" in breakdown):
        w = cfg.w_same_setup
        val = breakdown.get("same_setup", 0.0)
        same = val > 0 or (eqp_prev_oper == oper_id and eqp_prev_prod == ppk)
        emit(
            "same_setup",
            "w_same_setup · 1[동일 제품·공정 & 재공 잔존]",
            f"{w} · 1[{eqp_prev_prod}→{ppk}, {eqp_prev_oper}→{oper_id}] = {w if same else 0}",
            val,
            {"w_same_setup": w, "prev_prod": eqp_prev_prod, "prev_oper": eqp_prev_oper,
             "same_setup": same},
        )

    # --- bulk_block_bonus ---
    if cfg.w_bulk_block_bonus > 0 and (include_zero or "bulk_block_bonus" in breakdown):
        w = cfg.w_bulk_block_bonus
        budget = max(sim._takt_budget_carriers(ppk, oper_id), 1)
        if block_start and block_size > 0:
            ratio = min(block_size / budget, 1.0)
            val = breakdown.get("bulk_block_bonus", w * ratio)
            emit(
                "bulk_block_bonus",
                "w_bulk · min(N / takt예산, 1)   [블록 시작]",
                f"{w} · min({block_size} / {budget}, 1) = {w} · {_r(ratio, 3)}",
                val,
                {"w_bulk": w, "N": block_size, "budget": budget, "ratio": _r(ratio, 3)},
            )
        elif include_zero:
            emit(
                "bulk_block_bonus",
                "w_bulk · min(N / takt예산, 1)   [블록 시작]",
                "블록 시작 아님 → 0",
                0.0,
                {"w_bulk": w, "N": 0, "budget": budget},
            )

    # --- conversion ---
    if include_zero or "conversion" in breakdown:
        w = cfg.w_conversion
        val = breakdown.get("conversion", 0.0)
        emit(
            "conversion",
            "w_conversion · 1[LOT_CD/TEMP 전환]",
            f"{w} · 1[셋업 변경]" if val else f"{w} · 1[전환 없음] = 0",
            val,
            {"w_conversion": w},
        )

    # --- avoidable_conversion ---
    if include_zero or "avoidable_conversion" in breakdown:
        w = cfg.w_avoidable_conversion
        val = breakdown.get("avoidable_conversion", 0.0)
        emit(
            "avoidable_conversion",
            "w_avoidable · α   (α=회피가능 비율)",
            f"{w} · α" if val else f"전환 없음 또는 회피 불가 → 0",
            val,
            {"w_avoidable": w},
        )

    # --- redundant_cover ---
    if cfg.w_redundant_cover != 0 and (include_zero or "redundant_cover" in breakdown):
        w = cfg.w_redundant_cover
        val = breakdown.get("redundant_cover", 0.0)
        need = max(target - done_before, 1)
        cover_r = sim._bucket_projected_cover(ppk, oper_id, exclude_eqp=eqp_id)
        ratio = min(cover_r / need, 2.0)
        emit(
            "redundant_cover",
            "w_redundant · min(cover / need, 2)",
            (
                f"{w} · min({_r(cover_r, 2)} / {need}, 2) = {w} · {_r(ratio, 3)}"
                if abs(val) >= 0.005 else f"cover={_r(cover_r, 2)}, need={need} → 0"
            ),
            val,
            {"w_redundant": w, "cover": _r(cover_r, 2), "need": need},
        )

    # --- dedication_misuse ---
    if cfg.w_dedication_misuse != 0 and (include_zero or "dedication_misuse" in breakdown):
        w = cfg.w_dedication_misuse
        val = breakdown.get("dedication_misuse", 0.0)
        emit(
            "dedication_misuse",
            "w_dedication_misuse · 1[더 전용 idle 설비 존재]",
            f"{w} · 1[전용 오용]" if val else "더 전용 idle 설비 없음 → 0",
            val,
            {"w_dedication_misuse": w},
        )

    by_key = {d["key"]: d for d in details}
    if include_zero:
        return [_term_from_map(by_key, k) for k in BULK_REWARD_ORDER]
    order = list(breakdown.keys())
    details.sort(key=lambda d: order.index(d["key"]) if d["key"] in order else 99)
    return details


def trace_term_detail(trace_steps: List[dict], key: str, preferred_step: Optional[int]) -> Optional[Dict[str, Any]]:
    """트레이스에서 항목별 실측 예시 1건 추출."""
    if preferred_step:
        for st in trace_steps:
            if st.get("step") == preferred_step:
                for det in st.get("reward_formula_full") or st.get("reward_formula") or []:
                    if det.get("key") == key:
                        return {**det, "step": preferred_step, "eqp": st.get("eqp"), "ppk": st.get("ppk")}
    best = None
    for st in trace_steps:
        for det in st.get("reward_formula_full") or st.get("reward_formula") or []:
            if det.get("key") == key and abs(float(det.get("value", 0))) >= 0.005:
                return {**det, "step": st.get("step"), "eqp": st.get("eqp"), "ppk": st.get("ppk")}
    return best
