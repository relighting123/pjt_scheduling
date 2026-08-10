"""
allocation/eqp_allocator.py – PPK/OPER × EQP_MODEL_CD 사전 할당(장비 댓수) 산출

목적
----
스케줄링(RL/휴리스틱) 실행 **이전에** "이번 RULE_TIMEKEY 회차에 어느 PPK/OPER가
어느 EQP_MODEL_CD를 몇 대(댓수) 쓸 수 있는지"를 미리 계산해 DB(RTS_EQPALLOC_PLAN/HIS)에
저장한다. 이 결과는 시뮬레이터의 action mask(get_feasible_ppk_oper)를 추가로
좁히는 데 쓰인다 — RL이 "이 장비 모델은 이 PPK/OPER 몫이 아니다"를 매번 학습으로
알아내는 대신, 결정론적 사전 배분 규칙으로 행동 공간을 줄여준다.

입력은 preprocess.preprocess()와 동일한 raw 딕셔너리 포맷을 그대로 받는다
(data/sql.example/*.sql → *.json 컬럼명 그대로: PLAN_PROD_ATTR_VAL, OPER_ID,
EQP_MODEL_CD, ST, EQP_ID, WF_QTY, D0_PLAN_QTY, PLAN_PRIORITY, OPER_SEQ,
LOT_CD, TEMP, ...).

알고리즘 개요 (설계 근거는 README.md "장비 사전 할당" 참고)
--------------------------------------------------------------------------
1. 시간 축: CAPA/계획 모두 [RULE_TIMEKEY, RULE_TIMEKEY + window_minutes) 창 기준.
   window_minutes 기본값은 시뮬레이터가 이미 쓰는 CONFIG.env.soft_cutoff_minutes
   (RULE_TIMEKEY 07:00 기준 익일 05:00 = 1320분)와 동일 상수를 재사용한다 —
   simulator._achievable_qty/_carrier_takt_minutes가 D0_PLAN_QTY를 이 창의
   목표량으로 그대로 쓰는 기존 관례를 그대로 따른다(별도 비례배분 없음).
2. CAPA_PER_EQP = window_minutes / ST  (ST=분/장, UPH=60/ST 이므로
   UPH × (window_minutes/60) 과 동일한 값).
3. TARGET_QTY(ppk,oper) = min(D0_PLAN_QTY, 도달 가능 재공). 도달 가능 재공은 그
   (PPK,OPER)에 지금 있는 재공 + flow 상 선행 공정들에서 이 창 안에 흘러들어올
   수 있는 재공까지 합산한다(simulator._achievable_qty와 동일한 flow_prev
   체인 합산 방식, 시간 할인 없음) — 그렇지 않으면 하류 공정이 "지금 당장은
   비어 있지만 곧 채워질" 재공을 과소평가해 장비를 덜 받는 쪽으로 치우친다.
   WIP_QTY 컬럼도 이 값(도달 가능 재공)을 그대로 저장한다.
4. PPK 우선순위(PLAN_PRIORITY, 작을수록 우선) 오름차순으로 처리한다.
5. 전용 모델(abstract_arrange 상 해당 (PPK,OPER)에 모델이 1개뿐인 경우) 수요를
   공용 모델 수요보다 먼저 배정한다 — 대체 불가능한 수요이므로.
6. 공용 모델(PPK,OPER에 모델이 2개 이상 가능)은 같은 PPK 안에서 OPER별 목표 수량을
   예외 없이 균등 배분한다(공정별 CAPA 편차 방지, 마지막 OPER 달성을 위해 상류
   공정도 꾸준히 생산) — 목표 수량(TARGET_QTY) 자체가 이미 도달 가능 재공(점 3,
   상류에서 흘러들어올 재공 포함)을 반영하므로, 상류 공정이 병목이라 재공이
   쌓여 있는 흔한 경우는 하류 OPER의 목표도 함께 커져 있어 균등 배분이 특정
   OPER을 부당하게 캡하지 않는다. 재공이 한쪽에 쏠려도 별도 예외 없이 동일한
   규칙을 적용한다(과거의 WIP 쏠림 배수 기반 예외는 절벽 현상·처리 순서 편향
   등 부작용이 있어 제거했다). 모델 선택은 "지금까지 전체 보유 대수 대비 가장
   적게 쓰인 모델"부터 우선해 같은 모델로 수요가 쏠리지 않도록 분산한다.
7. 실제 대수를 확보할 때는 EQP_ID 단위로, 이미 그 (PPK,OPER)의 배치(batch_info의
   LOT_CD/TEMP)로 셋업돼 있어 전환이 필요 없는 장비를 우선 쓰고, 모자랄 때만
   전환이 필요한 장비를 CONFIG.env.max_conversions(전체 상한)/
   max_conversions_per_eqp(장비별 상한, 0이면 전환 자체를 아예 허용하지 않음)
   예산 안에서 추가로 끌어쓴다 — 시뮬레이터의 _would_need_conversion/
   _conversion_limit_blocks와 동일한 판단 기준(eqp_initial_state 대비
   batch_info)을 미리 반영해, 실제로는 전환 예산이 없어 못 돌릴 장비까지
   ALLOC_EQP_CNT/ALLOC_CAPA_QTY에 낙관적으로 잡히는 것을 막는다.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config import CONFIG, RULE_TIMEKEY_FMT, normalize_rule_timekey

PpkOper = Tuple[str, str]
LotBatch = Tuple[str, str]  # (LOT_CD, TEMP)


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _carrier_key(row: dict) -> str:
    carrier_id = row.get("CARRIER_ID")
    return str(carrier_id) if carrier_id else str(row["LOT_ID"])


def _build_flow_seq(flow_raw: List[dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in flow_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        out.setdefault(ppk, {})[r["OPER_ID"]] = _coerce_int(r.get("OPER_SEQ"), 0)
    return out


def _build_flow_prev(flow_raw: List[dict]) -> Dict[str, Dict[str, Optional[str]]]:
    """(ppk,oper) → 선행 OPER (flow.OPER_SEQ 오름차순 기준, 첫 공정은 None).
    simulator._flow_prev와 동일 규칙 — 도달 가능 재공(유입) 합산에 쓴다."""
    steps: Dict[str, List[Tuple[int, str]]] = {}
    for r in flow_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        steps.setdefault(ppk, []).append((_coerce_int(r.get("OPER_SEQ"), 0), r["OPER_ID"]))
    prev_map: Dict[str, Dict[str, Optional[str]]] = {}
    for ppk, seq_list in steps.items():
        ordered = [op for _, op in sorted(seq_list)]
        prev_map[ppk] = {op: (ordered[i - 1] if i > 0 else None) for i, op in enumerate(ordered)}
    return prev_map


def _reachable_wip_qty(
    keys: List[PpkOper],
    wip_current: Dict[PpkOper, int],
    flow_prev: Dict[str, Dict[str, Optional[str]]],
) -> Dict[PpkOper, int]:
    """keys(관심 (ppk,oper) 전체) → 현재 재공 + 선행 공정 체인에서 이 창 안에
    흘러들어올 수 있는 재공(유입) 합. simulator._achievable_qty()와 동일하게
    시간 할인 없이 상류 재공 전량을 도달 가능으로 간주한다."""
    reachable: Dict[PpkOper, int] = {}
    for ppk, oper in keys:
        total = wip_current.get((ppk, oper), 0)
        seen = {oper}
        prev = flow_prev.get(ppk, {}).get(oper)
        while prev and prev not in seen:
            seen.add(prev)
            total += wip_current.get((ppk, prev), 0)
            prev = flow_prev.get(ppk, {}).get(prev)
        reachable[(ppk, oper)] = total
    return reachable


def _build_eligible_models(
    abstract_raw: List[dict],
) -> Dict[PpkOper, Dict[str, float]]:
    """(PPK,OPER) → {EQP_MODEL_CD: ST(분/장)}. 중복 행은 먼저 나온 값을 유지한다."""
    out: Dict[PpkOper, Dict[str, float]] = {}
    for r in abstract_raw:
        key = (r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"])
        model = str(r["EQP_MODEL_CD"])
        st = _coerce_float(r.get("ST"))
        out.setdefault(key, {})
        out[key].setdefault(model, st)
    return out


def _build_eqp_ids_by_model(discrete_raw: List[dict]) -> Dict[str, List[str]]:
    """EQP_MODEL_CD → 전체 보유 EQP_ID 목록(discrete_arrange의 distinct EQP_ID 기준,
    정렬해 배정 결과가 결정적이도록 함)."""
    eqp_by_model: Dict[str, set] = {}
    for r in discrete_raw:
        model = r.get("EQP_MODEL_CD")
        if not model:
            continue
        eqp_by_model.setdefault(str(model), set()).add(r["EQP_ID"])
    return {model: sorted(eqps) for model, eqps in eqp_by_model.items()}


def _build_wip_qty(discrete_raw: List[dict]) -> Dict[PpkOper, int]:
    """(PPK,OPER) → 그 공정에 지금 있는 재공 수량(유입 미포함). discrete_arrange는
    (EQP,LOT) M:N 후보 선언이므로(같은 carrier가 여러 EQP 후보 행에 중복 등장),
    carrier 단위로 한 번만 센다(preprocess._carrier_instance_id와 동일한 dedup
    원칙)."""
    seen: Dict[Tuple[PpkOper, str], int] = {}
    for r in discrete_raw:
        key = (r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"])
        carrier = _carrier_key(r)
        seen[(key, carrier)] = _coerce_int(r.get("WF_QTY"), 0)
    totals: Dict[PpkOper, int] = {}
    for (key, _carrier), wf_qty in seen.items():
        totals[key] = totals.get(key, 0) + wf_qty
    return totals


def _build_plan_meta(plan_raw: List[dict]) -> Tuple[Dict[PpkOper, int], Dict[str, int]]:
    """plan → {(ppk,oper): d0_plan_qty}, {ppk: priority(min across its oper rows)}."""
    d0_qty: Dict[PpkOper, int] = {}
    priority_by_ppk: Dict[str, int] = {}
    for r in plan_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        oper = r["OPER_ID"]
        d0_qty[(ppk, oper)] = _coerce_int(r.get("D0_PLAN_QTY"), 0)
        priority = _coerce_int(r.get("PLAN_PRIORITY"), 1)
        if ppk not in priority_by_ppk or priority < priority_by_ppk[ppk]:
            priority_by_ppk[ppk] = priority
    return d0_qty, priority_by_ppk


def _normalize_batch(lot_cd: Optional[str], temp: Optional[str]) -> Optional[LotBatch]:
    lot_cd = (lot_cd or "").strip()
    if not lot_cd:
        return None
    return (lot_cd, (temp or "").strip())


def _build_batch_info_map(batch_info_raw: List[dict]) -> Dict[PpkOper, LotBatch]:
    """(PPK,OPER) → 그 공정을 처리하려면 필요한 배치(LOT_CD,TEMP).
    simulator._lot_cd_temp()가 참조하는 batch_info_map과 동일 소스."""
    out: Dict[PpkOper, LotBatch] = {}
    for r in batch_info_raw:
        batch = _normalize_batch(r.get("LOT_CD"), r.get("TEMP"))
        if batch is None:
            continue
        out[(r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"])] = batch
    return out


def _build_eqp_setup_map(eqp_initial_state_raw: List[dict]) -> Dict[str, LotBatch]:
    """EQP_ID → 현재 셋업된 배치(LOT_CD,TEMP). 셋업 정보가 없는(=아직 한 번도
    투입된 적 없는) 장비는 맵에 없다 — simulator._would_need_conversion()이
    prev_lot_cd=None인 장비를 항상 "전환 불필요"로 보는 것과 동일하게, 이후
    조회에서 없으면 전환 불필요로 취급한다."""
    out: Dict[str, LotBatch] = {}
    for r in eqp_initial_state_raw:
        eid = str(r.get("EQP_ID", "")).strip()
        batch = _normalize_batch(r.get("LOT_CD"), r.get("TEMP"))
        if eid and batch is not None:
            out[eid] = batch
    return out


def _capa_per_eqp(st_per_wafer: float, window_minutes: int) -> float:
    if st_per_wafer <= 0:
        return 0.0
    return window_minutes / st_per_wafer


def _units_needed(qty: float, capa_per_eqp: float) -> int:
    if capa_per_eqp <= 0 or qty <= 0:
        return 0
    return math.ceil(qty / capa_per_eqp)


def compute_eqp_allocation(
    raw: Dict[str, List[dict]],
    *,
    fac_id: str,
    rule_timekey: str,
    window_minutes: Optional[int] = None,
) -> List[dict]:
    """
    목적: PPK/OPER × EQP_MODEL_CD 할당 대수 산출 (DB RTS_EQPALLOC_PLAN 행 포맷)
    Input:
        raw: {"abstract_arrange": [...], "discrete_arrange": [...], "plan": [...],
              "flow": [...], "batch_info": [...](선택), "eqp_initial_state": [...](선택)}
             (모두 SQL 원본 컬럼명 그대로 — data/loader/preprocess.preprocess()와 동일 입력)
        fac_id, rule_timekey: 이번 회차 식별자
        window_minutes: 계획/CAPA 산정 창 길이(분). 기본 CONFIG.env.soft_cutoff_minutes.
    Output:
        List[dict] — RTS_EQPALLOC_PLAN 컬럼과 1:1 대응하는 행 목록(정렬됨:
        PLAN_PROD_ATTR_VAL, OPER_ID, EQP_MODEL_CD 순)

    CONFIG.env.max_conversions/max_conversions_per_eqp(전환 횟수 상한, 미설정 시
    무제한)를 이번 실행 전체에 공유되는 전환 예산으로 참조한다 — batch_info/
    eqp_initial_state가 없으면(둘 다 선택 입력) 전환 판단을 생략하고 모든 장비를
    전환 불필요로 취급한다(하위 호환, fail-open).
    """
    window_minutes = window_minutes or CONFIG.env.soft_cutoff_minutes
    rule_timekey = normalize_rule_timekey(rule_timekey)
    window_start_dt = datetime.strptime(rule_timekey, RULE_TIMEKEY_FMT)
    window_end_dt = window_start_dt + timedelta(minutes=window_minutes)

    abstract_raw = raw.get("abstract_arrange", [])
    discrete_raw = raw.get("discrete_arrange", [])
    plan_raw = raw.get("plan", [])
    flow_raw = raw.get("flow", [])
    batch_info_raw = raw.get("batch_info", [])
    eqp_initial_state_raw = raw.get("eqp_initial_state", [])

    eligible_models = _build_eligible_models(abstract_raw)
    eqp_ids_by_model = _build_eqp_ids_by_model(discrete_raw)
    model_total_eqp = {model: len(ids) for model, ids in eqp_ids_by_model.items()}
    wip_current = _build_wip_qty(discrete_raw)
    flow_prev = _build_flow_prev(flow_raw)
    wip_reachable = _reachable_wip_qty(list(eligible_models.keys()), wip_current, flow_prev)
    d0_plan_qty, ppk_priority = _build_plan_meta(plan_raw)
    flow_seq = _build_flow_seq(flow_raw)
    batch_info_map = _build_batch_info_map(batch_info_raw)
    eqp_setup_map = _build_eqp_setup_map(eqp_initial_state_raw)

    # 전환 예산: max_conversions(전체)는 이번 실행 전체가 공유하는 단일 카운터.
    # max_conversions_per_eqp=0이면 단일 정적 배정(같은 EQP를 이 실행 안에서
    # 두 번 이상 쓰지 않음)이라도 그 장비의 "첫 전환"조차 금지되므로, 전환이
    # 필요한 장비는 아예 배정 후보에서 제외한다.
    max_conversions = CONFIG.env.max_conversions
    max_conversions_per_eqp = CONFIG.env.max_conversions_per_eqp
    conversion_allowed = max_conversions_per_eqp is None or max_conversions_per_eqp >= 1
    conv_budget = {"remaining": max_conversions}  # None=무제한

    # (ppk,oper) 목표 수량 = min(계획, 도달 가능 재공). 계획/재공 중 하나라도
    # 없으면(target=0) 대상에서 제외한다(=이 (ppk,oper)에 대해서는 할당 행을
    # 만들지 않고, 마스킹 단계에서 "제약 없음"으로 폴백하게 한다 — 일시적 계획
    # 누락이 정당한 WIP 배정까지 막는 부작용을 피한다).
    demand: Dict[PpkOper, int] = {}
    for key, models in eligible_models.items():
        if not models:
            continue
        target = min(d0_plan_qty.get(key, 0), wip_reachable.get(key, 0))
        if target > 0:
            demand[key] = target

    exclusive_keys = [k for k in demand if len(eligible_models[k]) == 1]
    shared_keys = [k for k in demand if len(eligible_models[k]) > 1]

    def _sort_key(k: PpkOper):
        ppk, oper = k
        return (ppk_priority.get(ppk, 1), ppk, flow_seq.get(ppk, {}).get(oper, 0), oper)

    exclusive_keys.sort(key=_sort_key)

    remaining_eqp_by_model: Dict[str, List[str]] = {
        model: list(ids) for model, ids in eqp_ids_by_model.items()
    }
    model_usage: Dict[str, int] = {m: 0 for m in model_total_eqp}

    rows: List[dict] = []

    def _needs_conversion(eqp_id: str, target: Optional[LotBatch]) -> bool:
        if target is None:
            return False
        setup = eqp_setup_map.get(eqp_id)
        if setup is None:
            return False
        return setup != target

    def _draw_units(model: str, want: int, ppk: str, oper: str) -> Tuple[List[str], int]:
        """model 풀에서 최대 want대까지 실제 EQP_ID를 뽑는다. 이미 이 (ppk,oper)
        배치로 셋업된(전환 불필요) 장비를 우선 쓰고, 모자라면 전환이 필요한
        장비를 전환 예산이 남은 한도까지 추가로 쓴다.
        반환값: (뽑힌 EQP_ID 목록, 그중 전환이 필요했던 개수)."""
        if want <= 0:
            return [], 0
        pool = remaining_eqp_by_model.get(model, [])
        if not pool:
            return [], 0

        target = batch_info_map.get((ppk, oper))
        free = [e for e in pool if not _needs_conversion(e, target)]
        conv = [e for e in pool if _needs_conversion(e, target)]

        free_take = free[:want]
        remaining_want = want - len(free_take)
        conv_take: List[str] = []
        if remaining_want > 0 and conversion_allowed and conv:
            budget = conv_budget["remaining"]
            n = remaining_want if budget is None else max(min(remaining_want, budget), 0)
            conv_take = conv[:n]
            if budget is not None:
                conv_budget["remaining"] = budget - len(conv_take)

        picked = free_take + conv_take
        if picked:
            picked_set = set(picked)
            remaining_eqp_by_model[model] = [e for e in pool if e not in picked_set]
        return picked, len(conv_take)

    def _append_row(ppk: str, oper: str, model: str, st: float, alloc_cnt: int,
                     target_qty: float, is_exclusive: bool, needs_conv_cnt: int = 0) -> None:
        capa = _capa_per_eqp(st, window_minutes)
        rows.append({
            "FAC_ID": fac_id,
            "RULE_TIMEKEY": rule_timekey,
            "PLAN_PROD_ATTR_VAL": ppk,
            "OPER_ID": oper,
            "EQP_MODEL_CD": model,
            "PLAN_PRIORITY": ppk_priority.get(ppk, 1),
            "IS_EXCLUSIVE_MODEL": "Y" if is_exclusive else "N",
            "WINDOW_START_TM": rule_timekey,
            "WINDOW_END_TM": window_end_dt.strftime(RULE_TIMEKEY_FMT),
            "WINDOW_MINUTES": window_minutes,
            "PLAN_QTY": d0_plan_qty.get((ppk, oper), 0),
            "WIP_QTY": wip_reachable.get((ppk, oper), 0),
            "TARGET_QTY": target_qty,
            "ST": st,
            "CAPA_PER_EQP": round(capa, 4),
            "ALLOC_EQP_CNT": alloc_cnt,
            "ALLOC_CAPA_QTY": round(alloc_cnt * capa, 4),
            "TOTAL_EQP_CNT": model_total_eqp.get(model, 0),
            "NEEDS_CONV_EQP_CNT": needs_conv_cnt,
        })

    # ── Phase A: 전용 모델(대체 불가) 수요를 PPK 우선순위 순으로 먼저 배정 ──────
    for key in exclusive_keys:
        ppk, oper = key
        model, st = next(iter(eligible_models[key].items()))
        capa = _capa_per_eqp(st, window_minutes)
        target = demand[key]
        need = _units_needed(target, capa)
        picked, conv_n = _draw_units(model, need, ppk, oper)
        if not picked:
            continue
        model_usage[model] = model_usage.get(model, 0) + len(picked)
        _append_row(ppk, oper, model, st, len(picked), target, is_exclusive=True, needs_conv_cnt=conv_n)

    # ── Phase B: 공용 모델 수요 — PPK 우선순위 순, PPK 내부는 OPER 균등 배분 ────
    # 공용모델 후보를 "지금까지 전체 보유 대수 대비 가장 적게 쓰인 모델"부터
    # 우선 고르는 정렬 규칙(점 6). Phase A/B가 공유하는 remaining_eqp_by_model/
    # model_usage/conv_budget을 참조하므로 클로저로 둔다.
    def _ranked_candidates(key: PpkOper):
        return sorted(
            eligible_models[key].items(),
            key=lambda item: (
                model_usage.get(item[0], 0) / model_total_eqp[item[0]]
                if model_total_eqp.get(item[0]) else 1.0,
                -_capa_per_eqp(item[1], window_minutes),
                item[0],
            ),
        )

    def _take_one_unit(key: PpkOper) -> float:
        """이 (ppk,oper)에 가장 여유 있는 모델에서 1대를 배정. 반환값은 확보한
        CAPA(장), 배정 가능한 모델이 남아있지 않으면 0.0."""
        ppk_, oper = key
        for model, st in _ranked_candidates(key):
            picked, conv_n = _draw_units(model, 1, ppk_, oper)
            if not picked:
                continue
            model_usage[model] = model_usage.get(model, 0) + 1
            _append_row(ppk_, oper, model, st, 1, demand[key], is_exclusive=False, needs_conv_cnt=conv_n)
            return _capa_per_eqp(st, window_minutes)
        return 0.0

    shared_by_ppk: Dict[str, List[PpkOper]] = {}
    for key in shared_keys:
        shared_by_ppk.setdefault(key[0], []).append(key)

    for ppk in sorted(shared_by_ppk, key=lambda p: (ppk_priority.get(p, 1), p)):
        opers = sorted(shared_by_ppk[ppk], key=lambda k: (flow_seq.get(ppk, {}).get(k[1], 0), k[1]))

        # 모든 OPER을 예외 없이 균등 배분한다 — 라운드로빈(1대씩 순환 배정)으로
        # 특정 OPER이 먼저 처리된다고 물량을 독식하지 않도록 함(점 6: 공정별
        # CAPA 편차 방지).
        equal_share = sum(demand[k] for k in opers) / len(opers)
        remaining_qty = {k: min(demand[k], equal_share) for k in opers}
        progressed = True
        while progressed:
            progressed = False
            for key in opers:
                if remaining_qty[key] <= 0:
                    continue
                gained = _take_one_unit(key)
                if gained <= 0:
                    continue
                remaining_qty[key] -= gained
                progressed = True

    rows.sort(key=lambda r: (r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"], r["EQP_MODEL_CD"]))
    return _merge_rows(rows)


def _merge_rows(rows: List[dict]) -> List[dict]:
    """라운드로빈으로 (ppk,oper,model)당 여러 번(1대씩) 생성된 행을 합산한다."""
    merged: Dict[Tuple[str, str, str], dict] = {}
    for r in rows:
        key = (r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"], r["EQP_MODEL_CD"])
        if key not in merged:
            merged[key] = dict(r)
        else:
            merged[key]["ALLOC_EQP_CNT"] += r["ALLOC_EQP_CNT"]
            merged[key]["NEEDS_CONV_EQP_CNT"] += r["NEEDS_CONV_EQP_CNT"]
            merged[key]["ALLOC_CAPA_QTY"] = round(
                merged[key]["ALLOC_CAPA_QTY"] + r["ALLOC_CAPA_QTY"], 4,
            )
    return [merged[k] for k in sorted(merged)]


def build_eqp_alloc_map(alloc_rows: List[dict]) -> Dict[PpkOper, Dict[str, int]]:
    """할당 행 목록 → {(ppk,oper): {eqp_model_cd: alloc_eqp_cnt}} (마스킹용, ALLOC_EQP_CNT>0만).

    시뮬레이터는 (ppk,oper) 키가 이 맵에 아예 없으면 "이번 회차 할당 계산 대상이
    아니었다"로 보고 제약 없이 기존 abstract_arrange 매칭 결과를 그대로 쓴다
    (일시적 계획 누락 등으로 정당한 배정까지 막는 것을 방지 — fail-open).
    """
    out: Dict[PpkOper, Dict[str, int]] = {}
    for r in alloc_rows:
        cnt = _coerce_int(r.get("ALLOC_EQP_CNT"), 0)
        if cnt <= 0:
            continue
        key = (r["PLAN_PROD_ATTR_VAL"], r["OPER_ID"])
        out.setdefault(key, {})[r["EQP_MODEL_CD"]] = cnt
    return out
