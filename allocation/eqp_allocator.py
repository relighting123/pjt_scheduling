"""
allocation/eqp_allocator.py – PPK/OPER × EQP_MODEL_CD 사전 할당(장비 댓수) 산출

목적
----
스케줄링(RL/휴리스틱) 실행 **이전에** "이번 RULE_TIMEKEY 회차에 어느 PPK/OPER가
어느 EQP_MODEL_CD를 몇 대 쓸 수 있는지"를 미리 계산해 DB(RTS_EQPALLOC_PLAN/HIS)에
저장한다. 이 결과는 시뮬레이터의 action mask(get_feasible_ppk_oper)를 추가로
좁히는 데 쓰인다 — RL이 "이 장비 모델은 이 PPK/OPER 몫이 아니다"를 매번 학습으로
알아내는 대신, 결정론적 사전 배분 규칙으로 행동 공간을 줄여준다.

입력은 preprocess.preprocess()와 동일한 raw 딕셔너리 포맷을 그대로 받는다
(data/sql.example/*.sql → *.json 컬럼명 그대로: PLAN_PROD_ATTR_VAL, OPER_ID,
EQP_MODEL_CD, ST, EQP_ID, WF_QTY, D0_PLAN_QTY, PLAN_PRIORITY, OPER_SEQ, ...).

알고리즘 개요 (설계 근거는 README.md "장비 할당(Action Masking) 설계" 참고)
--------------------------------------------------------------------------
1. 시간 축: CAPA/계획 모두 [RULE_TIMEKEY, RULE_TIMEKEY + window_minutes) 창 기준.
   window_minutes 기본값은 시뮬레이터가 이미 쓰는 CONFIG.env.soft_cutoff_minutes
   (RULE_TIMEKEY 07:00 기준 익일 05:00 = 1320분)와 동일 상수를 재사용한다 —
   simulator._achievable_qty/_carrier_takt_minutes가 D0_PLAN_QTY를 이 창의
   목표량으로 그대로 쓰는 기존 관례를 그대로 따른다(별도 비례배분 없음).
2. CAPA_PER_EQP = window_minutes / ST  (ST=분/장, UPH=60/ST 이므로
   UPH × (window_minutes/60) 과 동일한 값).
3. TARGET_QTY(ppk,oper) = min(D0_PLAN_QTY, WIP_QTY)  — WIP_QTY는 discrete_arrange에
   선언된 해당 (PPK,OPER)의 distinct carrier WF_QTY 합("전체 재공").
4. PPK 우선순위(PLAN_PRIORITY, 작을수록 우선) 오름차순으로 처리한다.
5. 전용 모델(abstract_arrange 상 해당 (PPK,OPER)에 모델이 1개뿐인 경우) 수요를
   공용 모델 수요보다 먼저 배정한다 — 대체 불가능한 수요이므로.
6. 공용 모델(PPK,OPER에 모델이 2개 이상 가능)은 같은 PPK 안에서 OPER별 목표 수량을
   균등 배분한다(공정별 CAPA 편차 방지, 마지막 OPER 달성을 위해 상류 공정도 꾸준히
   생산). 단, 특정 OPER의 WIP가 나머지 대비 과도하게 몰려 있으면(기본 1.5배 이상)
   그 OPER은 균등 상한 없이 목표 수량 전체를 배정 대상으로 푼다(정체 해소 예외).
   모델 선택은 "지금까지 전체 보유 대수 대비 가장 적게 쓰인 모델"부터 우선해
   같은 모델로 수요가 쏠리지 않도록 분산한다.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config import CONFIG, RULE_TIMEKEY_FMT, normalize_rule_timekey

# WIP가 (PPK 안의) 균등 배분 기준 대비 이 배수 이상 한 OPER에 몰려 있으면
# 그 OPER은 균등 상한 없이 목표 수량 전체를 배정 후보로 푼다(점 7의 예외 케이스).
DEFAULT_WIP_SKEW_RELEASE_FACTOR = 1.5

PpkOper = Tuple[str, str]


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


def _build_model_total_eqp(discrete_raw: List[dict]) -> Dict[str, int]:
    """EQP_MODEL_CD → 전체 보유 대수(discrete_arrange의 distinct EQP_ID 기준)."""
    eqp_by_model: Dict[str, set] = {}
    for r in discrete_raw:
        model = r.get("EQP_MODEL_CD")
        if not model:
            continue
        eqp_by_model.setdefault(str(model), set()).add(r["EQP_ID"])
    return {model: len(eqps) for model, eqps in eqp_by_model.items()}


def _build_wip_qty(discrete_raw: List[dict]) -> Dict[PpkOper, int]:
    """(PPK,OPER) → 전체 재공 수량. discrete_arrange는 (EQP,LOT) M:N 후보
    선언이므로(같은 carrier가 여러 EQP 후보 행에 중복 등장), carrier 단위로
    한 번만 센다(preprocess._carrier_instance_id와 동일한 dedup 원칙)."""
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
    wip_skew_release_factor: Optional[float] = None,
) -> List[dict]:
    """
    목적: PPK/OPER × EQP_MODEL_CD 할당 대수 산출 (DB RTS_EQPALLOC_PLAN 행 포맷)
    Input:
        raw: {"abstract_arrange": [...], "discrete_arrange": [...], "plan": [...], "flow": [...]}
             (모두 SQL 원본 컬럼명 그대로 — data/loader/preprocess.preprocess()와 동일 입력)
        fac_id, rule_timekey: 이번 회차 식별자
        window_minutes: 계획/CAPA 산정 창 길이(분). 기본 CONFIG.env.soft_cutoff_minutes.
        wip_skew_release_factor: OPER 균등 배분 예외 임계값 배수. 기본 1.5.
    Output:
        List[dict] — RTS_EQPALLOC_PLAN 컬럼과 1:1 대응하는 행 목록(정렬됨:
        PLAN_PROD_ATTR_VAL, OPER_ID, EQP_MODEL_CD 순)
    """
    window_minutes = window_minutes or CONFIG.env.soft_cutoff_minutes
    skew_factor = (
        wip_skew_release_factor
        if wip_skew_release_factor is not None
        else DEFAULT_WIP_SKEW_RELEASE_FACTOR
    )
    rule_timekey = normalize_rule_timekey(rule_timekey)
    window_start_dt = datetime.strptime(rule_timekey, RULE_TIMEKEY_FMT)
    window_end_dt = window_start_dt + timedelta(minutes=window_minutes)

    abstract_raw = raw.get("abstract_arrange", [])
    discrete_raw = raw.get("discrete_arrange", [])
    plan_raw = raw.get("plan", [])
    flow_raw = raw.get("flow", [])

    eligible_models = _build_eligible_models(abstract_raw)
    model_total_eqp = _build_model_total_eqp(discrete_raw)
    wip_qty = _build_wip_qty(discrete_raw)
    d0_plan_qty, ppk_priority = _build_plan_meta(plan_raw)
    flow_seq = _build_flow_seq(flow_raw)

    # (ppk,oper) 목표 수량 = min(계획, 전체 재공). 계획/재공 정보가 없으면 대상에서
    # 제외한다(=이 (ppk,oper)에 대해서는 할당 행을 만들지 않고, 마스킹 단계에서
    # "제약 없음"으로 폴백하게 한다 — 일시적 계획 누락으로 정당한 WIP 배정까지
    # 막는 부작용을 피한다).
    demand: Dict[PpkOper, int] = {}
    for key, models in eligible_models.items():
        if not models:
            continue
        target = min(d0_plan_qty.get(key, 0), wip_qty.get(key, 0))
        if target > 0:
            demand[key] = target

    exclusive_keys = [k for k in demand if len(eligible_models[k]) == 1]
    shared_keys = [k for k in demand if len(eligible_models[k]) > 1]

    def _sort_key(k: PpkOper):
        ppk, oper = k
        return (ppk_priority.get(ppk, 1), ppk, flow_seq.get(ppk, {}).get(oper, 0), oper)

    exclusive_keys.sort(key=_sort_key)

    remaining_units: Dict[str, int] = dict(model_total_eqp)
    model_usage: Dict[str, int] = {m: 0 for m in model_total_eqp}

    rows: List[dict] = []

    def _append_row(ppk: str, oper: str, model: str, st: float, alloc_cnt: int,
                     target_qty: float, is_exclusive: bool) -> None:
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
            "WIP_QTY": wip_qty.get((ppk, oper), 0),
            "TARGET_QTY": target_qty,
            "ST": st,
            "CAPA_PER_EQP": round(capa, 4),
            "ALLOC_EQP_CNT": alloc_cnt,
            "ALLOC_CAPA_QTY": round(alloc_cnt * capa, 4),
            "TOTAL_EQP_CNT": model_total_eqp.get(model, 0),
        })

    # ── Phase A: 전용 모델(대체 불가) 수요를 PPK 우선순위 순으로 먼저 배정 ──────
    for key in exclusive_keys:
        ppk, oper = key
        model, st = next(iter(eligible_models[key].items()))
        capa = _capa_per_eqp(st, window_minutes)
        target = demand[key]
        need = _units_needed(target, capa)
        take = min(need, remaining_units.get(model, 0))
        if take <= 0:
            continue
        remaining_units[model] = remaining_units.get(model, 0) - take
        model_usage[model] = model_usage.get(model, 0) + take
        _append_row(ppk, oper, model, st, take, target, is_exclusive=True)

    # ── Phase B: 공용 모델 수요 — PPK 우선순위 순, PPK 내부는 OPER 균등 배분 ────
    # 공용모델 후보를 "지금까지 전체 보유 대수 대비 가장 적게 쓰인 모델"부터
    # 우선 고르는 정렬 규칙(점 6). Phase A/B가 공유하는 remaining_units/model_usage를
    # 참조하므로 클로저로 둔다.
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
            if remaining_units.get(model, 0) <= 0:
                continue
            remaining_units[model] -= 1
            model_usage[model] = model_usage.get(model, 0) + 1
            _append_row(ppk_, oper, model, st, 1, demand[key], is_exclusive=False)
            return _capa_per_eqp(st, window_minutes)
        return 0.0

    shared_by_ppk: Dict[str, List[PpkOper]] = {}
    for key in shared_keys:
        shared_by_ppk.setdefault(key[0], []).append(key)

    for ppk in sorted(shared_by_ppk, key=lambda p: (ppk_priority.get(p, 1), p)):
        opers = sorted(shared_by_ppk[ppk], key=lambda k: (flow_seq.get(ppk, {}).get(k[1], 0), k[1]))
        n = len(opers)
        total_wip = sum(wip_qty.get(k, 0) for k in opers) or 1

        # WIP가 균등 배분 기준(1/n) 대비 skew_factor 배 이상 몰린 OPER은
        # "풀어줘야 할" 예외로 보고 균등 상한 없이 먼저(그리디로) 채운다.
        released = [k for k in opers if (wip_qty.get(k, 0) / total_wip) >= skew_factor / n]
        capped = [k for k in opers if k not in released]

        # 서브페이즈 1: 몰린 OPER부터 그리디로 자기 목표량까지 채운다(대체 모델
        # 있는 채로도 정체 해소가 우선이므로 균등 배분 대상에서 뺀다).
        for key in released:
            qty_covered = 0.0
            while qty_covered < demand[key]:
                gained = _take_one_unit(key)
                if gained <= 0:
                    break
                qty_covered += gained

        # 서브페이즈 2: 나머지 OPER은 남은 풀을 라운드로빈(1대씩 순환 배정)으로
        # 균등하게 나눠 가진다 — 특정 OPER이 먼저 처리된다고 물량을 독식하지
        # 않도록 함(점 7: 공정별 CAPA 편차 방지).
        if capped:
            equal_share = sum(demand[k] for k in capped) / len(capped)
            remaining_qty = {k: min(demand[k], equal_share) for k in capped}
            progressed = True
            while progressed:
                progressed = False
                for key in capped:
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
