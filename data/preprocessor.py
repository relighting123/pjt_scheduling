"""
data/preprocessor.py – 원시 JSON 데이터 → RL 환경 입력 변환
loader.py로 불러온 딕셔너리를 시뮬레이터와 RL 환경이 바로 쓸 수 있는
구조화된 딕셔너리로 가공합니다.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.helpers import parse_datetime, datetime_to_minutes, build_index_map


def _coerce_proc_time(value) -> Optional[int]:
    """availability/schedule ST가 숫자(소요시간 분)이면 int로 반환"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _legacy_st_as_eqp_model(value) -> Optional[str]:
    """구 샘플: ST가 'A' 등 문자열이면 장비 MODEL로 해석"""
    if value is None:
        return None
    if isinstance(value, str) and not value.isdigit():
        return value
    return None


def preprocess(raw: Dict[str, List[dict]]) -> dict:
    """
    목적: 원시 입력 4종을 RL 환경·시뮬레이터가 사용하는 공통 데이터 구조로 변환
    Input:
        raw = {
            "schedule":     [{EQP_ID, LOT_ID, CARRIER_ID, PLAN_PROD_KEY, ST, SEQ, STARTTM, ENDTM}, ...],
            "availability": [{EQP_ID, LOT_ID, PLAN_PROD_KEY, ST, EQP_MODEL, WF_QTY}, ...],
            "plan":         [{PLAN_PROD_KEY, OPER_ID, D0_PLAN_QTY, D1_PLAN_QTY, PLAN_PRIORITY}, ...],
            "flow":         [{PLAN_PROD_KEY, SEQ_ID, OPER_ID}, ...]
        }
    Output:
        {
            "sim_base_time": datetime,       # 시뮬레이션 기준 시각
            "sim_end_minutes": int,          # 시뮬레이션 종료 (분)
            "eqp_ids": [str],                # 정렬된 EQP 목록
            "oper_ids": [str],               # 정렬된 OPER 목록
            "prod_keys": [str],              # 정렬된 제품 목록
            "oper_idx": {oper: int},         # OPER → 인덱스
            "prod_idx": {prod: int},         # PROD → 인덱스
            "lots": [LotDict],               # LOT 세부 정보
            "eqp_lot_map": {eqp: [lot_id]}, # EQP별 배정 가능 LOT
            "plan": [PlanDict],              # 계획 데이터
            "initial_schedule": [SchedDict], # 원본 스케줄 (비교용)
            "flow": {prod: [{seq, oper}]},   # 제품별 FLOW 순서
        }
    """
    sched_raw = raw["schedule"]
    avail_raw = raw["availability"]
    plan_raw  = raw["plan"]
    flow_raw  = raw["flow"]

    # ── 기준 시각 ─────────────────────────────────────────────────────────────
    all_start_times = [parse_datetime(r["STARTTM"]) for r in sched_raw]
    base_time = min(all_start_times)
    all_end_times   = [parse_datetime(r["ENDTM"])   for r in sched_raw]
    sim_end_minutes = max(datetime_to_minutes(t, base_time) for t in all_end_times)

    # ── 범주형 목록 & 인덱스 맵 ──────────────────────────────────────────────
    eqp_ids  = sorted({r["EQP_ID"]        for r in sched_raw})
    oper_ids = sorted({r["OPER_ID"]        for r in flow_raw})
    prod_keys= sorted({r["PLAN_PROD_KEY"]  for r in sched_raw})

    oper_idx = build_index_map(oper_ids)
    prod_idx = build_index_map(prod_keys)

    # ── 초기 스케줄: 처리 시간 및 OPER 파생 ──────────────────────────────────
    # flow로 SEQ → OPER 매핑 구성
    flow_map: Dict[str, Dict[int, str]] = {}   # {plan_prod_key: {seq: oper_id}}
    for r in flow_raw:
        ppk = r["PLAN_PROD_KEY"]
        if ppk not in flow_map:
            flow_map[ppk] = {}
        flow_map[ppk][int(r["SEQ_ID"])] = r["OPER_ID"]

    # availability로 EQP별 LOT 및 WF_QTY 조회
    avail_map: Dict[Tuple[str, str], int] = {}   # {(eqp_id, lot_id): wf_qty}
    eqp_lot_map: Dict[str, List[str]] = {}
    for r in avail_raw:
        key = (r["EQP_ID"], r["LOT_ID"])
        avail_map[key] = int(r["WF_QTY"])
        eqp_lot_map.setdefault(r["EQP_ID"], [])
        if r["LOT_ID"] not in eqp_lot_map[r["EQP_ID"]]:
            eqp_lot_map[r["EQP_ID"]].append(r["LOT_ID"])

    # LOT 정보 빌드
    lot_info: Dict[str, dict] = {}
    for r in sched_raw:
        lot_id = r["LOT_ID"]
        start  = datetime_to_minutes(parse_datetime(r["STARTTM"]), base_time)
        end    = datetime_to_minutes(parse_datetime(r["ENDTM"]),   base_time)
        proc_time = max(end - start, 1)

        ppk = r["PLAN_PROD_KEY"]
        seq = int(r["SEQ"])
        oper_id = flow_map.get(ppk, {}).get(seq, "OPER001")

        # availability에서 WF_QTY 가져오기 (기본 25)
        wf_qty = 25
        for eqp_id in eqp_ids:
            if (eqp_id, lot_id) in avail_map:
                wf_qty = avail_map[(eqp_id, lot_id)]
                break

        # 계획 우선순위
        priority = 1
        for p in plan_raw:
            if p["PLAN_PROD_KEY"] == ppk and p["OPER_ID"] == oper_id:
                priority = int(p.get("PLAN_PRIORITY", 1))
                break

        lot_info[lot_id] = {
            "lot_id":        lot_id,
            "carrier_id":    r["CARRIER_ID"],
            "plan_prod_key": ppk,
            "oper_id":       oper_id,
            "seq":           seq,
            "wf_qty":        wf_qty,
            "processing_time": proc_time,
            "priority":      priority,
            "original_eqp":  r["EQP_ID"],
        }

    # 초기 스케줄 레코드 (비교용)
    initial_schedule = []
    for r in sched_raw:
        lot_id = r["LOT_ID"]
        start  = datetime_to_minutes(parse_datetime(r["STARTTM"]), base_time)
        end    = datetime_to_minutes(parse_datetime(r["ENDTM"]),   base_time)
        ppk    = r["PLAN_PROD_KEY"]
        seq    = int(r["SEQ"])
        oper_id = flow_map.get(ppk, {}).get(seq, "OPER001")
        initial_schedule.append({
            "EQP_ID":        r["EQP_ID"],
            "LOT_ID":        lot_id,
            "CARRIER_ID":    r["CARRIER_ID"],
            "PLAN_PROD_KEY": ppk,
            "OPER_ID":       oper_id,
            "ST":            r["ST"],
            "SEQ":           seq,
            "START_TM":      start,
            "END_TM":        end,
        })

    # 계획 데이터 정리
    plan_list = []
    for p in plan_raw:
        plan_list.append({
            "plan_prod_key": p["PLAN_PROD_KEY"],
            "oper_id":       p["OPER_ID"],
            "d0_plan_qty":   int(p["D0_PLAN_QTY"]),
            "d1_plan_qty":   int(p["D1_PLAN_QTY"]),
            "priority":      int(p.get("PLAN_PRIORITY", 1)),
        })

    # FLOW 데이터 정리
    flow_list: Dict[str, List[dict]] = {}
    for r in flow_raw:
        ppk = r["PLAN_PROD_KEY"]
        flow_list.setdefault(ppk, [])
        flow_list[ppk].append({"seq_id": int(r["SEQ_ID"]), "oper_id": r["OPER_ID"]})
    for ppk in flow_list:
        flow_list[ppk].sort(key=lambda x: x["seq_id"])

    # ── (LOT, EQP) 조합별 처리시간 행렬 ────────────────────────────────────────
    # 처리시간은 재공(LOT) 단독이 아닌 WIP×EQP 조합에 의해 결정됨.
    # schedule에 기록된 원래 배정에서 처리시간을 추출하고,
    # availability에 있는 대체 조합(다른 EQP에서 같은 LOT)은 동일 OPER의
    # 해당 EQP 평균 처리시간으로 추정합니다.
    #
    # proc_time_matrix[(lot_id, eqp_id)] → 처리시간(분)
    proc_time_matrix: Dict[Tuple[str, str], int] = {}

    # Step 1: 원본 스케줄에서 직접 도출 (가장 신뢰할 수 있는 데이터)
    for r in sched_raw:
        lid = r["LOT_ID"]
        eid = r["EQP_ID"]
        start = datetime_to_minutes(parse_datetime(r["STARTTM"]), base_time)
        end   = datetime_to_minutes(parse_datetime(r["ENDTM"]),   base_time)
        proc_time_matrix[(lid, eid)] = max(end - start, 1)

    # Step 2: (EQP, OPER)별 평균 처리시간 계산 → 대체 조합 추정용
    eqp_oper_avg: Dict[Tuple[str, str], list] = {}  # {(eqp_id, oper_id): [times]}
    for (lid, eid), pt in proc_time_matrix.items():
        oper_id = lot_info.get(lid, {}).get("oper_id", "")
        eqp_oper_avg.setdefault((eid, oper_id), []).append(pt)
    eqp_oper_avg_val: Dict[Tuple[str, str], int] = {
        k: int(sum(v) / len(v)) for k, v in eqp_oper_avg.items()
    }

    # Step 3: availability – ST(소요시간) 및 schedule 미기록 조합 보완
    for r in avail_raw:
        lid = r["LOT_ID"]
        eid = r["EQP_ID"]
        pt = _coerce_proc_time(r.get("ST"))
        if pt is not None:
            proc_time_matrix[(lid, eid)] = pt
        elif (lid, eid) not in proc_time_matrix:
            oper_id = lot_info.get(lid, {}).get("oper_id", "")
            fallback = eqp_oper_avg_val.get(
                (eid, oper_id),
                lot_info.get(lid, {}).get("processing_time", 60),
            )
            proc_time_matrix[(lid, eid)] = fallback

    # ── EQP → 처리 가능 OPER 집합 ────────────────────────────────────────────
    eqp_oper_cap: Dict[str, List[str]] = {}
    for r in avail_raw:
        eid = r["EQP_ID"]
        lid = r["LOT_ID"]
        if lid in lot_info:
            oper_id = lot_info[lid]["oper_id"]
            eqp_oper_cap.setdefault(eid, [])
            if oper_id not in eqp_oper_cap[eid]:
                eqp_oper_cap[eid].append(oper_id)

    # ── EQP 인덱스 맵 (Group E 관측 행렬 열 인덱싱용) ─────────────────────────
    eqp_idx = {eid: i for i, eid in enumerate(eqp_ids)}

    # 집계 정규화 스케일 팩터
    max_proc_time = max((v for v in proc_time_matrix.values()), default=1)
    max_wf_qty    = max((v["wf_qty"] for v in lot_info.values()), default=1)

    # ── EQP 장비 MODEL (availability EQP_MODEL, 구형 ST 문자열 호환) ─────────
    eqp_model_map: Dict[str, str] = {}
    for r in avail_raw:
        eid = r["EQP_ID"]
        if r.get("EQP_MODEL"):
            eqp_model_map[eid] = str(r["EQP_MODEL"])
    for r in avail_raw:
        eid = r["EQP_ID"]
        if eid in eqp_model_map:
            continue
        legacy = _legacy_st_as_eqp_model(r.get("ST"))
        eqp_model_map[eid] = legacy if legacy else "A"

    # ── FLOW: seq → 다음 공정 (DES 유입용) ───────────────────────────────────
    flow_next: Dict[str, Dict[int, dict]] = {}
    for ppk, steps in flow_list.items():
        ordered = sorted(steps, key=lambda x: x["seq_id"])
        for i, step in enumerate(ordered):
            if i + 1 < len(ordered):
                nxt = ordered[i + 1]
                flow_next.setdefault(ppk, {})[step["seq_id"]] = {
                    "next_seq":  nxt["seq_id"],
                    "next_oper": nxt["oper_id"],
                }

    # OPER별 처리 가능 장비 MODEL(ST) 및 추상 처리시간
    oper_eqp_models: Dict[str, List[str]] = {}
    abstract_proc_time: Dict[Tuple[str, str], list] = {}
    for eid, opers in eqp_oper_cap.items():
        model = eqp_model_map.get(eid, "A")
        for oper_id in opers:
            oper_eqp_models.setdefault(oper_id, [])
            if model not in oper_eqp_models[oper_id]:
                oper_eqp_models[oper_id].append(model)
    for (lid, eid), pt in proc_time_matrix.items():
        if lid not in lot_info:
            continue
        oper_id = lot_info[lid]["oper_id"]
        model = eqp_model_map.get(eid, "A")
        abstract_proc_time.setdefault((oper_id, model), []).append(pt)
    abstract_proc_time_val: Dict[Tuple[str, str], int] = {
        k: int(sum(v) / len(v)) for k, v in abstract_proc_time.items()
    }

    plan_meta: Dict[Tuple[str, str], dict] = {}
    for p in plan_list:
        plan_meta[(p["plan_prod_key"], p["oper_id"])] = {
            "priority":    p["priority"],
            "d0_plan_qty": p["d0_plan_qty"],
        }

    lot_inject_deadline: Dict[str, int] = {}
    lot_initial_start: Dict[str, int] = {}
    for r in initial_schedule:
        lot_inject_deadline[r["LOT_ID"]] = r["START_TM"]
        lot_initial_start[r["LOT_ID"]] = r["START_TM"]

    # ── arrange 테이블 (availability 전체 – 초기 Actual 재공) ────────────────
    arrange_actual_table: List[dict] = []
    for r in avail_raw:
        lot_id = r["LOT_ID"]
        eid = r["EQP_ID"]
        proc_time = proc_time_matrix.get((lot_id, eid), 60)
        row_model = (
            str(r["EQP_MODEL"]) if r.get("EQP_MODEL")
            else _legacy_st_as_eqp_model(r.get("ST"))
            or eqp_model_map.get(eid, "A")
        )
        arrange_actual_table.append({
            "eqp_id":           eid,
            "lot_id":           lot_id,
            "plan_prod_key":    r["PLAN_PROD_KEY"],
            "st":               proc_time,
            "proc_time":        proc_time,
            "eqp_model":        row_model,
            "initial_start_tm": lot_initial_start.get(lot_id),
            "wf_qty":           int(r["WF_QTY"]),
        })

    return {
        "sim_base_time":    base_time,
        "sim_end_minutes":  sim_end_minutes,
        "eqp_ids":          eqp_ids,
        "oper_ids":         oper_ids,
        "prod_keys":        prod_keys,
        "oper_idx":         oper_idx,
        "prod_idx":         prod_idx,
        "lots":             list(lot_info.values()),
        "eqp_lot_map":      eqp_lot_map,
        "proc_time_matrix": proc_time_matrix,
        "eqp_oper_avg":     eqp_oper_avg_val,
        "eqp_oper_cap":     eqp_oper_cap,
        "eqp_idx":          eqp_idx,
        "eqp_model_map":    eqp_model_map,
        "flow_next":        flow_next,
        "oper_eqp_models":  oper_eqp_models,
        "abstract_proc_time": abstract_proc_time_val,
        "plan_meta":        plan_meta,
        "max_proc_time":    max_proc_time,
        "max_wf_qty":       max_wf_qty,
        "plan":             plan_list,
        "initial_schedule": initial_schedule,
        "flow":             flow_list,
        "arrange_actual_table": arrange_actual_table,
        "arrange_table":    arrange_actual_table,
        "lot_inject_deadline": lot_inject_deadline,
    }
