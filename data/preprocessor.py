"""
data/preprocessor.py – 원시 JSON 데이터 → RL 환경 입력 변환
loader.py로 불러온 딕셔너리를 시뮬레이터와 RL 환경이 바로 쓸 수 있는
구조화된 딕셔너리로 가공합니다.
"""
from datetime import datetime
from typing import Dict, List, Tuple

from utils.helpers import parse_datetime, datetime_to_minutes, build_index_map


def preprocess(raw: Dict[str, List[dict]]) -> dict:
    """
    목적: 원시 입력 4종을 RL 환경·시뮬레이터가 사용하는 공통 데이터 구조로 변환
    Input:
        raw = {
            "schedule":     [{EQP_ID, LOT_ID, CARRIER_ID, PLAN_PROD_KEY, ST, SEQ, STARTTM, ENDTM}, ...],
            "availability": [{EQP_ID, LOT_ID, PLAN_PROD_KEY, ST, WF_QTY}, ...],
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
        "plan":             plan_list,
        "initial_schedule": initial_schedule,
        "flow":             flow_list,
    }
