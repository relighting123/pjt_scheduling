"""
data/preprocessor.py – 원시 JSON 데이터 → RL 환경 입력 변환
loader.py로 불러온 딕셔너리를 시뮬레이터와 RL 환경이 바로 쓸 수 있는
구조화된 딕셔너리로 가공합니다.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import CONFIG, normalize_rule_timekey, RULE_TIMEKEY_FMT, PERIOD_SPLITS
from utils.helpers import parse_datetime, datetime_to_minutes, build_index_map, split_wf_qty


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


def _build_split_lookup(split_raw: List[dict]) -> Dict[Tuple[str, str], int]:
    lookup: Dict[Tuple[str, str], int] = {}
    for r in split_raw:
        ppk = r["PLAN_PROD_KEY"]
        model = str(r.get("EQP_MODEL", "*")).strip().upper() or "*"
        lookup[(ppk, model)] = int(r["SPLIT_QTY"])
    return lookup


def _resolve_split_qty(
    ppk: str,
    eqp_model: str,
    split_lookup: Dict[Tuple[str, str], int],
) -> Optional[int]:
    model = (eqp_model or "A").strip().upper()
    if (ppk, model) in split_lookup:
        return split_lookup[(ppk, model)]
    if (ppk, "*") in split_lookup:
        return split_lookup[(ppk, "*")]
    return None


def _child_lot_id(parent_id: str, index: int) -> str:
    return f"{parent_id}__S{index:02d}"


def _resolve_sim_base_time(sched_raw: List[dict], period_key: Optional[str]) -> datetime:
    """RULE_TIMEKEY 07:00 기준 시뮬 시작 시각."""
    if period_key:
        return datetime.strptime(normalize_rule_timekey(period_key), RULE_TIMEKEY_FMT)
    if CONFIG.path.train_snapshot and CONFIG.path.dataset_split in PERIOD_SPLITS:
        return datetime.strptime(
            normalize_rule_timekey(CONFIG.path.train_snapshot), RULE_TIMEKEY_FMT,
        )
    min_start = min(parse_datetime(r["STARTTM"]) for r in sched_raw)
    return min_start.replace(hour=7, minute=0, second=0, microsecond=0)


def _default_lot_cd(lot_id: str, plan_prod_key: str) -> str:
    suffix = plan_prod_key.replace("PPK", "") if "PPK" in plan_prod_key else lot_id[-3:]
    return f"LC{suffix.zfill(2)[-2:]}"


def _default_temp(plan_prod_key: str) -> str:
    n = sum(ord(c) for c in plan_prod_key)
    return "T650" if n % 2 == 0 else "T700"


def _build_lot_attributes(
    lot_ids: List[str],
    lot_info: Dict[str, dict],
    lot_master_raw: List[dict],
) -> Tuple[Dict[str, dict], List[str], List[str]]:
    """LOT_ID → {lot_cd, temp} 및 범주 목록."""
    master = {r["LOT_ID"]: r for r in lot_master_raw}
    lot_cd_set: set = set()
    temp_set: set = set()
    attrs: Dict[str, dict] = {}
    for lid in lot_ids:
        if lid in master:
            lot_cd = str(master[lid]["LOT_CD"])
            temp = str(master[lid]["TEMP"])
        elif lid in lot_info:
            ppk = lot_info[lid]["plan_prod_key"]
            lot_cd = _default_lot_cd(lid, ppk)
            temp = _default_temp(ppk)
        else:
            lot_cd = _default_lot_cd(lid, "PPK001")
            temp = "T650"
        attrs[lid] = {"lot_cd": lot_cd, "temp": temp}
        lot_cd_set.add(lot_cd)
        temp_set.add(temp)
        if lid in lot_info:
            lot_info[lid]["lot_cd"] = lot_cd
            lot_info[lid]["temp"] = temp
    return attrs, sorted(lot_cd_set), sorted(temp_set)


def _build_tool_capacity_map(
    tool_raw: List[dict],
    lot_cds: List[str],
    eqp_models: List[str],
) -> Dict[Tuple[str, str], int]:
    cap: Dict[Tuple[str, str], int] = {}
    for r in tool_raw:
        cap[(str(r["LOT_CD"]), str(r["EQP_MODEL"]))] = int(r["MAX_TOOL"])
    if cap:
        return cap
    for lc in lot_cds:
        for model in eqp_models:
            cap[(lc, model)] = 2
    return cap


def _apply_wafer_lot_split(
    lot_info: Dict[str, dict],
    eqp_lot_map: Dict[str, List[str]],
    proc_time_matrix: Dict[Tuple[str, str], int],
    initial_schedule: List[dict],
    discrete_raw: List[dict],
    eqp_model_map: Dict[str, str],
    split_lookup: Dict[Tuple[str, str], int],
    lot_inject_deadline: Dict[str, int],
    lot_initial_start: Dict[str, int],
) -> None:
    """PPK×MODEL SPLIT_QTY 규칙에 따라 LOT을 wafer sub-lot으로 분할"""
    if not split_lookup:
        return

    split_children: Dict[str, List[Tuple[str, int]]] = {}

    for parent_id in list(lot_info.keys()):
        info = lot_info[parent_id]
        wf = int(info["wf_qty"])
        model = eqp_model_map.get(info["original_eqp"], "A")
        split_qty = _resolve_split_qty(info["plan_prod_key"], model, split_lookup)
        if split_qty is None or split_qty <= 0 or wf <= split_qty:
            continue

        sizes = split_wf_qty(wf, split_qty)
        if len(sizes) <= 1:
            continue

        child_ids = [_child_lot_id(parent_id, i) for i in range(1, len(sizes) + 1)]
        split_children[parent_id] = list(zip(child_ids, sizes))
        del lot_info[parent_id]

        for cid, qty in zip(child_ids, sizes):
            lot_info[cid] = {
                **info,
                "lot_id": cid,
                "wf_qty": qty,
                "parent_lot_id": parent_id,
            }

        for eid, lots in eqp_lot_map.items():
            if parent_id not in lots:
                continue
            idx = lots.index(parent_id)
            lots[idx:idx + 1] = child_ids

        for (lid, eid), pt in list(proc_time_matrix.items()):
            if lid != parent_id:
                continue
            del proc_time_matrix[(lid, eid)]
            for cid in child_ids:
                proc_time_matrix[(cid, eid)] = pt

        for rec in initial_schedule:
            if rec["LOT_ID"] == parent_id:
                rec["LOT_ID"] = child_ids[0]

        deadline = lot_inject_deadline.pop(parent_id, None)
        start_tm = lot_initial_start.pop(parent_id, None)
        for cid in child_ids:
            if deadline is not None:
                lot_inject_deadline[cid] = deadline
            if start_tm is not None:
                lot_initial_start[cid] = start_tm

    expanded_avail: List[dict] = []
    for r in discrete_raw:
        lot_id = r["LOT_ID"]
        if lot_id in split_children:
            for cid, qty in split_children[lot_id]:
                row = dict(r)
                row["LOT_ID"] = cid
                row["WF_QTY"] = qty
                expanded_avail.append(row)
            continue
        expanded_avail.append(dict(r))

    discrete_raw.clear()
    discrete_raw.extend(expanded_avail)


def _build_abstract_route_maps(
    abstract_raw: List[dict],
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str], List[Tuple[str, int]]]]:
    """abstract_arrange.json → route lookup."""
    route_map: Dict[Tuple[str, str, str], int] = {}
    by_ppk_oper: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for r in abstract_raw:
        ppk = r["PLAN_PROD_KEY"]
        oper_id = r["OPER_ID"]
        model = str(r.get("EQP_MODEL") or "A")
        st = _coerce_proc_time(r.get("ST")) or 60
        route_map[(ppk, oper_id, model)] = st
        by_ppk_oper.setdefault((ppk, oper_id), []).append((model, st))
    return route_map, by_ppk_oper


def _rebuild_eqp_oper_cap(
    discrete_raw: List[dict],
    lot_info: Dict[str, dict],
) -> Dict[str, List[str]]:
    cap: Dict[str, List[str]] = {}
    for r in discrete_raw:
        lid = r["LOT_ID"]
        if lid not in lot_info:
            continue
        eid = r["EQP_ID"]
        oper_id = lot_info[lid]["oper_id"]
        cap.setdefault(eid, [])
        if oper_id not in cap[eid]:
            cap[eid].append(oper_id)
    return cap


def preprocess(raw: Dict[str, List[dict]], period_key: Optional[str] = None) -> dict:
    """
    목적: 원시 입력 4종을 RL 환경·시뮬레이터가 사용하는 공통 데이터 구조로 변환
    Input:
        raw = {
            "schedule":     [{EQP_ID, LOT_ID, CARRIER_ID, PLAN_PROD_KEY, ST, SEQ, STARTTM, ENDTM}, ...],
            "discrete_arrange": [{EQP_ID, LOT_ID, PLAN_PROD_KEY, ST, EQP_MODEL, WF_QTY}, ...],
            "abstract_arrange": [{PLAN_PROD_KEY, OPER_ID, EQP_MODEL, ST}, ...],
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
    discrete_raw = raw["discrete_arrange"]
    abstract_raw = raw.get("abstract_arrange", [])
    plan_raw  = raw["plan"]
    flow_raw  = raw["flow"]
    split_raw = raw.get("split", [])
    lot_master_raw = raw.get("lot_master", [])
    tool_capacity_raw = raw.get("tool_capacity", [])

    # ── 기준 시각 (RULE_TIMEKEY 07:00) ───────────────────────────────────────
    base_time = _resolve_sim_base_time(sched_raw, period_key)
    sim_end_minutes = CONFIG.env.hard_horizon_minutes
    soft_cutoff_minutes = CONFIG.env.soft_cutoff_minutes

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

    # discrete_arrange로 EQP별 LOT 및 WF_QTY 조회
    avail_map: Dict[Tuple[str, str], int] = {}   # {(eqp_id, lot_id): wf_qty}
    eqp_lot_map: Dict[str, List[str]] = {}
    for r in discrete_raw:
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

        # discrete_arrange에서 WF_QTY 가져오기 (기본 25)
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
    # discrete_arrange에 있는 대체 조합(다른 EQP에서 같은 LOT)은 동일 OPER의
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

    # Step 3: discrete_arrange – ST(소요시간) 및 schedule 미기록 조합 보완
    for r in discrete_raw:
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
    for r in discrete_raw:
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

    # ── EQP 장비 MODEL (availability EQP_MODEL, 구형 ST 문자열 호환) ─────────
    eqp_model_map: Dict[str, str] = {}
    for r in discrete_raw:
        eid = r["EQP_ID"]
        if r.get("EQP_MODEL"):
            eqp_model_map[eid] = str(r["EQP_MODEL"])
    for r in discrete_raw:
        eid = r["EQP_ID"]
        if eid in eqp_model_map:
            continue
        legacy = _legacy_st_as_eqp_model(r.get("ST"))
        eqp_model_map[eid] = legacy if legacy else "A"

    abstract_route_map, abstract_routes_by_ppk_oper = _build_abstract_route_maps(abstract_raw)

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

    split_lookup = _build_split_lookup(split_raw)
    _apply_wafer_lot_split(
        lot_info,
        eqp_lot_map,
        proc_time_matrix,
        initial_schedule,
        discrete_raw,
        eqp_model_map,
        split_lookup,
        lot_inject_deadline,
        lot_initial_start,
    )
    eqp_oper_cap = _rebuild_eqp_oper_cap(discrete_raw, lot_info)

    eqp_idx = build_index_map(eqp_ids)

    lot_ids = sorted(lot_info.keys())
    lot_attrs, lot_cd_ids, temp_ids = _build_lot_attributes(
        lot_ids, lot_info, lot_master_raw,
    )
    lot_cd_idx = build_index_map(lot_cd_ids)
    temp_idx = build_index_map(temp_ids)
    eqp_models = sorted(set(eqp_model_map.values()))
    tool_capacity = _build_tool_capacity_map(tool_capacity_raw, lot_cd_ids, eqp_models)

    # ── arrange 테이블 (availability 전체 – 초기 Actual 재공) ────────────────
    arrange_actual_table: List[dict] = []
    for r in discrete_raw:
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

    max_wf_qty = max((v["wf_qty"] for v in lot_info.values()), default=1)

    return {
        "sim_base_time":    base_time,
        "sim_end_minutes":  sim_end_minutes,
        "soft_cutoff_minutes": soft_cutoff_minutes,
        "conversion_minutes": CONFIG.env.conversion_minutes,
        "eqp_ids":          eqp_ids,
        "oper_ids":         oper_ids,
        "prod_keys":        prod_keys,
        "oper_idx":         oper_idx,
        "prod_idx":         prod_idx,
        "eqp_idx":          eqp_idx,
        "lot_cd_ids":       lot_cd_ids,
        "temp_ids":         temp_ids,
        "lot_cd_idx":       lot_cd_idx,
        "temp_idx":         temp_idx,
        "lot_attrs":        lot_attrs,
        "tool_capacity":    tool_capacity,
        "lots":             list(lot_info.values()),
        "eqp_lot_map":      eqp_lot_map,
        "proc_time_matrix": proc_time_matrix,
        "eqp_oper_avg":     eqp_oper_avg_val,
        "eqp_oper_cap":     eqp_oper_cap,
        "eqp_model_map":    eqp_model_map,
        "flow_next":        flow_next,
        "oper_eqp_models":  oper_eqp_models,
        "abstract_proc_time": abstract_proc_time_val,
        "plan_meta":        plan_meta,
        "max_proc_time":    max_proc_time,
        "max_wf_qty":       max_wf_qty,
        "abstract_route_map":       abstract_route_map,
        "abstract_routes_by_ppk_oper": abstract_routes_by_ppk_oper,
        "plan":             plan_list,
        "initial_schedule": initial_schedule,
        "flow":             flow_list,
        "arrange_actual_table": arrange_actual_table,
        "arrange_table":    arrange_actual_table,
        "split_rules":      split_lookup,
        "lot_inject_deadline": lot_inject_deadline,
    }
