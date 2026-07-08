"""
data/loader/preprocess.py – 원시 JSON → RL 환경 입력 변환
fetch.load_data()로 불러온 딕셔너리를 시뮬레이터·RL 환경용 구조로 가공합니다.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import (
    CONFIG, normalize_rule_timekey, RULE_TIMEKEY_FMT, PERIOD_SPLITS, rule_timekey_now,
)
from utils.helpers import (
    build_index_map, coerce_int, effective_proc_time, FORCED_LOT_STAT_ORDER,
    normalize_lot_stat_cd, normalize_tool_capacity_rows, split_wf_qty,
)


def _coerce_proc_time(value) -> Optional[int]:
    """discrete_arrange ST가 숫자(장당 소요시간 분/장)이면 int로 반환"""
    if value is None:
        return None
    try:
        return coerce_int(value, field="ST")
    except ValueError:
        return None


def _build_split_lookup(split_raw: List[dict]) -> Dict[Tuple[str, str, str], int]:
    lookup: Dict[Tuple[str, str, str], int] = {}
    for r in split_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        oper = str(r.get("OPER_ID", "*")).strip().upper() or "*"
        model = str(r["EQP_MODEL_CD"]).strip().upper()
        lookup[(ppk, oper, model)] = coerce_int(r["SPLIT_QTY"], field="SPLIT_QTY")
    return lookup


def _resolve_split_qty(
    ppk: str,
    oper_id: str,
    eqp_model: str,
    split_lookup: Dict[Tuple[str, str, str], int],
) -> Optional[int]:
    model = eqp_model.strip().upper()
    oper = (oper_id or "*").strip().upper()
    for key in (
        (ppk, oper, model),
        (ppk, oper, "*"),
        (ppk, "*", model),
        (ppk, "*", "*"),
    ):
        if key in split_lookup:
            return split_lookup[key]
    return None


def _child_lot_id(parent_id: str, index: int) -> str:
    return f"{parent_id}__S{index:02d}"


def _resolve_sim_base_time(period_key: Optional[str]) -> datetime:
    """RULE_TIMEKEY 07:00 기준 시뮬 시작 시각."""
    if period_key:
        return datetime.strptime(normalize_rule_timekey(period_key), RULE_TIMEKEY_FMT)
    if CONFIG.path.train_snapshot and CONFIG.path.dataset_split in PERIOD_SPLITS:
        return datetime.strptime(
            normalize_rule_timekey(CONFIG.path.train_snapshot), RULE_TIMEKEY_FMT,
        )
    return datetime.strptime(rule_timekey_now(), RULE_TIMEKEY_FMT)


def _build_flow_maps(
    flow_raw: List[dict],
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[str, int]]]:
    flow_map: Dict[str, Dict[int, str]] = {}
    oper_seq_map: Dict[str, Dict[str, int]] = {}
    for r in flow_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        seq = coerce_int(r["OPER_SEQ"], field="OPER_SEQ")
        oper_id = r["OPER_ID"]
        flow_map.setdefault(ppk, {})[seq] = oper_id
        oper_seq_map.setdefault(ppk, {})[oper_id] = seq
    return flow_map, oper_seq_map


def _resolve_lot_oper_seq(
    row: dict,
    flow_map: Dict[str, Dict[int, str]],
    oper_seq_map: Dict[str, Dict[str, int]],
) -> Tuple[str, int]:
    ppk = row["PLAN_PROD_ATTR_VAL"]
    if row.get("OPER_ID"):
        oper_id = row["OPER_ID"]
        seq = coerce_int(row.get("SEQ") or oper_seq_map.get(ppk, {}).get(oper_id, 1), field="SEQ")
        return oper_id, seq
    if row.get("SEQ") is not None:
        seq = coerce_int(row["SEQ"], field="SEQ")
        oper_id = flow_map.get(ppk, {}).get(seq, "OPER001")
        return oper_id, seq
    raise ValueError(
        f"discrete_arrange LOT {row.get('LOT_ID')}: OPER_ID 또는 SEQ가 필요합니다."
    )


def _default_lot_cd(lot_id: str, plan_prod_attr_val: str) -> str:
    suffix = plan_prod_attr_val.replace("PPK", "") if "PPK" in plan_prod_attr_val else lot_id[-3:]
    return f"LC{suffix.zfill(2)[-2:]}"


def _default_temp(plan_prod_attr_val: str) -> str:
    n = sum(ord(c) for c in plan_prod_attr_val)
    return "T650" if n % 2 == 0 else "T700"


def _build_batch_info_map(
    batch_info_raw: List[dict],
) -> Dict[Tuple[str, str], dict]:
    """(PPK, OPER) → {lot_cd, temp} — conversion/tool lookup."""
    out: Dict[Tuple[str, str], dict] = {}
    for r in batch_info_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        oper_id = r["OPER_ID"]
        out[(ppk, oper_id)] = {
            "lot_cd": str(r["LOT_CD"]),
            "temp":   str(r["TEMP"]),
        }
    return out


def _build_conversion_group_map(
    conv_group_raw: List[dict],
) -> Dict[Tuple[str, str], str]:
    """(LOT_CD, TEMP) → GROUP_ID. 같은 그룹끼리만 전환 허용(시뮬레이터에서 사용).

    입력 conversion_group.json 행: {GROUP_ID, LOT_CD, TEMP}.
    파일이 없거나 비면 빈 dict → 전환 그룹 제약 비활성(기존 동작 유지).
    """
    out: Dict[Tuple[str, str], str] = {}
    for r in conv_group_raw:
        gid = r.get("GROUP_ID")
        lot_cd = r.get("LOT_CD")
        if gid is None or lot_cd is None:
            continue
        temp = r.get("TEMP", "")
        out[(str(lot_cd), str(temp or ""))] = str(gid)
    return out


def _build_lot_attributes(
    lot_ids: List[str],
    lot_info: Dict[str, dict],
    lot_master_raw: List[dict],
    batch_info_map: Dict[Tuple[str, str], dict],
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
            ppk = lot_info[lid]["PLAN_PROD_ATTR_VAL"]
            oper_id = lot_info[lid]["oper_id"]
            route = batch_info_map.get((ppk, oper_id))
            if route:
                lot_cd = route["lot_cd"]
                temp = route["temp"]
            else:
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
    for route in batch_info_map.values():
        lot_cd_set.add(route["lot_cd"])
        temp_set.add(route["temp"])
    return attrs, sorted(lot_cd_set), sorted(temp_set)


def _normalize_eqp_initial_state(rows: List[dict]) -> List[dict]:
    """EQP 초기 LOT_CD/TEMP/PPK 상태 (conversion 시나리오용)."""
    out: List[dict] = []
    for r in rows:
        eid = str(r.get("EQP_ID", "")).strip()
        if not eid:
            continue
        out.append({
            "eqp_id":        eid,
            "lot_cd":        str(r.get("LOT_CD", "")).strip(),
            "temp":          str(r.get("TEMP", "")).strip(),
            "PLAN_PROD_ATTR_VAL": str(r.get("PLAN_PROD_ATTR_VAL", "")).strip() or None,
            "oper_id":       str(r.get("OPER_ID", "")).strip() or None,
        })
    return out


def _carrier_instance_id(row: dict) -> str:
    """discrete_arrange 1행 → 런타임 carrier 단위 키. LOT_ID:CARRIER_ID = 1:N."""
    carrier_id = str(row.get("CARRIER_ID") or "").strip()
    lot_id = row["LOT_ID"]
    return carrier_id if carrier_id else lot_id


def _build_tool_capacity_map(
    tool_raw: List[dict],
    lot_cds: List[str],
    eqp_models: List[str],
) -> Dict[Tuple[str, str], int]:
    cap: Dict[Tuple[str, str], int] = {}
    for r in tool_raw:
        lot_cd = str(r["LOT_CD"]).strip()
        model = str(r["EQP_MODEL_CD"]).strip().upper()
        cap[(lot_cd, model)] = coerce_int(r["MAX_TOOL"], field="MAX_TOOL")
    if cap:
        return cap
    for lc in lot_cds:
        for model in eqp_models:
            cap[(lc, model)] = 2
    return cap


def _apply_wafer_lot_split(
    lot_info: Dict[str, dict],
    eqp_lot_map: Dict[str, List[str]],
    proc_time_matrix: Dict[Tuple[str, str, str], int],
    discrete_raw: List[dict],
    eqp_model_map: Dict[str, str],
    split_lookup: Dict[Tuple[str, str, str], int],
    eqp_forced_queue: Optional[Dict[str, List[str]]] = None,
) -> None:
    """PPK×OPER×MODEL SPLIT_QTY 규칙에 따라 LOT을 wafer sub-lot으로 분할"""
    if not split_lookup:
        return

    split_children: Dict[str, List[Tuple[str, int]]] = {}

    for parent_id in list(lot_info.keys()):
        info = lot_info[parent_id]
        wf = coerce_int(info["wf_qty"], field="wf_qty")
        model = eqp_model_map[info["original_eqp"]]
        split_qty = _resolve_split_qty(
            info["PLAN_PROD_ATTR_VAL"], info["oper_id"], model, split_lookup,
        )
        if split_qty is None or split_qty <= 0 or wf <= split_qty:
            continue

        sizes = split_wf_qty(wf, split_qty)
        if len(sizes) <= 1:
            continue

        child_ids = [_child_lot_id(parent_id, i) for i in range(1, len(sizes) + 1)]
        split_children[parent_id] = list(zip(child_ids, sizes))
        del lot_info[parent_id]

        parent_carrier = info.get("carrier_id", "")
        parent_logical = info.get("logical_lot_id", parent_id)
        for i, (cid, qty) in enumerate(zip(child_ids, sizes), start=1):
            lot_info[cid] = {
                **info,
                "lot_id":          cid,
                "wf_qty":          qty,
                "carrier_id":      f"{parent_carrier}__S{i:02d}" if parent_carrier else cid,
                "parent_lot_id":   parent_id,
                "logical_lot_id":  parent_logical,
            }

        for eid, lots in eqp_lot_map.items():
            if parent_id not in lots:
                continue
            idx = lots.index(parent_id)
            lots[idx:idx + 1] = child_ids

        for eid, lots in (eqp_forced_queue or {}).items():
            if parent_id not in lots:
                continue
            idx = lots.index(parent_id)
            lots[idx:idx + 1] = child_ids

        for key, pt in list(proc_time_matrix.items()):
            lid, eid, oper = key
            if lid != parent_id:
                continue
            del proc_time_matrix[key]
            for cid in child_ids:
                proc_time_matrix[(cid, eid, oper)] = pt

    expanded_avail: List[dict] = []
    for r in discrete_raw:
        lot_id = r["LOT_ID"]
        if lot_id in split_children:
            parent_carrier = r.get("CARRIER_ID", "")
            for i, (cid, qty) in enumerate(split_children[lot_id], start=1):
                row = dict(r)
                row["LOT_ID"]     = cid
                row["WF_QTY"]     = qty
                row["CARRIER_ID"] = f"{parent_carrier}__S{i:02d}" if parent_carrier else cid
                expanded_avail.append(row)
            continue
        expanded_avail.append(dict(r))

    discrete_raw.clear()
    discrete_raw.extend(expanded_avail)


def _build_abstract_arrange_maps(
    abstract_raw: List[dict],
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str], List[Tuple[str, int]]]]:
    """abstract_arrange.json → arrange lookup."""
    arrange_map: Dict[Tuple[str, str, str], int] = {}
    by_ppk_oper: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for r in abstract_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        oper_id = r["OPER_ID"]
        model = str(r["EQP_MODEL_CD"])
        st = _coerce_proc_time(r.get("ST")) or 60
        arrange_map[(ppk, oper_id, model)] = st
        by_ppk_oper.setdefault((ppk, oper_id), []).append((model, st))
    return arrange_map, by_ppk_oper


def _build_abstract_inventory(
    abstract_raw: List[dict],
    flow_list: Dict[str, List[dict]],
    plan_meta: Dict[Tuple[str, str], dict],
    lot_info: Dict[str, dict],
    lot_initial_start: Dict[str, int],
    arrange_map: Dict[Tuple[str, str, str], int],
) -> Tuple[List[dict], Dict[Tuple[str, str], dict], Dict[str, dict]]:
    """
    PPK×OPER×MODEL abstract 템플릿(평균 ST) + (PPK,OPER)별 초기 WIP 카운터.

    WIP는 LOT 단위(+1/-1). oper_in_time / min_inject_time은 해당 풀의 최신·최早 투입 시각.
    """
    seq_map: Dict[Tuple[str, str], int] = {}
    for ppk, steps in flow_list.items():
        for s in steps:
            seq_map[(ppk, s["oper_id"])] = s["seq_id"]

    ppk_wf: Dict[str, int] = {}
    for ld in lot_info.values():
        ppk_wf[ld["PLAN_PROD_ATTR_VAL"]] = ld["wf_qty"]

    inventory: List[dict] = []
    seen: set = set()
    for r in abstract_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        oper_id = r["OPER_ID"]
        model = str(r["EQP_MODEL_CD"])
        abs_key = f"{ppk}|{oper_id}|{model}"
        if abs_key in seen:
            continue
        seen.add(abs_key)
        pm = plan_meta.get((ppk, oper_id), {})
        inventory.append({
            "abs_key":         abs_key,
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id":         oper_id,
            "eqp_model":       model,
            "seq":             seq_map.get((ppk, oper_id), 1),
            "proc_time":       arrange_map.get((ppk, oper_id, model), 60),
            "wf_qty":          ppk_wf.get(ppk, 25),
            "plan_priority":   pm.get("priority", 1),
            "d0_plan_qty":     pm.get("d0_plan_qty", 0),
        })

    wip_init: Dict[Tuple[str, str], dict] = {}
    lot_meta: Dict[str, dict] = {}
    for lid, ld in lot_info.items():
        key = (ld["PLAN_PROD_ATTR_VAL"], ld["oper_id"])
        st_tm = lot_initial_start.get(lid, 0)
        if key not in wip_init:
            wip_init[key] = {
                "wip_qty":         0,
                "wip_qty_init":    0,
                "oper_in_time":    0,
                "min_inject_time": st_tm,
                "lot_ids":         [],
            }
        wip = wip_init[key]
        wip["wip_qty"] += 1
        wip["wip_qty_init"] += 1
        wip["lot_ids"].append(lid)
        wip["oper_in_time"] = max(wip["oper_in_time"], st_tm)
        wip["min_inject_time"] = min(wip["min_inject_time"], st_tm)
        lot_meta[lid] = {
            "PLAN_PROD_ATTR_VAL": ld["PLAN_PROD_ATTR_VAL"],
            "oper_id":        ld["oper_id"],
            "seq":            ld["seq"],
            "wf_qty":         ld["wf_qty"],
            "carrier_id":     ld.get("carrier_id", ""),
            "logical_lot_id": ld.get("logical_lot_id", lid),
            "oper_in_time":   st_tm,
            "lot_stat_cd":    ld.get("lot_stat_cd", "WAIT"),
        }

    return inventory, wip_init, lot_meta


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
            "discrete_arrange": [{EQP_ID, LOT_ID, PLAN_PROD_ATTR_VAL, OPER_ID, ST, EQP_MODEL_CD, WF_QTY, ...}, ...],
            "abstract_arrange": [{PLAN_PROD_ATTR_VAL, OPER_ID, EQP_MODEL_CD, ST}, ...],
            "plan":         [{PLAN_PROD_ATTR_VAL, OPER_ID, D0_PLAN_QTY, D1_PLAN_QTY, PLAN_PRIORITY}, ...],
            "flow":         [{PLAN_PROD_ATTR_VAL, OPER_SEQ, OPER_ID}, ...]
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
            "lots": [LotDict],               # LOT 세부 정보 (lot_stat_cd 포함)
            "eqp_lot_map": {eqp: [lot_id]}, # EQP별 배정 가능 LOT
            "eqp_forced_queue": {eqp: [lot_id]}, # LOT_STAT_CD!=WAIT LOT을 입력 순서대로 강제 배정
            "plan": [PlanDict],              # 계획 데이터
            "flow": {prod: [{seq, oper}]},   # 제품별 FLOW 순서
        }
    """
    discrete_raw = list(raw["discrete_arrange"])
    plan_raw  = raw["plan"]
    flow_raw  = raw["flow"]
    split_raw = raw.get("split", [])
    lot_master_raw = raw.get("lot_master", [])
    batch_info_raw = raw.get("batch_info", [])
    tool_capacity_raw = normalize_tool_capacity_rows(raw.get("tool_capacity", []))

    abstract_raw = raw.get("abstract_arrange", [])
    if not abstract_raw:
        from data.generator import build_abstract_arrange
        abstract_raw = build_abstract_arrange(discrete_raw, flow_raw)

    flow_map, oper_seq_map = _build_flow_maps(flow_raw)

    # ── 기준 시각 (RULE_TIMEKEY 07:00) ───────────────────────────────────────
    base_time = _resolve_sim_base_time(period_key)
    sim_end_minutes = CONFIG.env.hard_horizon_minutes
    soft_cutoff_minutes = CONFIG.env.soft_cutoff_minutes

    # ── 범주형 목록 & 인덱스 맵 ──────────────────────────────────────────────
    eqp_ids  = sorted({r["EQP_ID"] for r in discrete_raw})
    oper_ids = sorted({r["OPER_ID"] for r in flow_raw})
    prod_keys = sorted({
        r["PLAN_PROD_ATTR_VAL"]
        for r in discrete_raw + abstract_raw + plan_raw
    })

    oper_idx = build_index_map(oper_ids)
    prod_idx = build_index_map(prod_keys)

    # discrete_arrange로 EQP별 LOT 및 WF_QTY 조회
    avail_map: Dict[Tuple[str, str], int] = {}
    eqp_lot_map: Dict[str, List[str]] = {}
    for r in discrete_raw:
        carrier_id = _carrier_instance_id(r)
        key = (r["EQP_ID"], carrier_id)
        avail_map[key] = coerce_int(r["WF_QTY"], field="WF_QTY")
        eqp_lot_map.setdefault(r["EQP_ID"], [])
        if carrier_id not in eqp_lot_map[r["EQP_ID"]]:
            eqp_lot_map[r["EQP_ID"]].append(carrier_id)

    # LOT 정보 빌드 — carrier 단위 1행 (LOT_ID:CARRIER_ID = 1:N)
    lot_info: Dict[str, dict] = {}
    seen_carriers: set = set()
    eqp_forced_queue: Dict[str, List[str]] = {}
    forced_input_seq: Dict[str, int] = {}
    forced_seq_counter = 0
    for r in discrete_raw:
        carrier_id = _carrier_instance_id(r)
        if carrier_id in seen_carriers:
            continue
        seen_carriers.add(carrier_id)
        logical_lot_id = r["LOT_ID"]

        oper_id, seq = _resolve_lot_oper_seq(r, flow_map, oper_seq_map)
        ppk = r["PLAN_PROD_ATTR_VAL"]
        wf_qty = coerce_int(r["WF_QTY"], field="WF_QTY")

        priority = 1
        for p in plan_raw:
            if p["PLAN_PROD_ATTR_VAL"] == ppk and p["OPER_ID"] == oper_id:
                priority = coerce_int(p.get("PLAN_PRIORITY", 1), field="PLAN_PRIORITY")
                break

        st_per_wafer = _coerce_proc_time(r.get("ST")) or 60
        eqp_id = r["EQP_ID"]
        lot_stat_cd = normalize_lot_stat_cd(r.get("LOT_STAT_CD"), lot_id=carrier_id)
        lot_info[carrier_id] = {
            "lot_id":          carrier_id,
            "logical_lot_id":  logical_lot_id,
            "carrier_id":      r.get("CARRIER_ID", f"CAR{logical_lot_id[-3:]}"),
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id":         oper_id,
            "seq":             seq,
            "wf_qty":          wf_qty,
            "processing_time": st_per_wafer,
            "priority":        priority,
            "original_eqp":    eqp_id,
            "lot_stat_cd":     lot_stat_cd,
        }
        if lot_stat_cd != "WAIT":
            eqp_forced_queue.setdefault(eqp_id, []).append(carrier_id)
            forced_input_seq[carrier_id] = forced_seq_counter
            forced_seq_counter += 1

    for eqp_id, queue in eqp_forced_queue.items():
        eqp_forced_queue[eqp_id] = sorted(
            queue,
            key=lambda cid: (
                FORCED_LOT_STAT_ORDER.get(lot_info[cid]["lot_stat_cd"], 99),
                forced_input_seq.get(cid, 0),
            ),
        )

    # 계획 데이터 정리
    plan_list = []
    for p in plan_raw:
        plan_list.append({
            "PLAN_PROD_ATTR_VAL": p["PLAN_PROD_ATTR_VAL"],
            "oper_id":       p["OPER_ID"],
            "d0_plan_qty":   coerce_int(p["D0_PLAN_QTY"], field="D0_PLAN_QTY"),
            "d1_plan_qty":   coerce_int(p["D1_PLAN_QTY"], field="D1_PLAN_QTY"),
            "priority":      coerce_int(p.get("PLAN_PRIORITY", 1), field="PLAN_PRIORITY"),
        })

    # FLOW 데이터 정리
    flow_list: Dict[str, List[dict]] = {}
    for r in flow_raw:
        ppk = r["PLAN_PROD_ATTR_VAL"]
        flow_list.setdefault(ppk, [])
        flow_list[ppk].append({"seq_id": coerce_int(r["OPER_SEQ"], field="OPER_SEQ"), "oper_id": r["OPER_ID"]})
    for ppk in flow_list:
        flow_list[ppk].sort(key=lambda x: x["seq_id"])

    # ── (LOT, EQP, OPER) 조합별 처리시간 행렬 ───────────────────────────────────
    proc_time_matrix: Dict[Tuple[str, str, str], int] = {}
    for r in discrete_raw:
        lid = _carrier_instance_id(r)
        eid = r["EQP_ID"]
        oper_id = r.get("OPER_ID") or lot_info.get(lid, {}).get("oper_id", "")
        pt = _coerce_proc_time(r.get("ST"))
        if pt is not None and oper_id:
            proc_time_matrix[(lid, eid, oper_id)] = pt

    eqp_oper_avg: Dict[Tuple[str, str], list] = {}
    for (lid, eid, oper_id), pt in proc_time_matrix.items():
        eqp_oper_avg.setdefault((eid, oper_id), []).append(pt)
    eqp_oper_avg_val: Dict[Tuple[str, str], int] = {
        k: int(sum(v) / len(v)) for k, v in eqp_oper_avg.items()
    }

    for key, pt in list(proc_time_matrix.items()):
        if pt > 0:
            continue
        lid, eid, oper_id = key
        proc_time_matrix[key] = eqp_oper_avg_val.get(
            (eid, oper_id),
            lot_info.get(lid, {}).get("processing_time", 60),
        )

    # ── EQP → 처리 가능 OPER 집합 ────────────────────────────────────────────
    eqp_oper_cap: Dict[str, List[str]] = {}
    for r in discrete_raw:
        eid = r["EQP_ID"]
        lid = _carrier_instance_id(r)
        if lid in lot_info:
            oper_id = lot_info[lid]["oper_id"]
            eqp_oper_cap.setdefault(eid, [])
            if oper_id not in eqp_oper_cap[eid]:
                eqp_oper_cap[eid].append(oper_id)

    # ── EQP 인덱스 맵 (Group E 관측 행렬 열 인덱싱용) ─────────────────────────
    eqp_idx = {eid: i for i, eid in enumerate(eqp_ids)}

    # 집계 정규화 스케일 팩터 (split 이후 LOT별 실제 소요시간 기준)
    max_proc_time = 1

    # ── EQP 장비 MODEL (availability EQP_MODEL_CD) ──────────────────────────
    eqp_model_map: Dict[str, str] = {}
    for r in discrete_raw:
        eid = r["EQP_ID"]
        eqp_model_map[eid] = str(r["EQP_MODEL_CD"])

    abstract_arrange_map, abstract_arranges_by_ppk_oper = _build_abstract_arrange_maps(abstract_raw)

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
        model = eqp_model_map[eid]
        for oper_id in opers:
            oper_eqp_models.setdefault(oper_id, [])
            if model not in oper_eqp_models[oper_id]:
                oper_eqp_models[oper_id].append(model)
    for (lid, eid, oper_id), pt in proc_time_matrix.items():
        if lid not in lot_info:
            continue
        model = eqp_model_map[eid]
        abstract_proc_time.setdefault((oper_id, model), []).append(pt)
    abstract_proc_time_val: Dict[Tuple[str, str], int] = {
        k: int(sum(v) / len(v)) for k, v in abstract_proc_time.items()
    }

    plan_meta: Dict[Tuple[str, str], dict] = {}
    for p in plan_list:
        plan_meta[(p["PLAN_PROD_ATTR_VAL"], p["oper_id"])] = {
            "priority":    p["priority"],
            "d0_plan_qty": p["d0_plan_qty"],
        }

    split_lookup = _build_split_lookup(split_raw)
    _apply_wafer_lot_split(
        lot_info,
        eqp_lot_map,
        proc_time_matrix,
        discrete_raw,
        eqp_model_map,
        split_lookup,
        eqp_forced_queue,
    )
    eqp_oper_cap = _rebuild_eqp_oper_cap(discrete_raw, lot_info)

    for (lid, _eid, _oper_id), st_pw in proc_time_matrix.items():
        wf = int(lot_info.get(lid, {}).get("wf_qty", 1))
        max_proc_time = max(max_proc_time, effective_proc_time(st_pw, wf))

    lot_initial_start: Dict[str, int] = {lid: 0 for lid in lot_info}

    abstract_inventory, abstract_wip_init, abstract_lot_meta = _build_abstract_inventory(
        abstract_raw,
        flow_list,
        plan_meta,
        lot_info,
        lot_initial_start,
        abstract_arrange_map,
    )

    eqp_idx = build_index_map(eqp_ids)

    batch_info_map = _build_batch_info_map(batch_info_raw)
    conv_group_map = _build_conversion_group_map(raw.get("conversion_group", []))

    lot_ids = sorted(lot_info.keys())
    lot_attrs, lot_cd_ids, temp_ids = _build_lot_attributes(
        lot_ids, lot_info, lot_master_raw, batch_info_map,
    )
    lot_cd_idx = build_index_map(lot_cd_ids)
    temp_idx = build_index_map(temp_ids)
    eqp_models = sorted(set(eqp_model_map.values()))
    tool_capacity = _build_tool_capacity_map(tool_capacity_raw, lot_cd_ids, eqp_models)

    # ── arrange 테이블 (availability 전체 – 초기 Actual 재공) ────────────────
    arrange_actual_table: List[dict] = []
    for r in discrete_raw:
        lot_id = _carrier_instance_id(r)
        eid = r["EQP_ID"]
        oper_id = r.get("OPER_ID") or lot_info.get(lot_id, {}).get("oper_id", "")
        st_per_wafer = proc_time_matrix.get((lot_id, eid, oper_id), 60)
        wf_qty_row = coerce_int(r["WF_QTY"], field="WF_QTY")
        row_model = str(r["EQP_MODEL_CD"])
        arrange_actual_table.append({
            "eqp_id":           eid,
            "lot_id":           lot_id,
            "oper_id":          oper_id,
            "PLAN_PROD_ATTR_VAL": r["PLAN_PROD_ATTR_VAL"],
            "st":               st_per_wafer,
            "proc_time":        effective_proc_time(st_per_wafer, wf_qty_row),
            "eqp_model":        row_model,
            "initial_start_tm": 0,
            "wf_qty":           wf_qty_row,
            "lot_stat_cd":      lot_info.get(lot_id, {}).get("lot_stat_cd", "WAIT"),
        })

    max_wf_qty = max((v["wf_qty"] for v in lot_info.values()), default=1)

    # ── Bucket(state) 사전계산: model 인덱스 / 공정별 장비 수 / takt 상수 / flow 전후 ──
    model_idx = build_index_map(eqp_models)
    n_eqp_per_oper: Dict[str, int] = {}
    for opers in eqp_oper_cap.values():
        for op in opers:
            n_eqp_per_oper[op] = n_eqp_per_oper.get(op, 0) + 1
    max_arrange_st = max((v for v in abstract_arrange_map.values()), default=1)
    flow_prev: Dict[str, Dict[str, Optional[str]]] = {}
    flow_post: Dict[str, Dict[str, Optional[str]]] = {}
    for ppk, steps in flow_list.items():
        ordered = [s["oper_id"] for s in sorted(steps, key=lambda x: x["seq_id"])]
        for i, op in enumerate(ordered):
            flow_prev.setdefault(ppk, {})[op] = ordered[i - 1] if i > 0 else None
            flow_post.setdefault(ppk, {})[op] = ordered[i + 1] if i + 1 < len(ordered) else None

    return {
        "sim_base_time":    base_time,
        "sim_end_minutes":  sim_end_minutes,
        "soft_cutoff_minutes": soft_cutoff_minutes,
        "conversion_minutes": CONFIG.env.conversion_minutes,
        "max_conversions": CONFIG.env.max_conversions,
        "max_conversions_per_eqp": CONFIG.env.max_conversions_per_eqp,
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
        "batch_info_map":   batch_info_map,
        "conv_group_map":   conv_group_map,
        "tool_capacity":    tool_capacity,
        "lots":             list(lot_info.values()),
        "eqp_lot_map":      eqp_lot_map,
        "eqp_forced_queue": eqp_forced_queue,
        "proc_time_matrix": proc_time_matrix,
        "eqp_oper_avg":     eqp_oper_avg_val,
        "eqp_oper_cap":     eqp_oper_cap,
        "eqp_model_map":    eqp_model_map,
        "eqp_models":       eqp_models,
        "model_idx":        model_idx,
        "n_eqp_per_oper":   n_eqp_per_oper,
        "max_arrange_st":     max_arrange_st,
        "flow_prev":        flow_prev,
        "flow_post":        flow_post,
        "flow_next":        flow_next,
        "oper_eqp_models":  oper_eqp_models,
        "abstract_proc_time": abstract_proc_time_val,
        "plan_meta":        plan_meta,
        "max_proc_time":    max_proc_time,
        "max_wf_qty":       max_wf_qty,
        "abstract_arrange_map":       abstract_arrange_map,
        "abstract_arranges_by_ppk_oper": abstract_arranges_by_ppk_oper,
        "abstract_inventory":       abstract_inventory,
        "abstract_wip_init":        abstract_wip_init,
        "abstract_lot_meta":        abstract_lot_meta,
        "plan":             plan_list,
        "flow":             flow_list,
        "arrange_actual_table": arrange_actual_table,
        "split_rules":      split_lookup,
        "lot_inject_deadline": {},
        "eqp_initial_state": _normalize_eqp_initial_state(raw.get("eqp_initial_state", [])),
    }
