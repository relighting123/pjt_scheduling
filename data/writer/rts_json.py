"""
data/writer/rts_json.py – 추론 schedule → RTS output.json

RTS_RSLT_INF / RTS_EQPCONVPLAN_INF 스키마 JSON 생성.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import CONFIG, normalize_rule_timekey, parse_input_folder, rule_timekey_now

DEFAULT_CRT_USER = "RTS"
PRB_CARD_PLACEHOLDER = "-"


def minutes_to_timekey(minutes: int, base: datetime) -> str:
    """시뮬 분 → RULE_TIMEKEY 형식 YYYYMMDDHHmmss."""
    return (base + timedelta(minutes=minutes)).strftime("%Y%m%d%H%M%S")


def minutes_to_conv_tm(minutes: int, base: datetime) -> str:
    """시뮬 분 → CONV_START_TM / CONV_END_TM (YYYYMMDDHHmm, 12자리)."""
    return (base + timedelta(minutes=minutes)).strftime("%Y%m%d%H%M")


def resolve_writer_meta(
    env_data: Optional[dict] = None,
    *,
    fac_id: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    crt_user_id: str = DEFAULT_CRT_USER,
) -> Dict[str, str]:
    """입력 dataset 경로에서 FAC_ID, RULE_TIMEKEY 추출."""
    if fac_id and rule_timekey:
        return {
            "FAC_ID": fac_id,
            "RULE_TIMEKEY": normalize_rule_timekey(rule_timekey),
            "CRT_USER_ID": crt_user_id,
        }
    folder = CONFIG.path.input_folder_key
    parsed_fac, _split, period = parse_input_folder(folder)
    fac = fac_id or parsed_fac or CONFIG.path.fac_id
    rtk = rule_timekey or period or rule_timekey_now()
    return {
        "FAC_ID": fac,
        "RULE_TIMEKEY": normalize_rule_timekey(rtk),
        "CRT_USER_ID": crt_user_id,
    }


def _build_discrete_eqp_index(env_data: dict) -> Dict[tuple, List[str]]:
    """proc_time_matrix(lot_id, eqp_id, oper_id) → (lot_id, oper_id)별 discrete EQP_ID 목록."""
    index: Dict[tuple, List[str]] = defaultdict(list)
    for (lot_id, eqp_id, oper_id) in env_data.get("proc_time_matrix", {}):
        index[(lot_id, oper_id)].append(eqp_id)
    return {k: sorted(v) for k, v in index.items()}


def _abstract_eligible_eqp_ids(ppk: str, oper_id: str, env_data: dict) -> List[str]:
    """(PPK,OPER) 모델 매칭 기준 투입 가능 EQP 목록 — discrete 조합이 전혀 없는 순수
    abstract 재공에만 쓰는 fallback. abstract_arrange_map(EQP 모델) 또는
    eqp_oper_cap(discrete 실적으로 확인된 EQP별 가능 OPER) 중 하나라도 해당하면
    투입 가능으로 본다(simulator._eqp_can_process()와 동일 규칙).
    """
    eqp_model_map = env_data.get("eqp_model_map", {})
    arrange_map = env_data.get("abstract_arrange_map", {})
    eqp_oper_cap = env_data.get("eqp_oper_cap", {})
    eligible = [
        eqp_id for eqp_id, model in eqp_model_map.items()
        if (ppk, oper_id, model) in arrange_map or oper_id in eqp_oper_cap.get(eqp_id, [])
    ]
    return sorted(eligible)


def _prgs_enable_eqp_lval(
    internal_lot_id: str,
    ppk: str,
    oper_id: str,
    env_data: dict,
    discrete_index: Dict[tuple, List[str]],
    cache: dict,
) -> str:
    """PRGS_ENABLE_EQP_LVAL: 해당 재공이 투입 가능한 EQP_ID 목록(콤마 구분, 4000자 제한).

    이 LOT에 discrete_arrange로 명시된 (lot_id, eqp_id, oper_id) 조합이 있으면 그
    EQP들만 사용한다(실제 discrete 조건 그대로 — LOT마다 다르게 나와야 정상).
    그런 조합이 전혀 없는 순수 abstract 재공일 때만 (PPK,OPER) 모델 매칭으로 대체한다.
    """
    discrete_eqps = discrete_index.get((internal_lot_id, oper_id))
    if discrete_eqps:
        return ",".join(discrete_eqps)[:4000]

    key = (ppk, oper_id)
    if key not in cache:
        cache[key] = ",".join(_abstract_eligible_eqp_ids(ppk, oper_id, env_data))[:4000]
    return cache[key]


def _build_rts_rslt_rows(
    schedule: List[dict],
    meta: Dict[str, str],
    base_time: datetime,
    env_data: dict,
) -> List[dict]:
    """EQP별 SEQ_NO 부여 후 RTS_RSLT_INF 행 생성."""
    by_eqp: Dict[str, List[dict]] = defaultdict(list)
    for rec in schedule:
        by_eqp[rec["EQP_ID"]].append(rec)

    plan_meta = env_data.get("plan_meta", {})
    discrete_eqp_index = _build_discrete_eqp_index(env_data)
    eqp_lval_cache: dict = {}

    rows: List[dict] = []
    for eqp_id in sorted(by_eqp.keys()):
        ordered = sorted(by_eqp[eqp_id], key=lambda r: (r["START_TM"], r.get("SEQ", 0)))
        for seq_no, rec in enumerate(ordered, start=1):
            ppk = rec["PLAN_PROD_ATTR_VAL"]
            oper_id = rec.get("OPER_ID", "")
            # 시뮬레이터 내부 lot_id(=carrier 단위 키) 복원: CARRIER_ID가 있으면 그것이
            # 곧 내부 키이고(preprocess._carrier_instance_id와 동일 규칙), 없으면 LOT_ID.
            internal_lot_id = rec.get("CARRIER_ID") or rec["LOT_ID"]
            rows.append({
                "FAC_ID":         meta["FAC_ID"],
                "RULE_TIMEKEY":   meta["RULE_TIMEKEY"],
                "LOT_CD":         rec.get("LOT_CD", ""),
                "TEMPER_VAL":     rec.get("TEMP", ""),
                "EQP_ID":         eqp_id,
                "EQP_MODEL_CD":   rec.get("EQP_MODEL", ""),
                "SEQ_NO":         seq_no,
                "PLAN_PROD_ATTR_VAL":  ppk,
                "OPER_ID":        oper_id,
                "LOT_ID":         rec["LOT_ID"],
                "CARRIER_ID":     rec.get("CARRIER_ID", ""),
                "LOT_STAT_CD":    rec.get("LOT_STAT_CD", "WAIT"),
                "FLOW_ID":        ppk,
                "WF_QTY":         int(rec.get("WF_QTY", 0)),
                "ST":             int(rec.get("ST", 0)),
                "PRGS_ENABLE_EQP_LVAL": _prgs_enable_eqp_lval(
                    internal_lot_id, ppk, oper_id, env_data, discrete_eqp_index, eqp_lval_cache,
                ),
                "PLAN_QTY":       int(plan_meta.get((ppk, oper_id), {}).get("d0_plan_qty", 0)),
                "START_TIME":     minutes_to_timekey(int(rec["START_TM"]), base_time),
                "END_TIME":       minutes_to_timekey(int(rec["END_TM"]), base_time),
                "PRODUCE_QTY":    int(rec.get("WF_QTY", 0)),
                "FUNCTION_NM":    "TEST",
                "CRT_USER_ID":    meta["CRT_USER_ID"],
            })
    return rows


def _build_rts_conv_rows(
    conversion_plans: List[dict],
    meta: Dict[str, str],
    base_time: datetime,
) -> List[dict]:
    """시뮬 conversion 이벤트 → RTS_EQPCONVPLAN_INF 행.

    CONFIG.env.conv_output_enabled(옵션, 기본 True)가 False면 저장하지 않는다.
    활성화된 경우 RULE_TIMEKEY 기준 CONFIG.env.conv_output_window_minutes(기본 60분)
    이내에 시작하는 전환만 포함한다 — 그보다 먼 미래의 전환은 재계획 여지가 커
    추측성이므로 확정 출력(RTS_EQPCONVPLAN_INF/HIS)에 싣지 않는다.
    """
    if not CONFIG.env.conv_output_enabled:
        return []
    window = CONFIG.env.conv_output_window_minutes
    in_window = [ev for ev in conversion_plans if int(ev["conv_start_min"]) <= window]

    rows: List[dict] = []
    for i, ev in enumerate(in_window, start=1):
        conv_start = int(ev["conv_start_min"])
        conv_end = int(ev["conv_end_min"])
        conv_start_tm = minutes_to_conv_tm(conv_start, base_time)
        eqp_id = ev["eqp_id"]
        job_id = f"{eqp_id}_{conv_start_tm}_{i:04d}"
        from_lot = ev.get("from_lot_cd") or ""
        to_lot = ev.get("to_lot_cd") or ""
        ppk = ev.get("PLAN_PROD_ATTR_VAL", "")
        oper_id = ev.get("oper_id", "")
        eqp_model = ev.get("eqp_model_cd", "")
        is_scheduled = ev.get("source") == "SCHEDULED"
        reason_cd = "MANUAL" if is_scheduled else "CONV"
        reason_ctn = "외부 확정 전환 계획(eqp_conv_plan)" if is_scheduled else "LOT_CD/TEMP conversion"
        rows.append({
            "FAC_ID":                 meta["FAC_ID"],
            "RULE_TIMEKEY":           meta["RULE_TIMEKEY"],
            "PRCS_STAT_CD":           "PLAN",
            "JOB_ID":                 job_id,
            "REQ_GBN_CD":             "RTS",
            "EQP_ID":                 eqp_id,
            "EQP_MODEL_CD":           eqp_model,
            "TESTER_EQP_MODEL_CD":    eqp_model,
            "CONV_START_TM":          conv_start_tm,
            "CONV_END_TM":            minutes_to_conv_tm(conv_end, base_time),
            "CONV_TIME":              int(ev.get("conv_time", conv_end - conv_start)),
            "LOT_CD":                 from_lot,
            "PRB_CARD_NO":            PRB_CARD_PLACEHOLDER,
            "TEMPER_VAL":             ev.get("from_temp", ""),
            "PLAN_PROD_ATTR_VAL":     ppk,
            "TO_LOT_CD":              to_lot,
            "TO_PRB_CARD_NO":         PRB_CARD_PLACEHOLDER,
            "PRB_CARD_NO_LVAL":       PRB_CARD_PLACEHOLDER,
            "TO_TEMPER_VAL":          ev.get("to_temp", ""),
            "TO_PLAN_PROD_ATTR_VAL":  ppk,
            "OPER_ID":                oper_id,
            "TO_OPER_ID":             oper_id,
            "REASON_CD":              reason_cd,
            "REASON_CTN":             reason_ctn,
            "TRANSMIT_YN":            "N",
            "TRANSMIT_TM":            None,
            "CRT_USER_ID":            meta["CRT_USER_ID"],
            "CHG_USER_ID":            meta["CRT_USER_ID"],
        })
    return rows


def _build_rts_perfmon_rows(
    result: dict,
    meta: Dict[str, str],
) -> List[dict]:
    """추론 stats → RTS_PERFMON_HIS KPI 행 (옵션: save_kpi)."""
    stats = result.get("stats", {}) or {}
    schedule = result.get("schedule", [])
    function_nm = result.get("algorithm", "scheduling_rl")

    eqp_ids = {rec["EQP_ID"] for rec in schedule if rec.get("EQP_ID")}
    busy_total = sum(
        int(rec.get("PROC_TIME") or (rec.get("END_TM", 0) - rec.get("START_TM", 0)))
        for rec in schedule
    )
    sim_end = stats.get("sim_end_minutes") or 0
    utilization = (
        round(100 * busy_total / (len(eqp_ids) * sim_end), 2) if eqp_ids and sim_end else 0.0
    )

    kpis: Dict[str, float] = {
        "IDLE_TOTAL":            stats.get("idle_total", 0),
        "OPER_SWITCHES":         stats.get("oper_switches", 0),
        "PROD_SWITCHES":         stats.get("prod_switches", 0),
        "CONVERSIONS":           stats.get("conversions", 0),
        "UTILIZATION_PCT":       utilization,
    }

    return [
        {
            "FAC_ID":       meta["FAC_ID"],
            "RULE_TIMEKEY": meta["RULE_TIMEKEY"],
            "FUNCTION_NM":  function_nm,
            "KPI_NM":       kpi_nm,
            "KPI_VAL":      kpi_val,
            "CRT_USER_ID":  meta["CRT_USER_ID"],
        }
        for kpi_nm, kpi_val in kpis.items()
    ]


def _build_rts_validation_rows(
    result: dict,
    meta: Dict[str, str],
) -> List[dict]:
    """discrete/abstract arrange 기준 투입 불가 장비 재공 선택 건수 → RTS_VALIDATION 행 (옵션: save_kpi)."""
    violations = (result.get("validation") or {}).get("eligibility_violations", [])
    function_nm = result.get("algorithm", "scheduling_rl")

    counts: Dict[tuple, int] = defaultdict(int)
    for v in violations:
        key = (v["eqp_id"], v["PLAN_PROD_ATTR_VAL"], v["oper_id"])
        counts[key] += 1

    return [
        {
            "FAC_ID":              meta["FAC_ID"],
            "RULE_TIMEKEY":        meta["RULE_TIMEKEY"],
            "FUNCTION_NM":         function_nm,
            "EQP_ID":              eqp_id,
            "PLAN_PROD_ATTR_VAL":  ppk,
            "OPER_ID":             oper_id,
            "VIOLATION_CNT":       cnt,
            "CRT_USER_ID":         meta["CRT_USER_ID"],
        }
        for (eqp_id, ppk, oper_id), cnt in sorted(counts.items())
    ]


def build_rts_output(
    result: dict,
    env_data: dict,
    *,
    fac_id: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    crt_user_id: str = DEFAULT_CRT_USER,
    include_kpi: bool = False,
) -> dict:
    """
    추론 결과 → RTS output.json 본문.

    Keys: meta, RTS_RSLT_INF, RTS_EQPCONVPLAN_INF, (옵션) RTS_PERFMON_HIS, RTS_VALIDATION
    """
    meta = resolve_writer_meta(
        env_data, fac_id=fac_id, rule_timekey=rule_timekey, crt_user_id=crt_user_id,
    )
    meta["ALGORITHM"] = result.get("algorithm", "scheduling_rl")
    base_time: datetime = env_data["sim_base_time"]
    schedule = result.get("schedule", [])
    conversion_plans = result.get("conversion_plans") or []

    payload = {
        "meta": meta,
        "RTS_RSLT_INF": _build_rts_rslt_rows(schedule, meta, base_time, env_data),
        "RTS_EQPCONVPLAN_INF": _build_rts_conv_rows(conversion_plans, meta, base_time),
    }
    if include_kpi:
        payload["RTS_PERFMON_HIS"] = _build_rts_perfmon_rows(result, meta)
        payload["RTS_VALIDATION"] = _build_rts_validation_rows(result, meta)
    return payload
