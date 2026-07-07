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


def _build_rts_rslt_rows(
    schedule: List[dict],
    meta: Dict[str, str],
    base_time: datetime,
) -> List[dict]:
    """EQP별 SEQ_NO 부여 후 RTS_RSLT_INF 행 생성."""
    by_eqp: Dict[str, List[dict]] = defaultdict(list)
    for rec in schedule:
        by_eqp[rec["EQP_ID"]].append(rec)

    rows: List[dict] = []
    for eqp_id in sorted(by_eqp.keys()):
        ordered = sorted(by_eqp[eqp_id], key=lambda r: (r["START_TM"], r.get("SEQ", 0)))
        for seq_no, rec in enumerate(ordered, start=1):
            rows.append({
                "RULE_TIMEKEY":   meta["RULE_TIMEKEY"],
                "LOT_CD":         rec.get("LOT_CD", ""),
                "TEMPER_VAL":     rec.get("TEMP", ""),
                "EQP_ID":         eqp_id,
                "EQP_MODEL_CD":   rec.get("EQP_MODEL", ""),
                "SEQ_NO":         seq_no,
                "PLAN_PROD_ATTR_VAL":  rec["PLAN_PROD_ATTR_VAL"],
                "OPER_ID":        rec.get("OPER_ID", ""),
                "LOT_ID":         rec["LOT_ID"],
                "CARRIER_ID":     rec.get("CARRIER_ID", ""),
                "START_TIME":     minutes_to_timekey(int(rec["START_TM"]), base_time),
                "END_TIME":       minutes_to_timekey(int(rec["END_TM"]), base_time),
                "PRODUCE_QTY":    int(rec.get("WF_QTY", 0)),
                "CRT_USER_ID":    meta["CRT_USER_ID"],
            })
    return rows


def _build_rts_conv_rows(
    conversion_plans: List[dict],
    meta: Dict[str, str],
    base_time: datetime,
) -> List[dict]:
    """시뮬 conversion 이벤트 → RTS_EQPCONVPLAN_INF 행."""
    rows: List[dict] = []
    for i, ev in enumerate(conversion_plans, start=1):
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
        rows.append({
            "FAC_ID":                 meta["FAC_ID"],
            "RULE_TIMEKEY":           meta["RULE_TIMEKEY"],
            "PRCS_STAT_CD":           "PLAN",
            "JOB_ID":                 job_id,
            "RTS_GBN_CD":             "CONV",
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
            "REASON_CD":              "CONV",
            "REASON_CTN":             "LOT_CD/TEMP conversion",
            "TRANSMIT_YN":            "N",
            "TRANSMIT_TM":            None,
            "CRT_USER_ID":            meta["CRT_USER_ID"],
            "CHG_USER_ID":            meta["CRT_USER_ID"],
        })
    return rows


def build_rts_output(
    result: dict,
    env_data: dict,
    *,
    fac_id: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    crt_user_id: str = DEFAULT_CRT_USER,
) -> dict:
    """
    추론 결과 → RTS output.json 본문.

    Keys: meta, RTS_RSLT_INF, RTS_EQPCONVPLAN_INF
    """
    meta = resolve_writer_meta(
        env_data, fac_id=fac_id, rule_timekey=rule_timekey, crt_user_id=crt_user_id,
    )
    meta["ALGORITHM"] = result.get("algorithm", "scheduling_rl")
    base_time: datetime = env_data["sim_base_time"]
    schedule = result.get("schedule", [])
    conversion_plans = result.get("conversion_plans") or []

    return {
        "meta": meta,
        "RTS_RSLT_INF": _build_rts_rslt_rows(schedule, meta, base_time),
        "RTS_EQPCONVPLAN_INF": _build_rts_conv_rows(conversion_plans, meta, base_time),
    }
