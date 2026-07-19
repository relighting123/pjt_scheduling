"""
data/writer/rts_sql.py – RTS output.json → Oracle INSERT SQL (적재용)

RTS_RSLT_INF: 동일 FAC_ID 기준 전체 DELETE 후 INSERT (RULE_TIMEKEY 무관, 최신 결과만 유지)
RTS_EQPCONVPLAN_INF: 동일 FAC_ID+RULE_TIMEKEY 기존 행만 DELETE 후 INSERT
                      (같은 회차 재실행 시 JOB_ID 중복/PK 위반 방지, 다른 회차는 계속 누적)
HIS: INSERT only (EVENT_TIMEKEY = 생성 시각)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from config import RULE_TIMEKEY_FMT
from data.writer.rts_json import PRB_CARD_PLACEHOLDER


def _sql_str(val: Any) -> str:
    if val is None:
        return "NULL"
    return "'" + str(val).replace("'", "''") + "'"


def _sql_num(val: Any) -> str:
    if val is None:
        return "NULL"
    return str(int(val))


def _sql_float(val: Any) -> str:
    if val is None:
        return "NULL"
    return str(float(val))


def _delete_inf(table: str, fac_id: str) -> str:
    """RTS_RSLT_INF는 매 회차 동일 FAC_ID의 기존 행을 모두 비우고 최신 결과만 적재한다(RULE_TIMEKEY 무관)."""
    return f"DELETE FROM {table} WHERE FAC_ID = {_sql_str(fac_id)};"


def _delete_inf_for_rule_timekey(table: str, fac_id: str, rule_timekey: str) -> str:
    """동일 FAC_ID+RULE_TIMEKEY 기존 행만 비우고 이번 회차 결과로 대체한다(다른 회차는 유지)."""
    return (
        f"DELETE FROM {table} "
        f"WHERE FAC_ID = {_sql_str(fac_id)} AND RULE_TIMEKEY = {_sql_str(rule_timekey)};"
    )


def _insert_rts_rslt_inf(rows: List[dict], *, history: bool) -> List[str]:
    table = "RTS_RSLT_HIS" if history else "RTS_RSLT_INF"
    lines: List[str] = []
    event_key = datetime.now().strftime(RULE_TIMEKEY_FMT)
    for r in rows:
        cols = [
            "FAC_ID", "RULE_TIMEKEY", "LOT_CD", "TEMPER_VAL", "EQP_ID", "EQP_MODEL_CD",
            "SEQ_NO", "PLAN_PROD_ATTR_VAL", "OPER_ID", "LOT_ID", "CARRIER_ID",
            "START_TIME", "END_TIME", "PRODUCE_QTY", "CRT_USER_ID",
        ]
        vals = [
            _sql_str(r["FAC_ID"]),
            _sql_str(r["RULE_TIMEKEY"]),
            _sql_str(r["LOT_CD"]),
            _sql_str(r["TEMPER_VAL"]),
            _sql_str(r["EQP_ID"]),
            _sql_str(r["EQP_MODEL_CD"]),
            _sql_num(r["SEQ_NO"]),
            _sql_str(r["PLAN_PROD_ATTR_VAL"]),
            _sql_str(r["OPER_ID"]),
            _sql_str(r["LOT_ID"]),
            _sql_str(r["CARRIER_ID"]),
            _sql_str(r["START_TIME"]),
            _sql_str(r["END_TIME"]),
            _sql_num(r["PRODUCE_QTY"]),
            _sql_str(r.get("CRT_USER_ID", "RTS")),
        ]
        if history:
            cols.extend(["CRT_TM", "EVENT_TIMEKEY"])
            vals.extend(["SYSTIMESTAMP", _sql_str(event_key)])
        else:
            cols.append("CRT_TM")
            vals.append("SYSTIMESTAMP")
        lines.append(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        )
    return lines


def _insert_rts_eqpconvplan(rows: List[dict], *, history: bool) -> List[str]:
    table = "RTS_EQPCONVPLAN_HIS" if history else "RTS_EQPCONVPLAN_INF"
    lines: List[str] = []
    event_key = datetime.now().strftime(RULE_TIMEKEY_FMT)
    for r in rows:
        cols = [
            "FAC_ID", "RULE_TIMEKEY", "PRCS_STAT_CD", "JOB_ID", "REQ_GBN_CD",
            "EQP_ID", "EQP_MODEL_CD", "TESTER_EQP_MODEL_CD",
            "CONV_START_TM", "CONV_END_TM", "CONV_TIME",
            "LOT_CD", "PRB_CARD_NO", "TEMPER_VAL", "PLAN_PROD_ATTR_VAL",
            "TO_LOT_CD", "TO_PRB_CARD_NO", "PRB_CARD_NO_LVAL",
            "TO_TEMPER_VAL", "TO_PLAN_PROD_ATTR_VAL",
            "OPER_ID", "TO_OPER_ID", "REASON_CD", "REASON_CTN",
            "TRANSMIT_YN", "TRANSMIT_TM", "CRT_USER_ID", "CHG_USER_ID",
        ]
        transmit_tm = r.get("TRANSMIT_TM")
        vals = [
            _sql_str(r["FAC_ID"]),
            _sql_str(r["RULE_TIMEKEY"]),
            _sql_str(r.get("PRCS_STAT_CD", "PLAN")),
            _sql_str(r["JOB_ID"]),
            _sql_str(r.get("REQ_GBN_CD", "RTS")),
            _sql_str(r["EQP_ID"]),
            _sql_str(r["EQP_MODEL_CD"]),
            _sql_str(r.get("TESTER_EQP_MODEL_CD", r["EQP_MODEL_CD"])),
            _sql_str(r["CONV_START_TM"]),
            _sql_str(r["CONV_END_TM"]),
            _sql_num(r["CONV_TIME"]),
            _sql_str(r["LOT_CD"]),
            _sql_str(r.get("PRB_CARD_NO", PRB_CARD_PLACEHOLDER)),
            _sql_str(r["TEMPER_VAL"]),
            _sql_str(r.get("PLAN_PROD_ATTR_VAL", "")),
            _sql_str(r["TO_LOT_CD"]),
            _sql_str(r.get("TO_PRB_CARD_NO", PRB_CARD_PLACEHOLDER)),
            _sql_str(r.get("PRB_CARD_NO_LVAL", PRB_CARD_PLACEHOLDER)),
            _sql_str(r["TO_TEMPER_VAL"]),
            _sql_str(r.get("TO_PLAN_PROD_ATTR_VAL", "")),
            _sql_str(r["OPER_ID"]),
            _sql_str(r.get("TO_OPER_ID", r["OPER_ID"])),
            _sql_str(r.get("REASON_CD", "CONV")),
            _sql_str(r.get("REASON_CTN", "LOT_CD/TEMP conversion")),
            _sql_str(r.get("TRANSMIT_YN", "N")),
            _sql_str(transmit_tm) if transmit_tm else "NULL",
            _sql_str(r.get("CRT_USER_ID", "RTS")),
            _sql_str(r.get("CHG_USER_ID", r.get("CRT_USER_ID", "RTS"))),
        ]
        if history:
            cols.extend(["CRT_TM", "CHG_TM", "EVENT_TIMEKEY"])
            vals.extend(["SYSDATE", "SYSDATE", _sql_str(event_key)])
        else:
            cols.extend(["CRT_TM", "CHG_TM"])
            vals.extend(["SYSDATE", "SYSDATE"])
        lines.append(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        )
    return lines


def _insert_rts_perfmon_his(rows: List[dict]) -> List[str]:
    lines: List[str] = []
    for r in rows:
        cols = ["FAC_ID", "RULE_TIMEKEY", "FUNCTION_NM", "KPI_NM", "KPI_VAL", "CRT_USER_ID", "CRT_TM"]
        vals = [
            _sql_str(r["FAC_ID"]),
            _sql_str(r["RULE_TIMEKEY"]),
            _sql_str(r["FUNCTION_NM"]),
            _sql_str(r["KPI_NM"]),
            _sql_float(r["KPI_VAL"]),
            _sql_str(r.get("CRT_USER_ID", "RTS")),
            "SYSTIMESTAMP",
        ]
        lines.append(
            f"INSERT INTO RTS_PERFMON_HIS ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        )
    return lines


def _insert_rts_validation(rows: List[dict]) -> List[str]:
    lines: List[str] = []
    for r in rows:
        cols = [
            "FAC_ID", "RULE_TIMEKEY", "FUNCTION_NM", "EQP_ID",
            "PLAN_PROD_ATTR_VAL", "OPER_ID", "VIOLATION_CNT", "CRT_USER_ID", "CRT_TM",
        ]
        vals = [
            _sql_str(r["FAC_ID"]),
            _sql_str(r["RULE_TIMEKEY"]),
            _sql_str(r["FUNCTION_NM"]),
            _sql_str(r["EQP_ID"]),
            _sql_str(r["PLAN_PROD_ATTR_VAL"]),
            _sql_str(r["OPER_ID"]),
            _sql_num(r["VIOLATION_CNT"]),
            _sql_str(r.get("CRT_USER_ID", "RTS")),
            "SYSTIMESTAMP",
        ]
        lines.append(
            f"INSERT INTO RTS_VALIDATION ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        )
    return lines


def build_writer_sql_scripts(payload: dict, *, include_history: bool = True) -> Dict[str, str]:
    """output.json 본문 → {파일명: SQL 텍스트}."""
    meta = payload.get("meta", {})
    rule_timekey = meta.get("RULE_TIMEKEY", "")
    fac_id = meta.get("FAC_ID", "")
    rslt_rows = payload.get("RTS_RSLT_INF", [])
    conv_rows = payload.get("RTS_EQPCONVPLAN_INF", [])
    perfmon_rows = payload.get("RTS_PERFMON_HIS", [])
    validation_rows = payload.get("RTS_VALIDATION", [])

    scripts: Dict[str, str] = {}

    inf_lines = [f"-- RTS_RSLT_INF FAC_ID={fac_id} RULE_TIMEKEY={rule_timekey}", ""]
    inf_lines.append(_delete_inf("RTS_RSLT_INF", fac_id))
    inf_lines.append("")
    inf_lines.extend(_insert_rts_rslt_inf(rslt_rows, history=False))
    scripts["rts_rslt_inf.sql"] = "\n".join(inf_lines) + "\n"

    conv_inf_lines = [f"-- RTS_EQPCONVPLAN_INF FAC_ID={fac_id} RULE_TIMEKEY={rule_timekey}", ""]
    if rule_timekey:
        conv_inf_lines.append(_delete_inf_for_rule_timekey("RTS_EQPCONVPLAN_INF", fac_id, rule_timekey))
        conv_inf_lines.append("")
    conv_inf_lines.extend(_insert_rts_eqpconvplan(conv_rows, history=False))
    scripts["rts_eqpconvplan_inf.sql"] = "\n".join(conv_inf_lines) + "\n"

    if include_history:
        his_lines = [f"-- RTS_RSLT_HIS RULE_TIMEKEY={rule_timekey}", ""]
        his_lines.extend(_insert_rts_rslt_inf(rslt_rows, history=True))
        scripts["rts_rslt_his.sql"] = "\n".join(his_lines) + "\n"

        conv_his_lines = [f"-- RTS_EQPCONVPLAN_HIS RULE_TIMEKEY={rule_timekey}", ""]
        conv_his_lines.extend(_insert_rts_eqpconvplan(conv_rows, history=True))
        scripts["rts_eqpconvplan_his.sql"] = "\n".join(conv_his_lines) + "\n"

    if perfmon_rows:
        perfmon_lines = [f"-- RTS_PERFMON_HIS RULE_TIMEKEY={rule_timekey}", ""]
        perfmon_lines.extend(_insert_rts_perfmon_his(perfmon_rows))
        scripts["rts_perfmon_his.sql"] = "\n".join(perfmon_lines) + "\n"

    if validation_rows:
        validation_lines = [f"-- RTS_VALIDATION RULE_TIMEKEY={rule_timekey}", ""]
        validation_lines.extend(_insert_rts_validation(validation_rows))
        scripts["rts_validation.sql"] = "\n".join(validation_lines) + "\n"

    return scripts


def write_sql(
    payload: Union[dict, Path, str],
    output_dir: Path,
    *,
    include_history: bool = True,
) -> List[Path]:
    """output.json(또는 dict) → Oracle 적재용 SQL 파일 저장."""
    if not isinstance(payload, dict):
        path = Path(payload)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, text in build_writer_sql_scripts(payload, include_history=include_history).items():
        out = output_dir / name
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written
