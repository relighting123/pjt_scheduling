"""
tests/test_rts_eqpconvplan_timestamps.py

RTS_EQPCONVPLAN_INF/HIS는 CRT_TM/CHG_TM 컬럼을 가지며, 적재 SQL은 이 두 값을
세션 NLS_DATE_FORMAT에 좌우되지 않는 고정 포맷(YYYY-MM-DD HH24:MI:SS)으로
채워야 한다(INF/HIS 공통). RTS_EQPCONVPLAN_INF는 삭제 없이 INSERT만 한다.
"""
from datetime import datetime

from data.writer.rts_json import _build_rts_conv_rows
from data.writer.rts_sql import build_writer_sql_scripts

BASE_TIME = datetime(2026, 7, 15, 7, 0, 0)
META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "CRT_USER_ID": "RTS"}
NOW_EXPR = "TO_CHAR(SYSDATE, 'YYYY-MM-DD HH24:MI:SS')"


def _conv(eqp_id: str, start_min: int) -> dict:
    return {
        "eqp_id": eqp_id,
        "conv_start_min": start_min,
        "conv_end_min": start_min + 60,
        "conv_time": 60,
        "from_lot_cd": "LC_A",
        "to_lot_cd": "LC_B",
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "oper_id": "OPER001",
        "eqp_model_cd": "A",
    }


def _scripts():
    rows = _build_rts_conv_rows([_conv("EQP001", 0)], META, BASE_TIME)
    payload = {"meta": META, "RTS_RSLT_INF": [], "RTS_EQPCONVPLAN_INF": rows}
    return build_writer_sql_scripts(payload)


def test_eqpconvplan_inf_insert_sets_crt_tm_and_chg_tm_to_fixed_format():
    sql = _scripts()["rts_eqpconvplan_inf.sql"]
    assert "DELETE" not in sql
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert "CRT_TM" in sql and "CHG_TM" in sql
    assert insert_line.count(NOW_EXPR) == 2


def test_eqpconvplan_his_insert_sets_crt_tm_and_chg_tm_to_fixed_format():
    sql = _scripts()["rts_eqpconvplan_his.sql"]
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert "CRT_TM" in sql and "CHG_TM" in sql
    assert insert_line.count(NOW_EXPR) == 2


def test_rslt_inf_insert_sets_crt_tm_to_fixed_format():
    payload = {
        "meta": META,
        "RTS_RSLT_INF": [{
            "FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "LOT_CD": "LC_A",
            "TEMPER_VAL": "T600", "EQP_ID": "EQP001", "EQP_MODEL_CD": "A", "SEQ_NO": 1,
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "LOT_ID": "LOT001",
            "CARRIER_ID": "CAR001", "START_TIME": "20260715070000",
            "END_TIME": "20260715080000", "PRODUCE_QTY": 25, "CRT_USER_ID": "RTS",
        }],
        "RTS_EQPCONVPLAN_INF": [],
    }
    sql = build_writer_sql_scripts(payload)["rts_rslt_inf.sql"]
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert insert_line.count(NOW_EXPR) == 1
