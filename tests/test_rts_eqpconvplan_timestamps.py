"""
tests/test_rts_eqpconvplan_timestamps.py

RTS_EQPCONVPLAN_INF/HIS는 CRT_TM/CHG_TM 컬럼(DATE, DEFAULT SYSDATE)을 가지며,
적재 SQL은 이 두 값을 모두 SYSDATE로 채워야 한다(INF/HIS 공통).
RTS_RSLT_MAS의 CRT_TM은 TIMESTAMP 컬럼이라 SYSTIMESTAMP를 채운다.
"""
from datetime import datetime

from data.writer.rts_json import _build_rts_conv_rows
from data.writer.rts_sql import build_writer_sql_scripts

BASE_TIME = datetime(2026, 7, 15, 7, 0, 0)
META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "CRT_USER_ID": "RTS"}


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
    payload = {"meta": META, "RTS_RSLT_MAS": [], "RTS_EQPCONVPLAN_INF": rows}
    return build_writer_sql_scripts(payload)


def test_eqpconvplan_inf_insert_sets_crt_tm_and_chg_tm_to_sysdate():
    sql = _scripts()["rts_eqpconvplan_inf.sql"]
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert "CRT_TM" in sql and "CHG_TM" in sql
    assert insert_line.count("SYSDATE") == 2


def test_eqpconvplan_his_insert_sets_crt_tm_and_chg_tm_to_sysdate():
    sql = _scripts()["rts_eqpconvplan_his.sql"]
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert "CRT_TM" in sql and "CHG_TM" in sql
    assert insert_line.count("SYSDATE") == 2


def test_eqpconvplan_his_insert_has_exec_timekey():
    scripts = _scripts()
    his_line = next(
        line for line in scripts["rts_eqpconvplan_his.sql"].splitlines()
        if line.startswith("INSERT INTO")
    )
    assert "EXEC_TIMEKEY" in his_line
    inf_sql = scripts["rts_eqpconvplan_inf.sql"]
    assert "EXEC_TIMEKEY" not in inf_sql


def test_rslt_mas_insert_sets_crt_tm_to_systimestamp():
    payload = {
        "meta": META,
        "RTS_RSLT_MAS": [{
            "FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "LOT_CD": "LC_A",
            "TEMPER_VAL": "T600", "EQP_ID": "EQP001", "EQP_MODEL_CD": "A", "SEQ_NO": 1,
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "LOT_ID": "LOT001",
            "CARRIER_ID": "CAR001", "START_TIME": "20260715070000",
            "END_TIME": "20260715080000", "PRODUCE_QTY": 25, "CRT_USER_ID": "RTS",
        }],
        "RTS_EQPCONVPLAN_INF": [],
    }
    sql = build_writer_sql_scripts(payload)["rts_rslt_mas.sql"]
    insert_line = next(line for line in sql.splitlines() if line.startswith("INSERT INTO"))
    assert "SYSTIMESTAMP" in insert_line
