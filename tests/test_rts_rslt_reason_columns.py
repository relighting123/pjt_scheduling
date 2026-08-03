"""
tests/test_rts_rslt_reason_columns.py

schedule 행의 PPK_OPER_REASON_CD/CTN, CARRIER_REASON_CD/CTN이 RTS_RSLT_MAS/HIS
output.json 및 Oracle INSERT SQL까지 그대로 전달되는지 검증한다
(tests/test_rts_rslt_new_columns.py와 동일한 패턴).
"""
from datetime import datetime

from data.writer.rts_json import build_rts_output
from data.writer.rts_sql import build_writer_sql_scripts

BASE_TIME = datetime(2026, 7, 19, 7, 0, 0)


def _schedule_row(**overrides):
    row = {
        "EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
        "START_TM": 0, "END_TM": 60, "WF_QTY": 25, "ST": 60,
        "LOT_CD": "LC_A", "TEMP": "T600", "EQP_MODEL": "A",
        "LOT_STAT_CD": "WAIT", "SEQ": 1,
        "PPK_OPER_REASON_CD": "SAME_SETUP",
        "PPK_OPER_REASON_CTN": "EQP001 ← PPK001/OPER001 선택 근거: 동일 셋업(전환 없음) 유지(+1.00)",
        "CARRIER_REASON_CD": "EARLIEST_ST",
        "CARRIER_REASON_CTN": "예상 종료 시각(전환+장수×ST)이 가장 이른 carrier 선택",
    }
    row.update(overrides)
    return row


def _env_data(**overrides):
    data = {
        "sim_base_time": BASE_TIME,
        "plan": [],
        "eqp_model_map": {"EQP001": "A"},
        "abstract_arrange_map": {("PPK001", "OPER001", "A"): 60},
        "eqp_oper_cap": {"EQP001": ["OPER001"]},
        "plan_meta": {("PPK001", "OPER001"): {"priority": 1, "d0_plan_qty": 75}},
    }
    data.update(overrides)
    return data


def _build(schedule, env_data):
    result = {"algorithm": "earliest_st", "schedule": schedule, "conversion_plans": []}
    return build_rts_output(result, env_data, fac_id="FAC001", rule_timekey="20260719070000")


def test_reason_columns_pass_through_to_rts_rslt_mas_row():
    payload = _build([_schedule_row()], _env_data())
    row = payload["RTS_RSLT_MAS"][0]
    assert row["PPK_OPER_REASON_CD"] == "SAME_SETUP"
    assert "PPK001/OPER001" in row["PPK_OPER_REASON_CTN"]
    assert row["CARRIER_REASON_CD"] == "EARLIEST_ST"
    assert "예상 종료" in row["CARRIER_REASON_CTN"]


def test_reason_columns_default_to_empty_string_when_missing_from_schedule_row():
    row = _schedule_row()
    for key in ("PPK_OPER_REASON_CD", "PPK_OPER_REASON_CTN", "CARRIER_REASON_CD", "CARRIER_REASON_CTN"):
        del row[key]
    payload = _build([row], _env_data())
    out = payload["RTS_RSLT_MAS"][0]
    assert out["PPK_OPER_REASON_CD"] == ""
    assert out["CARRIER_REASON_CTN"] == ""


def test_reason_columns_in_mas_and_his_insert_sql():
    payload = _build([_schedule_row()], _env_data())
    scripts = build_writer_sql_scripts(payload)
    for name in ("rts_rslt_mas.sql", "rts_rslt_his.sql"):
        insert_line = next(
            line for line in scripts[name].splitlines() if line.startswith("INSERT INTO")
        )
        for col in (
            "PPK_OPER_REASON_CD", "PPK_OPER_REASON_CTN",
            "CARRIER_REASON_CD", "CARRIER_REASON_CTN",
        ):
            assert col in insert_line, f"{col} missing from {name}"
        assert "SAME_SETUP" in insert_line
        assert "EARLIEST_ST" in insert_line
