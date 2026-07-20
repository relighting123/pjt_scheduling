"""
tests/test_rts_rslt_new_columns.py

RTS_RSLT_MAS/HIS에 추가된 컬럼들을 검증한다:
  - FUNCTION_NM: 항상 'TEST'
  - LOT_STAT_CD/FLOW_ID/WF_QTY/ST: schedule 행에서 그대로 파생
  - PRGS_ENABLE_EQP_LVAL: LOT에 discrete_arrange 조합이 있으면 그 EQP만(LOT마다
    달라야 함), 순수 abstract 재공이면 (PPK,OPER) 모델 매칭 EQP 목록(콤마 구분)
  - PLAN_QTY: plan_meta의 당일(D0) 계획 수량
  - RTS_RSLT_HIS: EXEC_TIMEKEY가 PK에 포함되어 같은 회차 재실행도 누적 가능
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
    }
    row.update(overrides)
    return row


def _env_data(**overrides):
    data = {
        "sim_base_time": BASE_TIME,
        "plan": [],
        "eqp_model_map": {"EQP001": "A", "EQP002": "B", "EQP003": "A"},
        "abstract_arrange_map": {("PPK001", "OPER001", "A"): 60},
        "eqp_oper_cap": {"EQP001": ["OPER001"]},
        "plan_meta": {("PPK001", "OPER001"): {"priority": 1, "d0_plan_qty": 75}},
    }
    data.update(overrides)
    return data


def _build(schedule, env_data):
    result = {"algorithm": "earliest_st", "schedule": schedule, "conversion_plans": []}
    return build_rts_output(result, env_data, fac_id="FAC001", rule_timekey="20260719070000")


def test_function_nm_is_always_test():
    payload = _build([_schedule_row()], _env_data())
    assert payload["RTS_RSLT_MAS"][0]["FUNCTION_NM"] == "TEST"


def test_lot_stat_cd_flow_id_wf_qty_st_passthrough():
    payload = _build([_schedule_row(LOT_STAT_CD="PROC", WF_QTY=30, ST=45)], _env_data())
    row = payload["RTS_RSLT_MAS"][0]
    assert row["LOT_STAT_CD"] == "PROC"
    assert row["FLOW_ID"] == "PPK001"
    assert row["WF_QTY"] == 30
    assert row["ST"] == 45


def test_prgs_enable_eqp_lval_falls_back_to_model_match_when_no_discrete_combo():
    # proc_time_matrix가 없는(순수 abstract) 재공 → (PPK,OPER) 모델 매칭으로 대체.
    # EQP001: model A (arrange 매칭) + eqp_oper_cap. EQP003: model A(arrange 매칭)만.
    # EQP002: model B, 매칭도 cap도 없음 → 제외.
    payload = _build([_schedule_row()], _env_data())
    lval = payload["RTS_RSLT_MAS"][0]["PRGS_ENABLE_EQP_LVAL"]
    assert lval == "EQP001,EQP003"


def test_prgs_enable_eqp_lval_uses_discrete_combo_per_lot_when_present():
    # 두 LOT이 같은 (PPK,OPER)를 쓰지만 discrete_arrange로 서로 다른 EQP에 명시돼 있으면
    # PRGS_ENABLE_EQP_LVAL도 LOT마다 그 discrete EQP만 반영해 서로 달라야 한다
    # (abstract 모델 매칭으로 뭉뚱그려 EQP001,EQP002,EQP003이 전부 나오면 안 됨).
    schedule = [
        _schedule_row(EQP_ID="EQP001", LOT_ID="LOT001", CARRIER_ID="CAR001"),
        _schedule_row(EQP_ID="EQP002", LOT_ID="LOT002", CARRIER_ID="CAR002", START_TM=60, END_TM=120),
    ]
    env_data = _env_data(proc_time_matrix={
        ("CAR001", "EQP001", "OPER001"): 60,
        ("CAR002", "EQP002", "OPER001"): 60,
    })
    payload = _build(schedule, env_data)
    rows = {r["CARRIER_ID"]: r for r in payload["RTS_RSLT_MAS"]}
    assert rows["CAR001"]["PRGS_ENABLE_EQP_LVAL"] == "EQP001"
    assert rows["CAR002"]["PRGS_ENABLE_EQP_LVAL"] == "EQP002"


def test_plan_qty_from_plan_meta_d0():
    payload = _build([_schedule_row()], _env_data())
    assert payload["RTS_RSLT_MAS"][0]["PLAN_QTY"] == 75


def test_plan_qty_defaults_to_zero_when_missing():
    payload = _build([_schedule_row()], _env_data(plan_meta={}))
    assert payload["RTS_RSLT_MAS"][0]["PLAN_QTY"] == 0


def test_rslt_his_insert_has_exec_timekey_in_pk_and_columns():
    payload = _build([_schedule_row()], _env_data())
    scripts = build_writer_sql_scripts(payload)
    his_sql = scripts["rts_rslt_his.sql"]
    assert "EXEC_TIMEKEY" in his_sql
    insert_line = next(line for line in his_sql.splitlines() if line.startswith("INSERT INTO"))
    assert "EXEC_TIMEKEY" in insert_line

    mas_sql = scripts["rts_rslt_mas.sql"]
    assert "EXEC_TIMEKEY" not in mas_sql


def test_rslt_mas_and_his_carry_new_columns_in_insert():
    payload = _build([_schedule_row()], _env_data())
    scripts = build_writer_sql_scripts(payload)
    for name in ("rts_rslt_mas.sql", "rts_rslt_his.sql"):
        insert_line = next(
            line for line in scripts[name].splitlines() if line.startswith("INSERT INTO")
        )
        for col in (
            "LOT_STAT_CD", "FLOW_ID", "WF_QTY", "ST",
            "PRGS_ENABLE_EQP_LVAL", "PLAN_QTY", "FUNCTION_NM",
        ):
            assert col in insert_line, f"{col} missing from {name}"
