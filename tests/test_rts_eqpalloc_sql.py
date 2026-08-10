"""
tests/test_rts_eqpalloc_sql.py

RTS_EQPALLOC_PLAN/HIS SQL 생성 — PPK/OPER × EQP_MODEL_CD 사전 할당 결과의 DB
적재용 스크립트. RTS_EQPCONVPLAN_INF/HIS와 동일 원칙: PLAN은 동일 FAC_ID+
RULE_TIMEKEY 기존 행만 DELETE 후 INSERT(같은 회차 재실행 시 PK 충돌 방지),
HIS는 삭제 없이 INSERT만 누적한다.
"""
from data.writer.rts_sql import build_writer_sql_scripts

META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260810070000", "CRT_USER_ID": "RTS"}


def _alloc_row(**overrides) -> dict:
    row = {
        "FAC_ID": "FAC001",
        "RULE_TIMEKEY": "20260810070000",
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "OPER_ID": "OPER001",
        "EQP_MODEL_CD": "M1",
        "PLAN_PRIORITY": 1,
        "IS_EXCLUSIVE_MODEL": "Y",
        "WINDOW_START_TM": "20260810070000",
        "WINDOW_END_TM": "20260811050000",
        "WINDOW_MINUTES": 1320,
        "PLAN_QTY": 100,
        "WIP_QTY": 200,
        "TARGET_QTY": 100,
        "ST": 60,
        "CAPA_PER_EQP": 22.0,
        "ALLOC_EQP_CNT": 5,
        "ALLOC_CAPA_QTY": 110.0,
        "TOTAL_EQP_CNT": 10,
    }
    row.update(overrides)
    return row


def _payload(rule_timekey: str = "20260810070000", fac_id: str = "FAC001", alloc_rows=None) -> dict:
    return {
        "meta": {**META, "RULE_TIMEKEY": rule_timekey, "FAC_ID": fac_id},
        "RTS_RSLT_MAS": [],
        "RTS_EQPCONVPLAN_INF": [],
        "RTS_EQPALLOC_PLAN": alloc_rows if alloc_rows is not None else [_alloc_row()],
    }


def test_eqpalloc_plan_delete_scoped_by_fac_id_and_rule_timekey():
    scripts = build_writer_sql_scripts(_payload())
    assert (
        "DELETE FROM RTS_EQPALLOC_PLAN WHERE FAC_ID = 'FAC001' "
        "AND RULE_TIMEKEY = '20260810070000';"
    ) in scripts["rts_eqpalloc_plan.sql"]


def test_eqpalloc_plan_insert_contains_row_values():
    scripts = build_writer_sql_scripts(_payload())
    sql = scripts["rts_eqpalloc_plan.sql"]
    assert "INSERT INTO RTS_EQPALLOC_PLAN" in sql
    assert "'PPK001'" in sql
    assert "'OPER001'" in sql
    assert "'M1'" in sql


def test_eqpalloc_plan_no_delete_without_rule_timekey():
    scripts = build_writer_sql_scripts(_payload(rule_timekey=""))
    assert "DELETE" not in scripts["rts_eqpalloc_plan.sql"]


def test_eqpalloc_his_has_no_delete_and_carries_exec_timekey():
    scripts = build_writer_sql_scripts(_payload())
    sql = scripts["rts_eqpalloc_his.sql"]
    assert "DELETE" not in sql
    assert "INSERT INTO RTS_EQPALLOC_HIS" in sql
    assert "EXEC_TIMEKEY" in sql


def test_eqpalloc_plan_empty_rows_still_produces_delete_only_script():
    # 할당 대상이 없던 회차(계획/재공 없음 등)도 DELETE는 실행되어야
    # 예전 회차의 stale 행이 그대로 남지 않는다.
    scripts = build_writer_sql_scripts(_payload(alloc_rows=[]))
    sql = scripts["rts_eqpalloc_plan.sql"]
    assert "DELETE FROM RTS_EQPALLOC_PLAN" in sql
    assert "INSERT" not in sql
