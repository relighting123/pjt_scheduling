"""
tests/test_rts_sql_delete_all.py

RTS_RSLT_MAS는 매 회차 동일 FAC_ID의 기존 행을 모두 DELETE한 뒤 INSERT하여 항상
최신 결과만 남아야 한다(RULE_TIMEKEY 무관, 다른 FAC_ID의 데이터는 건드리지 않는다).

RTS_EQPCONVPLAN_INF는 동일 FAC_ID+RULE_TIMEKEY 기존 행만 DELETE한 뒤 INSERT한다
(같은 회차 재실행 시 JOB_ID 중복/PK 위반을 막기 위함) — 다른 RULE_TIMEKEY의
행은 남겨두고 계속 누적된다. HIS 테이블들은 삭제 없이 INSERT만 한다.
"""
from data.writer.rts_sql import build_writer_sql_scripts

META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "CRT_USER_ID": "RTS"}


def _payload(rule_timekey: str = "20260715070000", fac_id: str = "FAC001") -> dict:
    return {
        "meta": {**META, "RULE_TIMEKEY": rule_timekey, "FAC_ID": fac_id},
        "RTS_RSLT_MAS": [],
        "RTS_EQPCONVPLAN_INF": [],
    }


def test_rslt_mas_delete_scoped_by_fac_id_only():
    scripts = build_writer_sql_scripts(_payload())
    assert "DELETE FROM RTS_RSLT_MAS WHERE FAC_ID = 'FAC001';" in scripts["rts_rslt_mas.sql"]


def test_rslt_mas_delete_scoped_to_own_fac_id_leaves_others_untouched():
    scripts = build_writer_sql_scripts(_payload(fac_id="FAC002"))
    assert "DELETE FROM RTS_RSLT_MAS WHERE FAC_ID = 'FAC002';" in scripts["rts_rslt_mas.sql"]
    assert "FAC001" not in scripts["rts_rslt_mas.sql"]


def test_rslt_mas_delete_present_even_without_rule_timekey():
    scripts = build_writer_sql_scripts(_payload(rule_timekey=""))
    assert "DELETE FROM RTS_RSLT_MAS WHERE FAC_ID = 'FAC001';" in scripts["rts_rslt_mas.sql"]


def test_eqpconvplan_inf_delete_scoped_by_fac_id_and_rule_timekey():
    scripts = build_writer_sql_scripts(_payload())
    assert (
        "DELETE FROM RTS_EQPCONVPLAN_INF WHERE FAC_ID = 'FAC001' "
        "AND RULE_TIMEKEY = '20260715070000';"
    ) in scripts["rts_eqpconvplan_inf.sql"]


def test_eqpconvplan_inf_delete_scoped_to_own_rule_timekey_only():
    scripts = build_writer_sql_scripts(_payload(rule_timekey="20260716070000"))
    sql = scripts["rts_eqpconvplan_inf.sql"]
    assert "RULE_TIMEKEY = '20260716070000'" in sql
    assert "20260715070000" not in sql


def test_eqpconvplan_inf_no_delete_without_rule_timekey():
    scripts = build_writer_sql_scripts(_payload(rule_timekey=""))
    assert "DELETE" not in scripts["rts_eqpconvplan_inf.sql"]


def test_his_scripts_have_no_delete():
    scripts = build_writer_sql_scripts(_payload())
    assert "DELETE" not in scripts["rts_rslt_his.sql"]
    assert "DELETE" not in scripts["rts_eqpconvplan_his.sql"]
