"""
tests/test_rts_sql_delete_all.py

RTS_RSLT_INF/RTS_EQPCONVPLAN_INF은 매 회차 전체를 DELETE한 뒤 INSERT하여
항상 최신 결과만 남아야 한다(RULE_TIMEKEY 단위 부분 삭제가 아님).
"""
from data.writer.rts_sql import build_writer_sql_scripts

META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "CRT_USER_ID": "RTS"}


def _payload(rule_timekey: str = "20260715070000") -> dict:
    return {
        "meta": {**META, "RULE_TIMEKEY": rule_timekey},
        "RTS_RSLT_INF": [],
        "RTS_EQPCONVPLAN_INF": [],
    }


def test_rslt_inf_delete_has_no_where_clause():
    scripts = build_writer_sql_scripts(_payload())
    assert "DELETE FROM RTS_RSLT_INF;" in scripts["rts_rslt_inf.sql"]
    assert "WHERE" not in scripts["rts_rslt_inf.sql"]


def test_eqpconvplan_inf_delete_has_no_where_clause():
    scripts = build_writer_sql_scripts(_payload())
    assert "DELETE FROM RTS_EQPCONVPLAN_INF;" in scripts["rts_eqpconvplan_inf.sql"]
    assert "WHERE" not in scripts["rts_eqpconvplan_inf.sql"]


def test_delete_all_present_even_without_rule_timekey():
    scripts = build_writer_sql_scripts(_payload(rule_timekey=""))
    assert "DELETE FROM RTS_RSLT_INF;" in scripts["rts_rslt_inf.sql"]
    assert "DELETE FROM RTS_EQPCONVPLAN_INF;" in scripts["rts_eqpconvplan_inf.sql"]


def test_his_scripts_have_no_delete():
    scripts = build_writer_sql_scripts(_payload())
    assert "DELETE" not in scripts["rts_rslt_his.sql"]
    assert "DELETE" not in scripts["rts_eqpconvplan_his.sql"]
