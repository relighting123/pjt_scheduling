"""SQL fetch 바인드 테스트."""

from data.loader.sql_binds import merge_fetch_binds


def test_merge_fetch_binds_fac_and_period():
    binds = merge_fetch_binds("FAC001", "20260621170000")
    assert binds["FAC_ID"] == "FAC001"
    assert binds["RULE_TIMEKEY"] == "20260621170000"
    assert "LOT_CD" not in binds


def test_merge_fetch_binds_period_optional():
    binds = merge_fetch_binds("FAC001")
    assert binds == {"FAC_ID": "FAC001"}
