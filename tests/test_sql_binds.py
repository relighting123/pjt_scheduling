"""SQL fetch 바인드 (LOT_CD) 테스트."""

from data.loader.sql_binds import merge_fetch_binds, resolve_lot_cd


def test_resolve_lot_cd_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("SQL_LOT_CD", "LC_ENV")
    monkeypatch.setenv("COLLECTOR_LOT_CD", "LC_COLL")
    assert resolve_lot_cd("LC_CLI") == "LC_CLI"


def test_resolve_lot_cd_env_priority(monkeypatch):
    monkeypatch.delenv("SQL_LOT_CD", raising=False)
    monkeypatch.setenv("COLLECTOR_LOT_CD", "LC_COLL")
    assert resolve_lot_cd() == "LC_COLL"
    monkeypatch.setenv("SQL_LOT_CD", "LC_SQL")
    assert resolve_lot_cd() == "LC_SQL"


def test_resolve_lot_cd_empty_is_none(monkeypatch):
    monkeypatch.delenv("SQL_LOT_CD", raising=False)
    monkeypatch.delenv("COLLECTOR_LOT_CD", raising=False)
    assert resolve_lot_cd("") is None
    assert resolve_lot_cd() is None


def test_merge_fetch_binds_includes_lot_cd():
    binds = merge_fetch_binds("FAC001", "20260621170000", lot_cd="LC001")
    assert binds["FAC_ID"] == "FAC001"
    assert binds["RULE_TIMEKEY"] == "20260621170000"
    assert binds["LOT_CD"] == "LC001"


def test_merge_fetch_binds_null_lot_cd_when_unset(monkeypatch):
    monkeypatch.delenv("SQL_LOT_CD", raising=False)
    monkeypatch.delenv("COLLECTOR_LOT_CD", raising=False)
    binds = merge_fetch_binds("FAC001", "20260621170000")
    assert binds["LOT_CD"] is None
