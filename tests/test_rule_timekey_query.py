"""RULE_TIMEKEY DB 메타 SQL 테스트."""
import textwrap
from pathlib import Path

import pytest

from data.loader import rule_timekey_query as rtq


def _write_meta_sql(tmp_path: Path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "rule_timekey_latest.sql").write_text(
        "@db:WT_RTS\nSELECT :FAC_ID AS RULE_TIMEKEY FROM dual",
        encoding="utf-8",
    )
    (sql_dir / "rule_timekey_list.sql").write_text(
        "@db:WT_RTS\nSELECT :FROM_RULE_TIMEKEY AS RULE_TIMEKEY FROM dual",
        encoding="utf-8",
    )
    (sql_dir / "rule_timekey_recent.sql").write_text(
        "@db:WT_RTS\nSELECT :FAC_ID AS RULE_TIMEKEY FROM dual",
        encoding="utf-8",
    )
    return sql_dir


def test_use_db_rule_timekey_auto(monkeypatch, tmp_path):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "rule_timekey_latest.sql").write_text("SELECT 1", encoding="utf-8")
    monkeypatch.delenv("RULE_TIMEKEY_FROM_DB", raising=False)
    monkeypatch.setattr("config.SQL_DIR", sql_dir)
    assert rtq.use_db_rule_timekey() is True


def test_use_db_rule_timekey_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "0")
    monkeypatch.setattr("config.SQL_DIR", tmp_path / "missing")
    assert rtq.use_db_rule_timekey() is False


def test_fetch_latest_rule_timekey(monkeypatch, tmp_path):
    sql_dir = _write_meta_sql(tmp_path)
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "1")
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    def fake_execute(sql_filename, binds, db_registry=None):
        if sql_filename == "rule_timekey_latest.sql":
            return [{"RULE_TIMEKEY": "20260620070000"}]
        return []

    monkeypatch.setattr(rtq, "execute_meta_sql", fake_execute)
    assert rtq.fetch_latest_rule_timekey("FAC001") == "20260620070000"


def test_fetch_rule_timekey_list(monkeypatch, tmp_path):
    sql_dir = _write_meta_sql(tmp_path)
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "1")
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    def fake_execute(sql_filename, binds, db_registry=None):
        if sql_filename == "rule_timekey_list.sql":
            return [
                {"RULE_TIMEKEY": "20260620070000"},
                {"RULE_TIMEKEY": "20260621070000"},
            ]
        return []

    monkeypatch.setattr(rtq, "execute_meta_sql", fake_execute)
    keys = rtq.fetch_rule_timekey_list(
        "FAC001", "20260620070000", "20260622070000",
    )
    assert keys == ["20260620070000", "20260621070000"]


def test_resolve_collect_periods_db_recent(monkeypatch, tmp_path):
    sql_dir = _write_meta_sql(tmp_path)
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "1")
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    def fake_execute(sql_filename, binds, db_registry=None):
        if sql_filename == "rule_timekey_recent.sql":
            return [
                {"RULE_TIMEKEY": "20260621070000"},
                {"RULE_TIMEKEY": "20260620070000"},
            ]
        return []

    monkeypatch.setattr(rtq, "execute_meta_sql", fake_execute)
    periods, source = rtq.resolve_collect_periods("FAC001", prevdays=2)
    assert source == "db"
    assert periods == ["20260620070000", "20260621070000"]


def test_resolve_collect_periods_require_db_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "0")
    monkeypatch.setattr("config.SQL_DIR", tmp_path / "empty")
    with pytest.raises(ValueError, match="DB RULE_TIMEKEY"):
        rtq.resolve_collect_periods("FAC001", prevdays=1, require_db=True)


def test_resolve_collect_periods_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "0")
    monkeypatch.setattr("config.SQL_DIR", tmp_path / "empty")
    periods, source = rtq.resolve_collect_periods("FAC001", prevdays=1)
    assert source == "local"
    assert len(periods) == 1


def test_resolve_snapshot_rule_timekey_db(monkeypatch, tmp_path):
    sql_dir = _write_meta_sql(tmp_path)
    monkeypatch.setenv("RULE_TIMEKEY_FROM_DB", "1")
    monkeypatch.setattr("config.SQL_DIR", sql_dir)
    monkeypatch.setattr(
        rtq,
        "fetch_latest_rule_timekey",
        lambda fac_id, db_registry=None: "20260620070000",
    )
    key, source = rtq.resolve_snapshot_rule_timekey("FAC001")
    assert key == "20260620070000"
    assert source == "db"
