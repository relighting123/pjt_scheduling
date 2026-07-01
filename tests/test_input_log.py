"""추론 input fetch 시 logging/input_log.txt 생성 테스트."""
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data.loader.fetch import (
    _append_input_log_entry,
    _format_sql_log_entry,
    _init_input_log,
    _inline_binds,
    fetch_from_db,
)


def test_inline_binds_substitutes_oracle_params():
    sql = "SELECT * FROM t WHERE fac = :FAC_ID AND tk = :RULE_TIMEKEY AND lot = :LOT_CD"
    binds = {
        "FAC_ID": "FAC001",
        "RULE_TIMEKEY": "20260621170000",
        "LOT_CD": None,
    }
    result = _inline_binds(sql, binds)
    assert result == (
        "SELECT * FROM t WHERE fac = 'FAC001' AND tk = '20260621170000' AND lot = NULL"
    )


def test_inline_binds_escapes_single_quotes():
    sql = "SELECT * FROM t WHERE lot = :LOT_CD"
    result = _inline_binds(sql, {"LOT_CD": "O'Brien"})
    assert result == "SELECT * FROM t WHERE lot = 'O''Brien'"


def test_format_sql_log_entry_includes_metadata_and_runnable_sql():
    entry = _format_sql_log_entry(
        "plan.sql",
        "Prd",
        "20260621170000",
        {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260621170000"},
        "SELECT 1 FROM dual WHERE fac = :FAC_ID",
        3,
    )
    assert "[plan.sql] @Prd" in entry
    assert "period=20260621170000" in entry
    assert "rows=3" in entry
    assert "fac = 'FAC001'" in entry


def test_init_and_append_input_log(tmp_path, monkeypatch):
    monkeypatch.setattr("config.BASE_DIR", tmp_path)

    _init_input_log("FAC001", "infer", "20260621170000")
    log_path = tmp_path / "logging" / "input_log.txt"
    assert log_path.exists()
    header = log_path.read_text(encoding="utf-8")
    assert "FAC_ID=FAC001" in header
    assert "split=infer" in header
    assert "RULE_TIMEKEY=20260621170000" in header

    _append_input_log_entry(
        "plan.sql",
        "Prd",
        "20260621170000",
        {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260621170000"},
        "SELECT 1 FROM dual WHERE fac = :FAC_ID",
        1,
    )
    content = log_path.read_text(encoding="utf-8")
    assert "[plan.sql] @Prd" in content
    assert "fac = 'FAC001'" in content


def _setup_fetch_fixtures(tmp_path: Path) -> Path:
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        textwrap.dedent("""
        default: WT_RTS
        WT_RTS:
          user: u
          password: p
          dsn: d
        """),
        encoding="utf-8",
    )
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    for name in (
        "discrete_arrange.sql",
        "abstract_arrange.sql",
        "plan.sql",
        "flow.sql",
        "split.sql",
        "batch_info.sql",
    ):
        (sql_dir / name).write_text(
            f"@db:WT_RTS\nSELECT '{name}' AS src WHERE fac = :FAC_ID AND tk = :RULE_TIMEKEY",
            encoding="utf-8",
        )
    return sql_dir


def test_fetch_infer_writes_input_log(tmp_path, monkeypatch):
    sql_dir = _setup_fetch_fixtures(tmp_path)
    monkeypatch.setenv("DB_CONFIG", str(tmp_path / "databases.yaml"))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)
    monkeypatch.setattr("config.BASE_DIR", tmp_path)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("SRC",)]
    mock_cursor.fetchall.return_value = [("plan.sql",)]
    mock_conn.cursor.return_value = mock_cursor

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "data.db_registry.DbRegistry.connect",
            lambda self, alias=None: mock_conn,
        )
        fetch_from_db(
            fac_id="FAC001",
            split="infer",
            period="20260621170000",
            lot_cd="LC001",
        )

    log_path = tmp_path / "logging" / "input_log.txt"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "FAC_ID=FAC001" in content
    assert "split=infer" in content
    assert "fac = 'FAC001'" in content
    assert "tk = '20260621170000'" in content
    assert "[discrete_arrange.sql]" in content
    assert "[plan.sql]" in content


def test_fetch_train_does_not_write_input_log(tmp_path, monkeypatch):
    sql_dir = _setup_fetch_fixtures(tmp_path)
    monkeypatch.setenv("DB_CONFIG", str(tmp_path / "databases.yaml"))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)
    monkeypatch.setattr("config.BASE_DIR", tmp_path)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("SRC",)]
    mock_cursor.fetchall.return_value = [("plan.sql",)]
    mock_conn.cursor.return_value = mock_cursor

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "data.db_registry.DbRegistry.connect",
            lambda self, alias=None: mock_conn,
        )
        fetch_from_db(
            fac_id="FAC001",
            split="train",
            period="20260621170000",
        )

    assert not (tmp_path / "logging" / "input_log.txt").exists()
