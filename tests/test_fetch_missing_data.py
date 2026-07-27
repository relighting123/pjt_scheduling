"""collect 시 필수 테이블 0건/필수 필드 누락을 감지해 실패 처리하는지 검증."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.loader.fetch import _missing_data_errors, fetch_from_db


def _valid_collected() -> dict:
    return {
        "discrete_arrange": [
            {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "PLAN_PROD_ATTR_VAL": "PPK001",
             "OPER_ID": "OPER001", "ST": 60, "EQP_MODEL_CD": "A", "WF_QTY": 1},
        ],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
        ],
        "plan": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "D0_PLAN_QTY": 10, "D1_PLAN_QTY": 10, "PLAN_PRIORITY": 1},
        ],
        "flow": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
        ],
        "split": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "EQP_MODEL_CD": "A", "SPLIT_QTY": 1},
        ],
        "batch_info": [
            {"LOT_CD": "LC_A", "TEMP": "T600", "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001"},
        ],
    }


def test_missing_data_errors_empty_when_all_required_present():
    assert _missing_data_errors(_valid_collected()) == []


def test_missing_data_errors_flags_empty_required_table():
    collected = _valid_collected()
    collected["plan"] = []  # 필수 테이블이 0건으로 돌아온 경우
    errors = _missing_data_errors(collected)
    assert any("plan" in e for e in errors)


def test_missing_data_errors_flags_missing_required_field():
    collected = _valid_collected()
    del collected["plan"][0]["PLAN_PRIORITY"]  # 필수 필드 누락
    errors = _missing_data_errors(collected)
    assert any("plan" in e for e in errors)


def test_missing_data_errors_flags_missing_key_entirely():
    collected = _valid_collected()
    del collected["flow"]  # 테이블 자체가 collected 에 없는 경우도 누락으로 처리
    errors = _missing_data_errors(collected)
    assert any("flow" in e for e in errors)


class _FakeRegistry:
    """DB 연결 없이 fetch_from_db 의 수집 루프만 검증하기 위한 가짜 registry."""

    def __init__(self, rows_by_sql: dict):
        self.default_alias = "main"
        self.default_warn = None
        self._rows_by_sql = rows_by_sql

    def connect(self, alias):
        return MagicMock()

    def get_credentials(self, alias):
        return {}

    def close_all(self):
        pass


def test_fetch_from_db_fails_and_cleans_up_dir_when_required_table_empty(tmp_path, monkeypatch):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    from config import SQL_JSON_MAP

    for key, (sql_file, _json_file) in SQL_JSON_MAP.items():
        (sql_dir / sql_file).write_text("SELECT 1 FROM DUAL", encoding="utf-8")

    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    output_dir = tmp_path / "out"
    assert not output_dir.exists()

    # plan 만 0건, 나머지는 최소 1건씩 반환하도록 _execute_query 를 패치
    def fake_execute_query(conn, sql, binds):
        return []  # 모든 쿼리가 0건 → plan 등 필수 테이블 누락으로 실패해야 함

    with patch("data.loader.fetch._execute_query", side_effect=fake_execute_query), \
         patch("data.loader.fetch.parse_sql_db_alias", return_value="main"):
        with pytest.raises(RuntimeError, match="수집 데이터 누락"):
            fetch_from_db(
                fac_id="FAC001",
                output_dir=output_dir,
                split="train",
                period="20260101070000",
                db_registry=_FakeRegistry({}),
            )

    # 실패 시 이번에 새로 만든 출력 폴더는 남지 않아야 한다
    assert not output_dir.exists()
