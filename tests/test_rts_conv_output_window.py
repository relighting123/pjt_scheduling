"""
tests/test_rts_conv_output_window.py

RTS_EQPCONVPLAN_INF/HIS 출력은 RULE_TIMEKEY 기준
CONFIG.env.conv_output_window_minutes 이내에 시작하는 전환만 포함해야 한다.
먼 미래의 전환은 재계획 여지가 커 추측성이므로 확정 출력에서 제외된다.
"""
from datetime import datetime

import pytest

from config import CONFIG
from data.writer.rts_json import _build_rts_conv_rows, build_rts_output
from data.writer.rts_sql import build_writer_sql_scripts

BASE_TIME = datetime(2026, 7, 15, 7, 0, 0)
META = {"FAC_ID": "FAC001", "RULE_TIMEKEY": "20260715070000", "CRT_USER_ID": "RTS"}


def _conv(eqp_id: str, start_min: int) -> dict:
    return {
        "eqp_id": eqp_id,
        "conv_start_min": start_min,
        "conv_end_min": start_min + 60,
        "conv_time": 60,
        "from_lot_cd": "LC_A",
        "to_lot_cd": "LC_B",
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "oper_id": "OPER001",
        "eqp_model_cd": "A",
    }


@pytest.fixture(autouse=True)
def _restore_window():
    original = CONFIG.env.conv_output_window_minutes
    yield
    CONFIG.env.conv_output_window_minutes = original


def test_default_window_is_60_minutes():
    assert CONFIG.env.conv_output_window_minutes == 60


def test_excludes_conversions_starting_after_window():
    CONFIG.env.conv_output_window_minutes = 60
    plans = [_conv("EQP001", 0), _conv("EQP002", 60), _conv("EQP003", 61), _conv("EQP004", 600)]
    rows = _build_rts_conv_rows(plans, META, BASE_TIME)
    assert [r["EQP_ID"] for r in rows] == ["EQP001", "EQP002"]


def test_window_is_configurable():
    CONFIG.env.conv_output_window_minutes = 120
    plans = [_conv("EQP001", 0), _conv("EQP002", 90), _conv("EQP003", 121)]
    rows = _build_rts_conv_rows(plans, META, BASE_TIME)
    assert [r["EQP_ID"] for r in rows] == ["EQP001", "EQP002"]


def test_full_pipeline_json_and_sql_respect_window():
    CONFIG.env.conv_output_window_minutes = 60
    env_data = {"sim_base_time": BASE_TIME, "plan": []}
    result = {
        "algorithm": "earliest_st",
        "schedule": [],
        "conversion_plans": [_conv("EQP001", 30), _conv("EQP002", 300)],
    }
    payload = build_rts_output(result, env_data, fac_id="FAC001", rule_timekey="20260715070000")
    assert [r["EQP_ID"] for r in payload["RTS_EQPCONVPLAN_INF"]] == ["EQP001"]

    scripts = build_writer_sql_scripts(payload)
    assert scripts["rts_eqpconvplan_inf.sql"].count("INSERT INTO") == 1
    assert scripts["rts_eqpconvplan_his.sql"].count("INSERT INTO") == 1
    assert "EQP002" not in scripts["rts_eqpconvplan_inf.sql"]


def test_raw_conversion_plans_unaffected():
    """result['conversion_plans']는 API/UI용 원본이라 필터링과 무관하게 그대로 유지된다."""
    CONFIG.env.conv_output_window_minutes = 60
    env_data = {"sim_base_time": BASE_TIME, "plan": []}
    result = {
        "algorithm": "earliest_st",
        "schedule": [],
        "conversion_plans": [_conv("EQP001", 30), _conv("EQP002", 300)],
    }
    build_rts_output(result, env_data, fac_id="FAC001", rule_timekey="20260715070000")
    assert len(result["conversion_plans"]) == 2
