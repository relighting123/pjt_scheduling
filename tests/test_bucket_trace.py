"""
tests/test_bucket_trace.py

배정 시점에 고를 수 있었던 (PPK,OPER) 버킷 후보 전부(선택된 것 포함) 추적 검증:
  - simulator.SchedulingSimulator._bucket_characteristics(): get_bucket_features()
    스냅샷에서 선택된 (ppk, oper, model) 채널만 뽑아온다
  - simulator.SchedulingSimulator._candidate_bucket_snapshot(): feasible 버킷 전부에
    대해 _bucket_characteristics()를 적용하고 실제 선택된 버킷을 is_selected로 표시
  - assign_ppk_oper()로 성사된 배정의 schedule 행에 CANDIDATE_BUCKETS가 붙는다
  - data/writer: RTS_TRACE_INF/HIS(RTS_EQPCONVPLAN_INF/HIS와 동일 패턴) INSERT 생성
"""
from datetime import datetime

import pytest

from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from data.writer.rts_json import build_rts_output
from data.writer.rts_sql import build_writer_sql_scripts
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"
BASE_TIME = datetime(2026, 7, 19, 7, 0, 0)


@pytest.fixture(scope="module")
def env_data(tmp_path_factory):
    input_dir = tmp_path_factory.mktemp("input")
    generate_sample_data(scenario="default", output_dir=input_dir)
    raw = load_data(input_dir)
    return preprocess(raw, period_key=RULE_TIMEKEY)


def _first_idle_bucket(sim: SchedulingSimulator):
    eqp_id = sim.current_idle_eqp()
    assert eqp_id is not None, "reset 직후 idle EQP가 있어야 함"
    feasible = sim.get_feasible_ppk_oper(eqp_id)
    assert feasible, "idle EQP에 feasible 버킷이 있어야 함"
    ppk, oper_id = sim.ppk_oper_from_flat(feasible[0])
    return eqp_id, ppk, oper_id


# ── _bucket_characteristics ──────────────────────────────────────────────────

def test_bucket_characteristics_returns_expected_keys(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)
    model = sim._eqp_model_map[eqp_id]

    snap = sim._bucket_characteristics(ppk, oper_id, model)
    for key in (
        "wip_share", "urgency", "coverage_ratio", "starve_norm",
        "needs_conv", "avoidable_frac", "setup_changed",
    ):
        assert key in snap
    assert 0.0 <= snap["wip_share"] <= 1.0
    assert isinstance(snap["needs_conv"], bool)


def test_bucket_characteristics_empty_for_unknown_bucket(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    assert sim._bucket_characteristics("__NOPE__", "__NOPE__", "__NOPE__") == {}


# ── assign_ppk_oper() → schedule row (CANDIDATE_BUCKETS) ───────────────────

def test_successful_assignment_carries_candidate_buckets(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)

    reward = sim.assign_ppk_oper(eqp_id, ppk, oper_id)
    assert reward != -1.0
    assert sim.schedule, "성공한 배정은 schedule에 행을 남겨야 함"

    candidates = sim.schedule[-1]["CANDIDATE_BUCKETS"]
    assert candidates, "feasible 버킷이 있었으니 후보 목록도 비어있으면 안 됨"
    selected = [c for c in candidates if c["is_selected"]]
    assert len(selected) == 1
    assert selected[0]["ppk"] == ppk and selected[0]["oper_id"] == oper_id
    for c in candidates:
        assert "wip_share" in c and "coverage_ratio" in c
    assert all(isinstance(c["is_selected"], bool) for c in candidates)


def test_candidate_bucket_characteristics_match_bucket_characteristics(env_data):
    """후보 목록의 특성값은 _bucket_characteristics()로 직접 계산한 값과 같아야 함."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id = sim.current_idle_eqp()
    feasible = sim.get_feasible_ppk_oper(eqp_id)
    model = sim._eqp_model_map[eqp_id]
    expected = {}
    for flat in feasible:
        p, o = sim.ppk_oper_from_flat(flat)
        expected[(p, o)] = sim._bucket_characteristics(p, o, model)

    ppk, oper_id = sim.ppk_oper_from_flat(feasible[0])
    sim.assign_ppk_oper(eqp_id, ppk, oper_id)
    candidates = sim.schedule[-1]["CANDIDATE_BUCKETS"]

    for c in candidates:
        key = (c["ppk"], c["oper_id"])
        if key in expected:
            assert c["wip_share"] == pytest.approx(expected[key]["wip_share"])
            assert c["coverage_ratio"] == pytest.approx(expected[key]["coverage_ratio"])


def test_forced_replay_has_no_candidate_buckets(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id = next(iter(sim.eqps))
    row = {
        "lot_id": "FORCED_LOT", "PLAN_PROD_ATTR_VAL": "PPK_X", "oper_id": "OP_X",
        "wf_qty": 25, "carrier_id": "CAR_X", "logical_lot_id": "FORCED_LOT",
        "start_min": 0, "end_min": 60, "seq": 1,
    }
    sim._activate_fixed_queue_row(eqp_id, row)

    rec = sim.schedule[-1]
    assert rec["CANDIDATE_BUCKETS"] == []


# ── data/writer: RTS_TRACE_INF/HIS(다른 선택 가능 버킷 이력) ──────────────────

def _schedule_row(**overrides):
    row = {
        "EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
        "START_TM": 0, "END_TM": 60, "WF_QTY": 25, "ST": 60,
        "LOT_CD": "LC_A", "TEMP": "T600", "EQP_MODEL": "A",
        "LOT_STAT_CD": "WAIT", "SEQ": 1,
        "CANDIDATE_BUCKETS": [
            {
                "ppk": "PPK001", "oper_id": "OPER001", "is_selected": True,
                "wip_share": 0.5, "urgency": 0.25, "coverage_ratio": 0.75,
                "starve_norm": 1.0, "needs_conv": False, "avoidable_frac": 0.0,
                "setup_changed": True,
            },
            {
                "ppk": "PPK002", "oper_id": "OPER001", "is_selected": False,
                "wip_share": 0.2, "urgency": 0.1, "coverage_ratio": 0.4,
                "starve_norm": 0.5, "needs_conv": True, "avoidable_frac": 0.3,
                "setup_changed": False,
            },
        ],
    }
    row.update(overrides)
    return row


def _writer_env_data(**overrides):
    data = {
        "sim_base_time": BASE_TIME,
        "plan": [],
        "eqp_model_map": {"EQP001": "A"},
        "abstract_arrange_map": {("PPK001", "OPER001", "A"): 60},
        "eqp_oper_cap": {"EQP001": ["OPER001"]},
        "plan_meta": {("PPK001", "OPER001"): {"priority": 1, "d0_plan_qty": 75}},
    }
    data.update(overrides)
    return data


def test_rts_rslt_mas_no_longer_carries_bucket_fields():
    """RTS_RSLT_MAS/HIS는 원래 컬럼만 유지하고, 후보/사유 관련 컬럼은 붙지 않아야 함."""
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    row = payload["RTS_RSLT_MAS"][0]
    for col in (
        "SELECT_REASON_CD", "SELECT_REASON_CTN",
        "SIZE_SELECT_REASON_CD", "SIZE_SELECT_REASON_CTN",
        "BUCKET_WIP_SHARE", "BUCKET_URGENCY", "BUCKET_COVERAGE_RATIO",
        "BUCKET_STARVE_NORM", "BUCKET_NEEDS_CONV", "BUCKET_AVOIDABLE_FRAC",
        "BUCKET_SETUP_CHANGED",
    ):
        assert col not in row


def test_trace_rows_include_selected_and_unselected_buckets():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    rows = payload["RTS_TRACE_INF"]

    assert len(rows) == 2
    by_ppk = {r["CANDIDATE_PPK"]: r for r in rows}
    assert by_ppk["PPK001"]["IS_SELECTED"] is True
    assert by_ppk["PPK002"]["IS_SELECTED"] is False
    assert by_ppk["PPK002"]["WIP_SHARE"] == 0.2
    # 부모 배정 행(EQP_ID/CARRIER_ID)으로 RTS_RSLT_MAS와 연결 가능해야 함
    assert by_ppk["PPK001"]["EQP_ID"] == "EQP001"
    assert by_ppk["PPK001"]["CARRIER_ID"] == "CAR001"


def test_trace_rows_empty_when_no_candidates_recorded():
    row = _schedule_row(CANDIDATE_BUCKETS=[])
    result = {"algorithm": "earliest_st", "schedule": [row], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    assert payload["RTS_TRACE_INF"] == []


def test_trace_rows_empty_when_output_disabled():
    from config import CONFIG
    original = CONFIG.env.trace_output_enabled
    try:
        CONFIG.env.trace_output_enabled = False
        result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
        payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
        assert payload["RTS_TRACE_INF"] == []
    finally:
        CONFIG.env.trace_output_enabled = original


def test_trace_inf_and_his_insert_sql_include_expected_columns_and_values():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload)

    assert "rts_trace_inf.sql" in scripts and "rts_trace_his.sql" in scripts
    for name, table in (("rts_trace_inf.sql", "RTS_TRACE_INF"), ("rts_trace_his.sql", "RTS_TRACE_HIS")):
        insert_lines = [
            line for line in scripts[name].splitlines()
            if line.startswith(f"INSERT INTO {table}")
        ]
        assert len(insert_lines) == 2, name
        for col in (
            "FAC_ID", "RULE_TIMEKEY", "EQP_ID", "CARRIER_ID",
            "CANDIDATE_PPK", "CANDIDATE_OPER_ID", "IS_SELECTED",
            "WIP_SHARE", "URGENCY", "COVERAGE_RATIO", "STARVE_NORM",
            "NEEDS_CONV", "AVOIDABLE_FRAC", "SETUP_CHANGED", "EXEC_TIMEKEY",
        ):
            assert col in insert_lines[0], f"{col} missing from {name}"
        assert any("'Y'" in line for line in insert_lines)  # 선택된 후보
        assert any("'N'" in line for line in insert_lines)  # 선택되지 않은 후보


def test_trace_inf_generated_unconditionally_but_his_omitted_without_history():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload, include_history=False)
    assert "rts_trace_inf.sql" in scripts
    assert "rts_trace_his.sql" not in scripts


def test_trace_inf_deletes_same_rule_timekey_before_insert():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload)
    assert "DELETE FROM RTS_TRACE_INF" in scripts["rts_trace_inf.sql"]
    assert "FAC001" in scripts["rts_trace_inf.sql"]
    assert "20260719070000" in scripts["rts_trace_inf.sql"]
