"""
tests/test_bucket_selection_reason.py

버킷(PPK/OPER) 선택 및 블록 크기 선택 사유 + 선택 시점 버킷 특성 적재 검증:
  - simulator._select_reason_from_terms(): reward_breakdown → 사유 코드/설명
  - simulator.SchedulingSimulator._bucket_characteristics(): get_bucket_features()
    스냅샷에서 선택된 (ppk, oper, model) 채널만 뽑아온다
  - assign_ppk_oper()로 성사된 배정의 schedule 행에 SELECT_REASON_*/BUCKET_* 필드가 붙는다
  - env.scheduling_rl_env._size_select_reason(): 블록 크기 산출 근거 → 사유 코드/설명
  - SchedulingRLEnv 블록 시작/연속 스텝이 SIZE_SELECT_REASON_CD를 채운다
  - data/writer: RTS_RSLT_MAS/HIS INSERT에 새 컬럼이 포함된다
"""
from datetime import datetime

import numpy as np
import pytest

from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from data.writer.rts_json import build_rts_output
from data.writer.rts_sql import build_writer_sql_scripts
from env.scheduling_rl_env import SchedulingRLEnv, _size_select_reason
from simulation.simulator import SchedulingSimulator, _select_reason_from_terms

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


# ── _select_reason_from_terms ────────────────────────────────────────────────

def test_select_reason_neutral_when_no_terms():
    cd, ctn = _select_reason_from_terms({})
    assert cd == "NEUTRAL"


def test_select_reason_neutral_when_all_zero():
    cd, ctn = _select_reason_from_terms({"same_setup": 0.0, "pacing": 0.0})
    assert cd == "NEUTRAL"


def test_select_reason_picks_dominant_by_abs_value():
    cd, ctn = _select_reason_from_terms({"same_setup": 0.1, "pacing": -0.9})
    assert cd == "PACING"
    assert "pacing" in ctn and "same_setup" in ctn


def test_select_reason_maps_known_terms_to_fixed_codes():
    assert _select_reason_from_terms({"sametool_setup": -0.3})[0] == "TOOL_KEPT_SETUP_CHG"
    assert _select_reason_from_terms({"flow_balance": 0.2})[0] == "FLOW_BALANCE"
    assert _select_reason_from_terms({"avoidable_conversion": -0.5})[0] == "AVOIDABLE_CONV"


def test_select_reason_falls_back_to_uppercased_key_for_unknown_term():
    cd, _ = _select_reason_from_terms({"some_new_term": 1.0})
    assert cd == "SOME_NEW_TERM"


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


# ── assign_ppk_oper() → schedule row ────────────────────────────────────────

def test_successful_assignment_carries_reason_and_bucket_fields(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)

    reward = sim.assign_ppk_oper(eqp_id, ppk, oper_id)
    assert reward != -1.0
    assert sim.schedule, "성공한 배정은 schedule에 행을 남겨야 함"

    row = sim.schedule[-1]
    assert row["SELECT_REASON_CD"]
    assert row["SELECT_REASON_CTN"] is not None
    # 기본값(RL 블록 미개입) — 휴리스틱/단건 배정은 그대로 SINGLE_CARRIER
    assert row["SIZE_SELECT_REASON_CD"] == "SINGLE_CARRIER"
    assert -1e-9 <= row["BUCKET_WIP_SHARE"] <= 1.0 + 1e-9


def test_forced_replay_uses_fixed_reason_and_null_bucket_fields(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id = next(iter(sim.eqps))
    row = {
        "lot_id": "FORCED_LOT", "PLAN_PROD_ATTR_VAL": "PPK_X", "oper_id": "OP_X",
        "wf_qty": 25, "carrier_id": "CAR_X", "logical_lot_id": "FORCED_LOT",
        "start_min": 0, "end_min": 60, "seq": 1,
    }
    sim._activate_fixed_queue_row(eqp_id, row)

    rec = sim.schedule[-1]
    assert rec["SELECT_REASON_CD"] == "FORCED_QUEUE_INIT"
    assert rec["SIZE_SELECT_REASON_CD"] == "FORCED_QUEUE_INIT"
    assert rec["BUCKET_WIP_SHARE"] is None


# ── env.scheduling_rl_env._size_select_reason ───────────────────────────────

def test_size_select_reason_no_wip():
    calc = {"block_size": 0, "target": 0, "cap": 0, "wip_carriers": 0,
            "plan_carriers": 0, "has_plan": False, "level": 0, "n_levels": 4, "frac": 0.25}
    assert _size_select_reason(calc)[0] == "NO_WIP"


def test_size_select_reason_wip_limited():
    calc = {"block_size": 3, "target": 10, "cap": 3, "wip_carriers": 3,
            "plan_carriers": 3, "has_plan": False, "level": 3, "n_levels": 4, "frac": 1.0}
    assert _size_select_reason(calc)[0] == "WIP_LIMITED"


def test_size_select_reason_plan_limited():
    calc = {"block_size": 2, "target": 10, "cap": 2, "wip_carriers": 8,
            "plan_carriers": 2, "has_plan": True, "level": 3, "n_levels": 4, "frac": 1.0}
    assert _size_select_reason(calc)[0] == "PLAN_LIMITED"


def test_size_select_reason_size_level():
    calc = {"block_size": 2, "target": 2, "cap": 10, "wip_carriers": 10,
            "plan_carriers": 10, "has_plan": False, "level": 0, "n_levels": 4, "frac": 0.25}
    assert _size_select_reason(calc)[0] == "SIZE_LEVEL"


# ── SchedulingRLEnv 블록 시작/연속 스텝 ──────────────────────────────────────

def test_rl_env_block_start_and_continue_fill_size_reason(env_data):
    env = SchedulingRLEnv(env_data, record_history=False, record_event_log=False)
    env.reset()
    rng = np.random.default_rng(0)

    start_reasons = []
    continue_reasons = []
    for _ in range(300):
        mask = env.action_masks()
        bucket_mask = mask[: env._n_bucket]
        buckets = np.flatnonzero(bucket_mask)
        bucket = int(rng.choice(buckets)) if buckets.size else 0
        schedule_before = len(env.sim.schedule)
        eqp_before_ids = {r["EQP_ID"] for r in env.sim.schedule}
        _, _, terminated, truncated, _ = env.step([bucket, env._L - 1])
        if len(env.sim.schedule) > schedule_before:
            row = env.sim.schedule[-1]
            cd = row["SIZE_SELECT_REASON_CD"]
            if cd == "BLOCK_CONTINUE":
                continue_reasons.append(cd)
            elif cd != "SINGLE_CARRIER":
                start_reasons.append(cd)
        if terminated or truncated:
            break

    assert start_reasons, "블록 시작 스텝에서 SIZE_LEVEL/WIP_LIMITED/PLAN_LIMITED류 사유가 남아야 함"
    assert all(
        cd in ("SIZE_LEVEL", "WIP_LIMITED", "PLAN_LIMITED", "NO_WIP")
        for cd in start_reasons
    )


# ── data/writer: RTS_RSLT_MAS/HIS 새 컬럼 ───────────────────────────────────

def _schedule_row(**overrides):
    row = {
        "EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
        "START_TM": 0, "END_TM": 60, "WF_QTY": 25, "ST": 60,
        "LOT_CD": "LC_A", "TEMP": "T600", "EQP_MODEL": "A",
        "LOT_STAT_CD": "WAIT", "SEQ": 1,
        "SELECT_REASON_CD": "SAME_SETUP", "SELECT_REASON_CTN": "same_setup:+0.3000",
        "SIZE_SELECT_REASON_CD": "SIZE_LEVEL", "SIZE_SELECT_REASON_CTN": "level 1/4",
        "BUCKET_WIP_SHARE": 0.5, "BUCKET_URGENCY": 0.25,
        "BUCKET_COVERAGE_RATIO": 0.75, "BUCKET_STARVE_NORM": 1.0,
        "BUCKET_NEEDS_CONV": False, "BUCKET_AVOIDABLE_FRAC": 0.0,
        "BUCKET_SETUP_CHANGED": True,
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


def test_rts_rslt_row_carries_select_reason_and_bucket_fields():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    row = payload["RTS_RSLT_MAS"][0]
    assert row["SELECT_REASON_CD"] == "SAME_SETUP"
    assert row["SIZE_SELECT_REASON_CD"] == "SIZE_LEVEL"
    assert row["BUCKET_WIP_SHARE"] == 0.5
    assert row["BUCKET_NEEDS_CONV"] is False


def test_rts_rslt_mas_and_his_insert_include_new_columns():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload)
    for name in ("rts_rslt_mas.sql", "rts_rslt_his.sql"):
        insert_line = next(
            line for line in scripts[name].splitlines() if line.startswith("INSERT INTO")
        )
        for col in (
            "SELECT_REASON_CD", "SELECT_REASON_CTN",
            "SIZE_SELECT_REASON_CD", "SIZE_SELECT_REASON_CTN",
            "BUCKET_WIP_SHARE", "BUCKET_URGENCY", "BUCKET_COVERAGE_RATIO",
            "BUCKET_STARVE_NORM", "BUCKET_NEEDS_CONV", "BUCKET_AVOIDABLE_FRAC",
            "BUCKET_SETUP_CHANGED",
        ):
            assert col in insert_line, f"{col} missing from {name}"


def test_rts_rslt_insert_encodes_bool_bucket_fields_as_y_n():
    result = {"algorithm": "earliest_st", "schedule": [_schedule_row()], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload)
    insert_line = next(
        line for line in scripts["rts_rslt_mas.sql"].splitlines() if line.startswith("INSERT INTO")
    )
    assert "'N'" in insert_line  # BUCKET_NEEDS_CONV=False
    assert "'Y'" in insert_line  # BUCKET_SETUP_CHANGED=True


def test_rts_rslt_insert_handles_null_bucket_fields():
    row = _schedule_row(
        SELECT_REASON_CD=None, BUCKET_WIP_SHARE=None, BUCKET_NEEDS_CONV=None,
    )
    result = {"algorithm": "earliest_st", "schedule": [row], "conversion_plans": []}
    payload = build_rts_output(result, _writer_env_data(), fac_id="FAC001", rule_timekey="20260719070000")
    scripts = build_writer_sql_scripts(payload)
    insert_line = next(
        line for line in scripts["rts_rslt_mas.sql"].splitlines() if line.startswith("INSERT INTO")
    )
    assert "NULL" in insert_line
