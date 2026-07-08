"""discrete_arrange LOT_STAT_CD (PROC/LOAD/SELE/RESV/WAIT) 강제 배정 단위 테스트."""
import pytest

from data.loader.preprocess import preprocess
from env.scheduling_env import SchedulingEnv
from validation.output_checks import check_forced_placement

ST, WF = 5, 25


def _disc(eqp, lot, lot_stat_cd=None):
    row = {
        "EQP_ID": eqp,
        "LOT_ID": lot,
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "OPER_ID": "OPER001",
        "ST": ST,
        "EQP_MODEL_CD": "A",
        "WF_QTY": WF,
        "CARRIER_ID": f"C{lot}",
    }
    if lot_stat_cd is not None:
        row["LOT_STAT_CD"] = lot_stat_cd
    return row


def _raw(discrete):
    return {
        "discrete_arrange": discrete,
        "plan": [{
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1,
        }],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
    }


def _build_scenario():
    discrete = [
        _disc("EQP001", "LOT_P", "PROC"),
        _disc("EQP001", "LOT_L", "LOAD"),
        _disc("EQP001", "LOT_W1", "WAIT"),
        _disc("EQP002", "LOT_W2"),  # LOT_STAT_CD 미지정 -> WAIT 기본값
    ]
    return preprocess(_raw(discrete))


def test_preprocess_builds_forced_queue_in_stat_priority_order():
    """입력 순서와 무관하게 PROC → LOAD → RESV → SELE 순으로 큐 정렬."""
    discrete = [
        _disc("EQP001", "LOT_S", "SELE"),
        _disc("EQP001", "LOT_R", "RESV"),
        _disc("EQP001", "LOT_L", "LOAD"),
        _disc("EQP001", "LOT_P", "PROC"),
    ]
    env_data = preprocess(_raw(discrete))
    assert env_data["eqp_forced_queue"] == {
        "EQP001": ["CLOT_P", "CLOT_L", "CLOT_R", "CLOT_S"],
    }


def test_preprocess_builds_forced_queue_proc_before_load():
    env_data = _build_scenario()
    assert env_data["eqp_forced_queue"] == {"EQP001": ["CLOT_P", "CLOT_L"]}
    lots = {l["lot_id"]: l for l in env_data["lots"]}
    assert lots["CLOT_P"]["lot_stat_cd"] == "PROC"
    assert lots["CLOT_P"]["logical_lot_id"] == "LOT_P"
    assert lots["CLOT_L"]["lot_stat_cd"] == "LOAD"
    assert lots["CLOT_W1"]["lot_stat_cd"] == "WAIT"
    assert lots["CLOT_W2"]["lot_stat_cd"] == "WAIT"


def test_missing_lot_stat_cd_defaults_to_wait_no_forced_queue():
    discrete = [_disc("EQP001", "LOT_A"), _disc("EQP002", "LOT_B")]
    env_data = preprocess(_raw(discrete))
    assert env_data["eqp_forced_queue"] == {}
    lots = {l["lot_id"]: l for l in env_data["lots"]}
    assert lots["CLOT_A"]["lot_stat_cd"] == "WAIT"
    assert lots["CLOT_B"]["lot_stat_cd"] == "WAIT"


def test_invalid_lot_stat_cd_raises():
    discrete = [_disc("EQP001", "LOT_X", "BOGUS")]
    with pytest.raises(ValueError):
        preprocess(_raw(discrete))


def _run_naive_rollout(env, max_steps=2000):
    """항상 action=0을 고르는 에이전트 — feasible mask/보정에만 의존해 강제 배정을 검증."""
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    return env.sim.schedule


def test_forced_lots_pin_to_equipment_in_written_order():
    env_data = _build_scenario()
    env = SchedulingEnv(env_data, record_history=False, record_event_log=False)
    schedule = _run_naive_rollout(env)

    eqp1_rows = sorted(
        (r for r in schedule if r["EQP_ID"] == "EQP001"), key=lambda r: r["START_TM"],
    )
    eqp1_forced_order = [r["LOT_ID"] for r in eqp1_rows if r["LOT_ID"] in ("CLOT_P", "CLOT_L")]
    assert eqp1_forced_order == ["CLOT_P", "CLOT_L"]

    # 강제 carrier는 지정된 EQP001 외 어디에도 배정되지 않는다.
    other_eqp_lots = {r["LOT_ID"] for r in schedule if r["EQP_ID"] != "EQP001"}
    assert "CLOT_P" not in other_eqp_lots
    assert "CLOT_L" not in other_eqp_lots

    scheduled_lot_ids = {r["LOT_ID"] for r in schedule}
    assert {"CLOT_P", "CLOT_L", "CLOT_W1", "CLOT_W2"} <= scheduled_lot_ids

    assert check_forced_placement(schedule, env_data) == []


def test_check_forced_placement_flags_wrong_equipment_and_order():
    env_data = _build_scenario()
    bad_schedule = [
        {"EQP_ID": "EQP002", "LOT_ID": "CLOT_P", "START_TM": 0, "END_TM": 100},
        {"EQP_ID": "EQP001", "LOT_ID": "CLOT_L", "START_TM": 0, "END_TM": 100},
    ]
    violations = check_forced_placement(bad_schedule, env_data)
    assert violations
    reasons = {v["reason"] for v in violations}
    assert any("EQP" in r for r in reasons)
