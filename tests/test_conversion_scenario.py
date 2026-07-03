"""LOT_CD/TEMP conversion 시나리오 – 초기 EQP 상태, bucket, conversion 수행."""
import numpy as np
import pytest

from config import CONFIG
from data.conversion_scenarios import bootstrap_conv_test_suite, build_conv_2ppk_1oper
from data.loader import validate_data, preprocess
from env.scheduling_env import SchedulingEnv, compute_obs_dim
from inference.runner import run_inference
from simulation.simulator import SchedulingSimulator


def _env_data():
    discrete, plan, flow, lot_master, abstract, eqp_init, tool_cap = build_conv_2ppk_1oper()
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": lot_master,
        "tool_capacity": tool_cap,
        "eqp_initial_state": eqp_init,
    }
    assert not validate_data(raw), validate_data(raw)
    return preprocess(raw)


def test_eqp_initial_state_applied():
    env_data = _env_data()
    assert env_data["eqp_initial_state"]
    sim = SchedulingSimulator(env_data, record_history=False)
    eqp = sim.eqps["EQP001"]
    assert eqp.prev_lot_cd == "LC001"
    assert eqp.prev_temp == "T650"
    assert eqp.prev_prod == "PPK001"
    eqp2 = sim.eqps["EQP002"]
    assert eqp2.prev_lot_cd is None


def test_bucket_needs_conversion_for_ppk002_on_eqp001():
    env_data = _env_data()
    sim = SchedulingSimulator(env_data, record_history=False)
    assert sim.current_idle_eqp() == "EQP001"

    O, P, K = CONFIG.env.max_oper_count, CONFIG.env.max_prod_count, CONFIG.env.max_model_count
    F = SchedulingSimulator.BUCKET_FEATURES
    bucket = sim.get_bucket_features()

    oi = env_data["oper_idx"]["OPER001"]
    pi2 = env_data["prod_idx"]["PPK002"]
    mi = env_data["model_idx"]["A"]

    assert bucket[oi, pi2, mi, 10] > 0, "wip lot_cd channel"
    assert bucket[oi, pi2, mi, 11] > 0, "wip temp channel"
    assert bucket[oi, pi2, mi, 12] == 1.0, "PPK002 on EQP001 should need conversion"

    pi1 = env_data["prod_idx"]["PPK001"]
    assert bucket[oi, pi1, mi, 12] == 0.0, "PPK001 same LOT_CD should not need conversion"


def test_obs_dim_with_extended_bucket():
    env_data = _env_data()
    env = SchedulingEnv(env_data)
    obs, _ = env.reset()
    assert obs.shape == (compute_obs_dim(),)
    assert SchedulingSimulator.BUCKET_FEATURES == 14


def test_conversion_performed_when_assigning_ppk002():
    env_data = _env_data()
    result = run_inference(env_data, algorithm="earliest_st", record_history=False)
    conv_rows = [r for r in result["schedule"] if r.get("CONVERSION")]
    assert result["stats"]["conversions"] >= 1, "PPK002 배정 시 conversion 기대"
    assert conv_rows, "schedule에 CONVERSION=True 행 기대"
    ppk002_on_eqp1 = [
        r for r in result["schedule"]
        if r["EQP_ID"] == "EQP001" and r["PLAN_PROD_ATTR_VAL"] == "PPK002"
    ]
    assert ppk002_on_eqp1, "EQP001에 PPK002 배정이 있어야 함"
    assert any(r.get("CONVERSION") for r in ppk002_on_eqp1)


def test_bootstrap_conv_dataset():
    info = bootstrap_conv_test_suite()
    from pathlib import Path
    inp = Path(info["input_dir"])
    assert (inp / "eqp_initial_state.json").is_file()
    assert (inp / "discrete_arrange.json").is_file()


if __name__ == "__main__":
    test_eqp_initial_state_applied()
    test_bucket_needs_conversion_for_ppk002_on_eqp001()
    test_conversion_performed_when_assigning_ppk002()
    print("conversion tests passed")
