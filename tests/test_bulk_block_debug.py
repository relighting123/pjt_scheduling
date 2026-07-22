"""
tests/test_bulk_block_debug.py

스텝 디버거의 '버킷/블록 크기' 표시용 데이터 검증:
  - simulator.bulk_block_size_breakdown()은 bulk_block_size()와 동일한 최종값과
    산출 근거(wip/plan/cap/takt예산/level비율/target)를 함께 돌려준다.
  - SchedulingRLEnv(record_decision_log=True)는 블록 시작 스텝에
    block_size_calc(산출 근거)를, 시작/연속 스텝에 block_progress(done/total/
    remaining)를 기록한다.
"""
import numpy as np
import pytest

from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


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


def test_breakdown_matches_bulk_block_size(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)

    for level in range(4):
        calc = sim.bulk_block_size_breakdown(eqp_id, ppk, oper_id, level, 4)
        assert calc["block_size"] == sim.bulk_block_size(eqp_id, ppk, oper_id, level, 4)


def test_breakdown_fields_are_consistent(env_data):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)

    calc = sim.bulk_block_size_breakdown(eqp_id, ppk, oper_id, 3, 4)

    assert calc["ppk"] == ppk and calc["oper_id"] == oper_id
    assert calc["cap"] == min(calc["wip_carriers"], calc["plan_carriers"])
    assert calc["frac"] == pytest.approx((calc["level"] + 1) / calc["n_levels"])
    assert calc["target"] == max(int(round(calc["frac"] * calc["takt_budget"])), 1)
    if calc["wip_carriers"] == 0 or calc["cap"] <= 0:
        assert calc["block_size"] == 0
    else:
        assert calc["block_size"] == max(min(calc["target"], calc["cap"]), 1)
    if not calc["has_plan"]:
        assert calc["plan_carriers"] == calc["wip_carriers"]


def test_level_monotonic_and_capped(env_data):
    """레벨이 커질수록 블록 크기는 줄지 않고, cap을 넘지 않는다."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id, ppk, oper_id = _first_idle_bucket(sim)

    sizes = [sim.bulk_block_size(eqp_id, ppk, oper_id, lv, 4) for lv in range(4)]
    cap = sim.bulk_block_size_breakdown(eqp_id, ppk, oper_id, 0, 4)["cap"]
    assert sizes == sorted(sizes)
    assert all(s <= max(cap, 1) for s in sizes)


def _run_env_collect_log(env_data, max_steps: int = 400):
    env = SchedulingRLEnv(
        env_data,
        record_history=False,
        record_event_log=False,
        record_decision_log=True,
    )
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(max_steps):
        mask = env.action_masks()
        bucket_mask = mask[: env._n_bucket]
        buckets = np.flatnonzero(bucket_mask)
        bucket = int(rng.choice(buckets)) if buckets.size else 0
        obs, reward, terminated, truncated, info = env.step([bucket, env._L - 1])
        if terminated or truncated:
            break
    return env.get_decision_log()


def test_decision_log_records_block_size_calc_and_progress(env_data):
    log = _run_env_collect_log(env_data)

    starts = [e for e in log if e.get("block_start")]
    assert starts, "블록 시작 스텝이 최소 1개는 기록되어야 함"

    for e in starts:
        calc = e.get("block_size_calc")
        assert calc is not None, "블록 시작 스텝에는 block_size_calc가 있어야 함"
        assert calc["block_size"] == e["block_size"]
        assert calc["level"] == e["size_level"]
        for key in ("wip_carriers", "plan_carriers", "cap", "takt_budget", "frac", "target"):
            assert key in calc
        prog = e.get("block_progress")
        assert prog is not None
        assert prog["done"] == 1
        assert prog["total"] == max(e["block_size"], 1)
        assert prog["remaining"] == max(e["block_size"] - 1, 0)


def test_decision_log_block_continuation_progress(env_data):
    log = _run_env_collect_log(env_data)

    conts = [e for e in log if e.get("block_progress") and not e.get("block_start")]
    assert conts, "블록 연속 스텝에 block_progress가 기록되어야 함"

    for e in conts:
        p = e["block_progress"]
        assert p["total"] is not None
        # done + remaining = total, 연속 스텝은 항상 2번째 이후 carrier
        assert p["done"] + p["remaining"] == p["total"]
        assert 2 <= p["done"] <= p["total"]

        # 연속 스텝의 total은 같은 EQP의 직전 블록 시작 block_size와 일치
        prior_start = next(
            (
                s for s in reversed(log)
                if s["step"] < e["step"] and s.get("eqp_id") == e["eqp_id"] and s.get("block_start")
            ),
            None,
        )
        assert prior_start is not None
        assert prior_start["block_size"] == p["total"]


def test_non_start_steps_have_no_calc(env_data):
    log = _run_env_collect_log(env_data)
    for e in log:
        if not e.get("block_start"):
            assert "block_size_calc" not in e
