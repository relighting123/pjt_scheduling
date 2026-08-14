"""
tests/test_sametool_setup.py

sametool_setup(BATCHID는 동일·PPK/OPER만 전환)을 실제 TOOL 전환과 구분해
카운트/리워드하는 기능을 검증한다. 설계서:
docs/superpowers/specs/2026-08-14-sametool-setup-dedication-design.md
"""
from pathlib import Path

import pytest

from config import CONFIG, apply_reward_params, reward_params_dict
from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


def test_reward_params_round_trip_includes_w_sametool_setup():
    original = CONFIG.reward.w_sametool_setup
    try:
        d = reward_params_dict()
        assert "w_sametool_setup" in d
        assert d["w_sametool_setup"] == pytest.approx(original)

        apply_reward_params({"w_sametool_setup": 0.75})
        assert CONFIG.reward.w_sametool_setup == pytest.approx(0.75)
    finally:
        CONFIG.reward.w_sametool_setup = original


@pytest.fixture()
def sim(tmp_path: Path) -> SchedulingSimulator:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_data(scenario="default", output_dir=input_dir)
    raw = load_data(input_dir)
    env_data = preprocess(raw, period_key=RULE_TIMEKEY)
    return SchedulingSimulator(env_data, record_history=False, record_event_log=False)


def _real_ppk_oper(sim: SchedulingSimulator):
    """샘플 데이터 안에서 실제로 feasible한 (ppk, oper) 하나를 가져온다."""
    flat, _ = next(iter(sim.get_feasible_assignments()))
    return sim.ppk_oper_from_flat(flat)


def test_same_ppk_oper_still_gives_bonus_unchanged(sim, monkeypatch):
    """배치 비교 로직 추가 후에도 기존 same_oper&&same_prod 보너스 경로는 그대로."""
    monkeypatch.setattr(CONFIG.reward, "w_same_setup", 1.0)
    ppk, oper_id = _real_ppk_oper(sim)
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = ppk
    eqp.prev_oper = oper_id
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, ppk, oper_id, 1, "LOTA", "T1")

    assert reward == pytest.approx(1.0)
    assert sim.stats["sametool_setup_count"] == before  # 카운트 변화 없음


def test_same_batch_different_ppk_oper_is_sametool_setup(sim, monkeypatch):
    monkeypatch.setattr(CONFIG.reward, "w_sametool_setup", 0.5)
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = "PPKX"
    eqp.prev_oper = "OP1"
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTA", "T1")

    assert reward == pytest.approx(-0.5)
    assert sim.stats["sametool_setup_count"] == before + 1


def test_different_batch_is_not_counted_as_sametool_setup(sim):
    """BATCHID 자체가 다르면(=TOOL 전환) 이 함수는 관여하지 않는다(0.0, 카운트 변화 없음)."""
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = "PPKX"
    eqp.prev_oper = "OP1"
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTB", "T9")

    assert reward == 0.0
    assert sim.stats["sametool_setup_count"] == before


def test_first_assignment_no_prior_batch_is_not_sametool_setup(sim):
    """prev_lot_cd가 None(첫 배정)이면 비교 대상이 없어 0.0."""
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = None
    eqp.prev_oper = None
    eqp.prev_lot_cd = None
    eqp.prev_temp = None
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTA", "T1")

    assert reward == 0.0
    assert sim.stats["sametool_setup_count"] == before
