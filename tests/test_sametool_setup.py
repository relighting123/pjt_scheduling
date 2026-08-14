"""
tests/test_sametool_setup.py

sametool_setup(BATCHID는 동일·PPK/OPER만 전환)을 실제 TOOL 전환과 구분해
카운트/리워드하는 기능을 검증한다. 설계서:
docs/superpowers/specs/2026-08-14-sametool-setup-dedication-design.md
"""
from pathlib import Path

import pytest

from agent.kpi_eval import KPI, KPIResult
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


def test_partial_prior_state_prev_prod_none_is_not_sametool_setup(sim):
    """prev_lot_cd만 채워지고 prev_prod/prev_oper가 아직 None(=실제 첫 배정 전,
    초기 스냅샷이 일부만 채운 상태)이면 비교할 이전 PPK/OPER 정체성이 없으므로
    sametool_setup으로 오판하면 안 된다(위 switch 카운터 가드와 동일한 취지)."""
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = None
    eqp.prev_oper = None
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTA", "T1")

    assert reward == 0.0
    assert sim.stats["sametool_setup_count"] == before


def test_reward_breakdown_files_penalty_under_sametool_setup_key(sim, monkeypatch):
    """same_batch 페널티 branch(음수)는 terms['same_setup']이 아니라
    terms['sametool_setup']에 기록돼야 한다(부호가 반대인 두 항을 같은 키에
    몰아넣으면 downstream 트레이스/프론트 라벨이 잘못 해석함)."""
    monkeypatch.setattr(sim, "_same_setup_reward", lambda *a, **k: -0.5)
    eqp_id = sim.current_idle_eqp()
    assert eqp_id is not None
    lots = sim.available_lots(eqp_id)
    assert lots
    reward = sim.assign_lot(eqp_id, lots[0]["lot_id"])
    assert reward != -1.0, "배정이 실패하면 reward_breakdown 자체가 채워지지 않아 테스트가 무의미해짐"

    bd = sim._last_reward_breakdown
    assert bd.get("sametool_setup") == pytest.approx(-0.5)
    assert "same_setup" not in bd


def test_reward_breakdown_files_bonus_under_same_setup_key(sim, monkeypatch):
    """same_oper&&same_prod 보너스 branch(양수)는 기존대로 terms['same_setup']에
    기록돼야 한다(회귀 확인 — sametool_setup 분리로 기존 경로가 깨지지 않았는지)."""
    monkeypatch.setattr(sim, "_same_setup_reward", lambda *a, **k: 0.5)
    eqp_id = sim.current_idle_eqp()
    assert eqp_id is not None
    lots = sim.available_lots(eqp_id)
    assert lots
    sim.assign_lot(eqp_id, lots[0]["lot_id"])

    bd = sim._last_reward_breakdown
    assert bd.get("same_setup") == pytest.approx(0.5)
    assert "sametool_setup" not in bd


def test_history_snapshot_exposes_sametool_setup(sim):
    """history를 켠 시뮬에서 스냅샷마다 sametool_setup 키가 stats와 일치해야 한다."""
    sim._record_history = True
    eqp_id = sim.current_idle_eqp()
    assert eqp_id is not None
    lots = sim.available_lots(eqp_id)
    assert lots
    sim.assign_lot(eqp_id, lots[0]["lot_id"])
    sim.save_history_step()

    assert sim.history, "assign_lot 이후 history가 최소 1건 기록돼야 함"
    last = sim.history[-1]
    assert "sametool_setup" in last
    assert last["sametool_setup"] == sim.stats["sametool_setup_count"]


def test_kpi_as_dict_includes_sametool_setup():
    k = KPI(produced=5, producible=10, conversions=2, sametool_setup=3, reward=1.0, steps=5)
    assert k.as_dict()["sametool_setup"] == 3


def test_kpi_result_aggregates_sametool_setup():
    k1 = KPI(produced=5, producible=10, conversions=2, sametool_setup=3, reward=1.0, steps=5)
    k2 = KPI(produced=4, producible=10, conversions=1, sametool_setup=1, reward=0.5, steps=4)
    result = KPIResult(per_dataset=[k1, k2])
    assert result.sametool_setup == 4
    assert result.as_dict()["sametool_setup"] == 4
