"""학습 환경이 추론 환경과 같은 조건인지 검증 (train/serve skew 방지).

학습과 추론이 서로 다른 MDP를 돌면 학습 지표는 좋은데 벤치마크가 안 따라온다.
실측 사례: 같은 모델인데 학습 중 KPI는 전환 63회, 벤치마크에서는 83회였다 —
학습 env가 sim_end에서 끊겨(`truncate_on_time=True`) 컷오프 이후 구간을 아예
겪지 않았는데, 벤치마크는 그 구간의 전환까지 점수에 반영했기 때문이다.
"""
import numpy as np
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import (
    RandomizedSchedulingRLEnv,
    SchedulingRLEnv,
    build_env,
)
from agent.rl_agent import align_datasets_with_inference

RULE_TIMEKEY = "20260712070000"


def _raw(pairs, carriers=3):
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate(pairs):
        for c in range(carriers):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": carriers, "D1_PLAN_QTY": carriers,
                     "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                         "EQP_MODEL_CD": "A", "ST": 30})
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed(pairs, carriers=3, sim_end=180):
    ed = preprocess(_raw(pairs, carriers), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = sim_end
    ed["conversion_minutes"] = 30
    return ed


@pytest.fixture
def rl_cfg():
    cfg = CONFIG.rl
    original = cfg.train_truncate_on_time
    yield cfg
    cfg.train_truncate_on_time = original


# ── 데이터셋 플래그 정렬 ─────────────────────────────────────────────────────

def test_alignment_fills_inference_flags():
    ed = _load_ed([("EQP001", "PPK001")])
    ed.pop("termination_mode", None)
    ed.pop("enable_wip_inflow", None)

    aligned = align_datasets_with_inference([ed])[0]
    assert aligned["termination_mode"] == "current_wip_assigned"
    assert aligned["enable_wip_inflow"] is False


def test_alignment_respects_explicit_dataset_values():
    ed = _load_ed([("EQP001", "PPK001")])
    ed["termination_mode"] = "all_wip"
    ed["enable_wip_inflow"] = True

    aligned = align_datasets_with_inference([ed])[0]
    assert aligned["termination_mode"] == "all_wip"
    assert aligned["enable_wip_inflow"] is True


def test_alignment_does_not_mutate_input():
    ed = _load_ed([("EQP001", "PPK001")])
    ed.pop("termination_mode", None)
    align_datasets_with_inference([ed])
    assert "termination_mode" not in ed


# ── build_env ────────────────────────────────────────────────────────────────

def test_build_env_uses_configured_truncate_flag(rl_cfg):
    ed = _load_ed([("EQP001", "PPK001")])

    rl_cfg.train_truncate_on_time = False
    assert build_env(SchedulingRLEnv, ed)._truncate_on_time is False

    rl_cfg.train_truncate_on_time = True
    assert build_env(SchedulingRLEnv, ed)._truncate_on_time is True


def test_build_env_kwargs_override_defaults(rl_cfg):
    ed = _load_ed([("EQP001", "PPK001")])
    rl_cfg.train_truncate_on_time = False
    env = build_env(SchedulingRLEnv, ed, truncate_on_time=True, record_history=True)
    assert env._truncate_on_time is True
    assert env._record_history is True


# ── 도메인 랜덤화 env ────────────────────────────────────────────────────────

def test_randomized_env_visits_multiple_datasets():
    pool = [
        _load_ed([("EQP001", "PPK001")], carriers=2),
        _load_ed([("EQP001", "PPK002")], carriers=4),
        _load_ed([("EQP001", "PPK003")], carriers=6),
    ]
    env = RandomizedSchedulingRLEnv(pool, pool_seed=0, record_history=False,
                                    record_event_log=False)
    seen = set()
    for _ in range(40):
        env.reset()
        seen.add(env.current_dataset_index)
    assert len(seen) == len(pool), f"풀 일부만 방문: {seen}"


def test_randomized_env_matches_delegate_observation_space():
    pool = [_load_ed([("EQP001", "PPK001")]), _load_ed([("EQP001", "PPK002")])]
    env = RandomizedSchedulingRLEnv(pool, pool_seed=1, record_history=False,
                                    record_event_log=False)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.action_masks().shape[0] == env._n_bucket + env._L


def test_randomized_env_switches_scenario_state_on_reset():
    """reset마다 시나리오가 바뀌면 producible_carriers도 따라 바뀌어야 한다."""
    pool = [
        _load_ed([("EQP001", "PPK001")], carriers=2),
        _load_ed([("EQP001", "PPK002")], carriers=8),
    ]
    env = RandomizedSchedulingRLEnv(pool, pool_seed=3, record_history=False,
                                    record_event_log=False)
    seen = set()
    for _ in range(30):
        env.reset()
        seen.add(env.producible_carriers())
    assert seen == {2, 8}


def test_randomized_env_rejects_empty_pool():
    with pytest.raises(ValueError, match="비어 있"):
        RandomizedSchedulingRLEnv([])


def test_randomized_env_is_steppable():
    pool = [_load_ed([("EQP001", "PPK001")], carriers=2)]
    env = RandomizedSchedulingRLEnv(pool, pool_seed=0, record_history=False,
                                    record_event_log=False)
    env.reset()
    for _ in range(50):
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, 0], dtype=np.int64)
        _o, _r, terminated, truncated, _i = env.step(action)
        if terminated or truncated:
            break
    assert env.sim is not None
