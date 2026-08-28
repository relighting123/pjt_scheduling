"""env.scheduling_rl_env.randomize_axis_order — 학습 시 축(공정/제품/설비/모델)
순서 랜덤화(도메인 랜덤화)가 올바르게 동작하는지 검증.

배경: 정책 네트워크가 순열 불변이 아닌 평범한 MLP라, 축 순서가 회차마다
데이터에 따라 재산출되면 정책이 슬롯 위치에 종속된 지름길을 학습해버린다
(axis_map.json으로 순서를 고정하는 우회 대신, 학습 중 순서를 오히려 매번
무작위로 흔들어 정책이 위치에 의존할 수 없게 만드는 접근으로 전환했다).
"""
import numpy as np
import pytest

from data.loader.preprocess import preprocess
from env.scheduling_rl_env import (
    RandomizedSchedulingRLEnv,
    SchedulingRLEnv,
    _AXIS_IDX_KEYS,
    build_env,
    randomize_axis_order,
)

RULE_TIMEKEY = "20260712070000"


def _raw(pairs, carriers=3, sts=None):
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate(pairs):
        st = (sts or {}).get(ppk, 30)
        for c in range(carriers):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": f"OPER00{(i % 3) + 1}", "ST": st,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": f"OPER00{(i % 3) + 1}",
                     "D0_PLAN_QTY": carriers, "D1_PLAN_QTY": carriers, "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": f"OPER00{(i % 3) + 1}"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": f"OPER00{(i % 3) + 1}",
                         "EQP_MODEL_CD": "A", "ST": st})
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed(raw, sim_end=240):
    ed = preprocess(raw, period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = sim_end
    ed["conversion_minutes"] = 30
    return ed


@pytest.fixture
def multi_ed():
    """축이 여러 개(3공정×3제품×3설비)라 순서 섞임이 관찰 가능한 데이터셋."""
    return _load_ed(_raw([
        ("EQP001", "PPK001"), ("EQP002", "PPK002"), ("EQP003", "PPK003"),
    ]))


# ── randomize_axis_order() 단위 테스트 ──────────────────────────────────────

def test_randomize_axis_order_preserves_membership(multi_ed):
    rng = np.random.default_rng(0)
    out = randomize_axis_order(multi_ed, rng)
    for axis_key in _AXIS_IDX_KEYS:
        assert sorted(out[axis_key]) == sorted(multi_ed[axis_key])


def test_randomize_axis_order_idx_matches_position(multi_ed):
    rng = np.random.default_rng(0)
    out = randomize_axis_order(multi_ed, rng)
    for axis_key, idx_key in _AXIS_IDX_KEYS.items():
        items = out[axis_key]
        idx_map = out[idx_key]
        assert len(idx_map) == len(items)
        for pos, name in enumerate(items):
            assert idx_map[name] == pos


def test_randomize_axis_order_does_not_mutate_input(multi_ed):
    before = {k: list(multi_ed[k]) for k in _AXIS_IDX_KEYS}
    randomize_axis_order(multi_ed, np.random.default_rng(0))
    for axis_key, items in before.items():
        assert multi_ed[axis_key] == items


def test_randomize_axis_order_actually_reorders_over_many_draws(multi_ed):
    """운(같은 순서로 뽑힘)으로 인한 거짓 통과를 피하기 위해 여러 시드로 확인."""
    base = tuple(multi_ed["prod_keys"])
    assert len(base) > 1
    saw_different_order = False
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out = randomize_axis_order(multi_ed, rng)
        if tuple(out["prod_keys"]) != base:
            saw_different_order = True
            break
    assert saw_different_order


# ── env 레벨 배선 ────────────────────────────────────────────────────────────

def test_default_env_does_not_randomize(multi_ed):
    """기본값(randomize_axis_order=False)이면 여러 번 reset해도 축 순서 불변."""
    env = build_env(SchedulingRLEnv, multi_ed)
    env.reset()
    order1 = list(env._env_data["prod_keys"])
    env.reset()
    order2 = list(env._env_data["prod_keys"])
    env.close()
    assert order1 == order2 == list(multi_ed["prod_keys"])


def test_randomized_env_varies_axis_order_across_resets(multi_ed):
    env = build_env(SchedulingRLEnv, multi_ed, randomize_axis_order=True)
    orders = set()
    for _ in range(30):
        env.reset()
        orders.add(tuple(env._env_data["prod_keys"]))
    env.close()
    assert len(orders) > 1


def test_randomized_env_stays_functionally_valid(multi_ed):
    """축 순서가 섞여도 masked step이 정상 동작해야 한다(관측/마스크 shape 불변)."""
    env = build_env(SchedulingRLEnv, multi_ed, randomize_axis_order=True)
    for _ in range(5):
        obs, _info = env.reset()
        assert obs.shape == env.observation_space.shape
        mask = env.action_masks()
        assert mask.shape[0] == env._n_bucket + env._L
        assert mask[: env._n_bucket].any()
    env.close()


def test_randomized_pool_env_randomizes_axis_order(multi_ed):
    other_ed = _load_ed(_raw([("EQP004", "PPK004"), ("EQP005", "PPK005")]))
    env = build_env(
        RandomizedSchedulingRLEnv, [multi_ed, other_ed],
        pool_seed=0, randomize_axis_order=True,
    )
    orders = set()
    for _ in range(30):
        env.reset()
        orders.add(tuple(env._env_data["prod_keys"]))
    env.close()
    assert len(orders) > 1


def test_eval_env_unrandomized_by_default_via_build_env(multi_ed):
    """추론/평가 경로가 쓰는 build_env() 기본 호출은 randomize_axis_order 인자를
    안 주면 항상 False다 — inference/kpi_eval 등에서 실수로 흔들리지 않게."""
    env = build_env(SchedulingRLEnv, multi_ed)
    assert env._randomize_axis_order is False
