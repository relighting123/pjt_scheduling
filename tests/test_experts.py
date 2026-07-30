"""agent/experts.py – BC 시연자 후보 · 시나리오별 최적 전문가 선택 검증.

시연자를 DedicationAgent로 고정하면 정책의 출발점 상한이 전담 휴리스틱이
된다. 그런데 전담이 늘 최선은 아니라서(처리시간이 이질적인 시나리오에서는
Earliest-ST 계열이 더 낫다), 시나리오마다 후보를 굴려보고 최적을 고른다.
"""
import numpy as np
import pytest

from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv, build_env
from agent.experts import (
    DEFAULT_EXPERT_CANDIDATES,
    EXPERT_FACTORIES,
    ExpertPolicy,
    make_expert,
    run_expert_episode,
    select_best_expert,
)

RULE_TIMEKEY = "20260712070000"


def _raw(pairs, carriers=3, sts=None, qty=None):
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate(pairs):
        st = (sts or {}).get(ppk, 30)
        n = (qty or {}).get(ppk, carriers)
        for c in range(n):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": st,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": n, "D1_PLAN_QTY": n, "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
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
def sym_ed():
    """EQP001<->PPK001, EQP002<->PPK002 전담 시 전환 0회가 최적."""
    return _load_ed(_raw([("EQP001", "PPK001"), ("EQP002", "PPK002")]))


# ── 전문가 개별 동작 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(EXPERT_FACTORIES))
def test_every_expert_produces_masked_actions(name, sym_ed):
    """어떤 전문가든 마스크 밖 행동을 내면 안 된다 (BC 라벨로 바로 쓰이므로)."""
    expert = make_expert(name, sym_ed)
    env = build_env(SchedulingRLEnv, sym_ed)
    env.reset()
    n_bucket = env._n_bucket

    for _ in range(200):
        mask = env.action_masks()
        eqp_id = env.sim.current_idle_eqp()
        flat = expert.resolve(env.sim, eqp_id, mask, n_bucket)
        level = expert.size_level(mask, n_bucket)
        assert 0 <= flat < n_bucket
        assert mask[flat], f"{name}: bucket {flat} outside mask"
        assert mask[n_bucket + level], f"{name}: level {level} outside mask"
        _o, _r, terminated, truncated, _i = env.step(
            np.array([flat, level], dtype=np.int64)
        )
        if terminated or truncated:
            break


def test_unknown_expert_name_raises(sym_ed):
    with pytest.raises(ValueError, match="알 수 없는 전문가"):
        make_expert("nope", sym_ed)


def test_size_level_policy_max_vs_min(sym_ed):
    """전담은 긴 블록(최대 레벨), Earliest-ST는 매 carrier 재결정(레벨 0)."""
    env = build_env(SchedulingRLEnv, sym_ed)
    env.reset()
    n_bucket = env._n_bucket
    mask = env.action_masks()
    allowed = np.flatnonzero(mask[n_bucket:])

    assert make_expert("dedication", sym_ed).size_level(mask, n_bucket) == int(allowed[-1])
    assert make_expert("earliest_st", sym_ed).size_level(mask, n_bucket) == int(allowed[0])


def test_expert_resolve_falls_back_on_broken_policy(sym_ed):
    """전문가가 예외를 던지거나 말도 안 되는 값을 줘도 feasible로 보정."""

    class _Broken(ExpertPolicy):
        def bucket(self, sim, eqp_id, mask, n_bucket):
            raise RuntimeError("boom")

    class _OutOfRange(ExpertPolicy):
        def bucket(self, sim, eqp_id, mask, n_bucket):
            return 10 ** 6

    env = build_env(SchedulingRLEnv, sym_ed)
    env.reset()
    n_bucket = env._n_bucket
    mask = env.action_masks()
    eqp_id = env.sim.current_idle_eqp()

    for expert in (_Broken(), _OutOfRange()):
        flat = expert.resolve(env.sim, eqp_id, mask, n_bucket)
        assert mask[flat]


# ── 시나리오별 최적 전문가 선택 ──────────────────────────────────────────────

def test_dedication_wins_on_symmetric_scenario(sym_ed):
    best, scores = select_best_expert(
        sym_ed, env_factory=lambda d: build_env(SchedulingRLEnv, d),
    )
    assert best == "dedication"
    assert scores["dedication"]["conversions"] == 0
    assert scores["dedication"]["score"] >= scores["earliest_st"]["score"]


def test_run_expert_episode_reports_kpi(sym_ed):
    env = build_env(SchedulingRLEnv, sym_ed)
    kpi = run_expert_episode(env, make_expert("dedication", sym_ed), 2000)
    assert kpi["produced"] == 6           # 2제품 × 3캐리어
    assert kpi["conversions"] == 0
    assert kpi["score"] == 6.0
    assert kpi["steps"] > 0


def test_single_candidate_selection_is_that_candidate(sym_ed):
    best, scores = select_best_expert(
        sym_ed, env_factory=lambda d: build_env(SchedulingRLEnv, d),
        candidates=["minprogress"],
    )
    assert best == "minprogress"
    assert set(scores) == {"minprogress"}


def test_selection_prefers_earlier_candidate_on_tie(sym_ed):
    """동점이면 후보 목록 앞쪽(기본은 dedication)을 고른다 — 결정적 동작."""
    best, _scores = select_best_expert(
        sym_ed, env_factory=lambda d: build_env(SchedulingRLEnv, d),
        candidates=["dedication", "minprogress"],
    )
    assert best == "dedication"


def test_default_candidates_are_registered():
    for name in DEFAULT_EXPERT_CANDIDATES:
        assert name in EXPERT_FACTORIES
