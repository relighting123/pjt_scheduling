"""agent/bc.py – DAgger식 시연 수집 · BC 학습 · 전문가 앵커 검증.

기존 BC 구현의 실패 모드(표본 부족 / 크리틱 미학습 / 결정적 붕괴 /
PPO 중 파국적 망각)를 각각 막고 있는지 확인한다.
"""
from types import SimpleNamespace

import numpy as np
import torch

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from agent.bc import (
    ExpertAnchorCallback,
    behavior_clone,
    collect_expert_dataset,
)

RULE_TIMEKEY = "20260712070000"


def _sym_2x2_raw():
    """EQP001<->PPK001, EQP002<->PPK002 전담 시 전환 0회가 최적인 대칭 케이스."""
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate([("EQP001", "PPK001"), ("EQP002", "PPK002")]):
        for c in range(3):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 3, "D1_PLAN_QTY": 3, "PLAN_PRIORITY": 1,
        })
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
            "EQP_MODEL_CD": "A", "ST": 30,
        })
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed():
    ed = preprocess(_sym_2x2_raw(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 180
    ed["conversion_minutes"] = 30
    return ed


def _make_model(ed):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agent.rl_agent import _mask_fn

    def make_env():
        return Monitor(ActionMasker(
            SchedulingRLEnv(ed, record_history=False, record_event_log=False), _mask_fn,
        ))

    return MaskablePPO(
        "MlpPolicy", DummyVecEnv([make_env]),
        n_steps=64, batch_size=16, device="cpu", seed=0, verbose=0,
    )


# ── 시연 수집 ────────────────────────────────────────────────────────────────

def test_collected_actions_are_always_within_mask():
    """교란을 넣어도 라벨(전문가 행동)은 항상 마스크 안이어야 한다.

    마스크 밖 라벨을 주면 로그확률이 −inf에 가까워져 BC가 폭주한다.
    """
    ed = _load_ed()
    ds = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=4, noise_eps=0.5, seed=0,
    )
    assert len(ds) > 0
    n_bucket = SchedulingRLEnv(ed, record_history=False, record_event_log=False)._n_bucket
    for i in range(len(ds)):
        bucket, level = ds.actions[i]
        assert 0 <= bucket < n_bucket
        assert ds.masks[i, bucket], f"step {i}: bucket {bucket} outside mask"
        assert ds.masks[i, n_bucket + level], f"step {i}: level {level} outside mask"


def test_noise_expands_dataset_beyond_single_deterministic_episode():
    """DedicationAgent는 결정적이라 교란 없이는 몇 번을 굴려도 같은 궤적이다.

    ε 교란이 상태분포를 실제로 넓혀 표본 수가 늘어나는지 확인한다(BC 표본
    부족이 옛 구현의 최대 약점이었다).
    """
    ed = _load_ed()
    plain = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=6, noise_eps=0.0, seed=0,
    )
    noisy = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=6, noise_eps=0.4, seed=0,
    )
    assert plain.n_episodes == 1        # 교란 0 → 반복 수집을 건너뛴다
    assert noisy.n_episodes == 6
    assert len(noisy) > len(plain)


def test_returns_are_discounted_reward_to_go():
    ed = _load_ed()
    ds = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=1, noise_eps=0.0, gamma=0.9, seed=0,
    )
    assert ds.returns.shape[0] == len(ds)
    assert np.isfinite(ds.returns).all()


# ── BC 학습 ─────────────────────────────────────────────────────────────────

def test_behavior_clone_reduces_nll_and_trains_value_head():
    """정책 NLL이 줄고 **크리틱도 같이 학습**되는지.

    옛 구현은 value head를 무작위로 남겨둬서 PPO 첫 업데이트의 advantage가
    통째로 잡음이 됐고, 그게 "BC 직후엔 좋은데 RL을 돌리면 나빠진다"의
    핵심 원인이었다.
    """
    ed = _load_ed()
    model = _make_model(ed)
    ds = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=8, noise_eps=0.3, seed=0,
    )

    obs_t = torch.as_tensor(ds.obs, dtype=torch.float32)
    act_t = torch.as_tensor(ds.actions, dtype=torch.long)
    mask_t = torch.as_tensor(ds.masks, dtype=torch.bool)
    ret_t = torch.as_tensor(ds.returns, dtype=torch.float32)
    ret_norm = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    with torch.no_grad():
        values0, logp0, _ = model.policy.evaluate_actions(obs_t, act_t, mask_t)
    nll_before = float(-logp0.mean())
    v_err_before = float(torch.nn.functional.mse_loss(values0.flatten(), ret_norm))

    behavior_clone(
        model, ds, epochs=60, lr=1e-3, batch_size=64,
        value_coef=0.5, val_fraction=0.0, seed=0,
    )

    with torch.no_grad():
        values1, logp1, _ = model.policy.evaluate_actions(obs_t, act_t, mask_t)
    nll_after = float(-logp1.mean())
    v_err_after = float(torch.nn.functional.mse_loss(values1.flatten(), ret_norm))

    assert nll_after < nll_before
    assert v_err_after < v_err_before, "value head가 학습되지 않았다"


def test_behavior_clone_keeps_entropy_above_zero():
    """엔트로피 정규화가 결정적 붕괴(entropy≈0)를 막는지.

    붕괴하면 PPO가 탐색도 학습도 못 해 학습 곡선이 평평해진다.
    """
    ed = _load_ed()
    model = _make_model(ed)
    ds = collect_expert_dataset(
        [ed], SchedulingRLEnv, episodes_per_dataset=8, noise_eps=0.3, seed=0,
    )
    summary = behavior_clone(
        model, ds, epochs=150, lr=1e-3, batch_size=64,
        entropy_coef=0.02, value_coef=0.5, val_fraction=0.0, seed=0,
    )
    assert summary["entropy"] > 0.01


def test_behavior_clone_skips_on_empty_dataset():
    ed = _load_ed()
    model = _make_model(ed)
    empty = collect_expert_dataset([], SchedulingRLEnv)
    summary = behavior_clone(model, empty, epochs=5)
    assert summary["skipped"] is True


# ── 전문가 앵커 ──────────────────────────────────────────────────────────────

def _anchor_with_progress(cb, elapsed):
    cb.model = SimpleNamespace(_current_progress_remaining=1.0 - elapsed)
    return cb._current_coef()


def test_expert_anchor_coefficient_decays_then_holds():
    ed = _load_ed()
    ds = collect_expert_dataset([ed], SchedulingRLEnv, episodes_per_dataset=2,
                                noise_eps=0.2, seed=0)
    cb = ExpertAnchorCallback(ds, coef_start=1.0, coef_final=0.0, decay_fraction=0.5)

    assert abs(_anchor_with_progress(cb, 0.0) - 1.0) < 1e-9
    assert abs(_anchor_with_progress(cb, 0.25) - 0.5) < 1e-9
    assert abs(_anchor_with_progress(cb, 0.5) - 0.0) < 1e-9
    # 감쇠 구간을 지나면 final에 고정
    assert abs(_anchor_with_progress(cb, 0.9) - 0.0) < 1e-9


def test_expert_anchor_inactive_without_data_or_coef():
    ed = _load_ed()
    ds = collect_expert_dataset([ed], SchedulingRLEnv, episodes_per_dataset=2,
                                noise_eps=0.2, seed=0)
    assert ExpertAnchorCallback(ds, coef_start=1.0, steps_per_rollout=2).active
    assert not ExpertAnchorCallback(ds, coef_start=0.0, steps_per_rollout=2).active
    assert not ExpertAnchorCallback(ds, coef_start=1.0, steps_per_rollout=0).active
    empty = collect_expert_dataset([], SchedulingRLEnv)
    assert not ExpertAnchorCallback(empty, coef_start=1.0, steps_per_rollout=2).active


def test_expert_anchor_pulls_policy_toward_expert():
    """앵커 스텝이 실제로 전문가 NLL을 낮추는 방향으로 정책을 옮기는지."""
    ed = _load_ed()
    model = _make_model(ed)
    ds = collect_expert_dataset([ed], SchedulingRLEnv, episodes_per_dataset=6,
                                noise_eps=0.3, seed=0)
    cb = ExpertAnchorCallback(
        ds, coef_start=1.0, coef_final=0.0, decay_fraction=1.0,
        steps_per_rollout=30, batch_size=64, lr=1e-3, seed=0,
    )
    # learn() 밖에서 콜백을 직접 돌리므로 SB3가 해주는 logger 세팅을 대신한다.
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, []))
    cb.model = model
    cb._on_training_start()

    obs_t = torch.as_tensor(ds.obs, dtype=torch.float32)
    act_t = torch.as_tensor(ds.actions, dtype=torch.long)
    mask_t = torch.as_tensor(ds.masks, dtype=torch.bool)
    with torch.no_grad():
        _v, logp0, _e = model.policy.evaluate_actions(obs_t, act_t, mask_t)

    model._current_progress_remaining = 1.0
    cb._on_rollout_start()

    with torch.no_grad():
        _v, logp1, _e = model.policy.evaluate_actions(obs_t, act_t, mask_t)

    assert float(-logp1.mean()) < float(-logp0.mean())
