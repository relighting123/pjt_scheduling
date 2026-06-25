"""학습 진행 상태 UI 스냅샷 테스트."""
import json
from types import SimpleNamespace

import numpy as np

from agent.train_progress import (
    TrainProgressState,
    TRAIN_BUDGET_EPISODES,
    TRAIN_BUDGET_TIMESTEPS,
    read_rollout_ep_rew_mean,
)


def test_snapshot_completed_with_metrics_is_json_serializable():
    state = TrainProgressState()
    state.set_running(total_timesteps=10_000, budget_mode=TRAIN_BUDGET_TIMESTEPS)
    state.add_log("학습 시작")
    state.record_rollout_metrics(2048, {"rollout/ep_rew_mean": 1.5, "train/value_loss": 0.2})
    state.set_completed({"mean_reward": 1.2, "mean_completion": 0.8})
    snap = state.snapshot()
    raw = json.dumps(snap)
    parsed = json.loads(raw)
    assert parsed["status"] == "completed"
    assert parsed["metrics"]["mean_reward"] == 1.2
    assert len(parsed["logs"]) == 1
    assert parsed["series"]["timesteps"] == [2048]


def test_episode_budget_progress():
    state = TrainProgressState()
    state.set_running(total_episodes=5, budget_mode=TRAIN_BUDGET_EPISODES)
    state.update_episode_progress(3, 5)
    snap = state.snapshot()
    assert snap["train_budget_mode"] == TRAIN_BUDGET_EPISODES
    assert snap["episodes"] == 3
    assert snap["progress"] == 0.6


def test_reset_clears_previous_run():
    state = TrainProgressState()
    state.set_completed({"mean_reward": 1.0})
    state.reset()
    snap = state.snapshot()
    assert snap["status"] == "idle"
    assert snap["metrics"] is None
    assert snap["logs"] == []


def test_read_rollout_ep_rew_mean_from_episode_buffer():
    model = SimpleNamespace(
        ep_info_buffer=[{"r": 10.0, "l": 100}, {"r": 20.0, "l": 120}],
        rollout_buffer=SimpleNamespace(pos=0, rewards=np.zeros(1)),
    )
    mean, source = read_rollout_ep_rew_mean(model)
    assert mean == 15.0
    assert source == "episode"


def test_read_rollout_ep_rew_mean_fallback_to_step_rewards():
    model = SimpleNamespace(
        ep_info_buffer=[],
        rollout_buffer=SimpleNamespace(
            pos=4,
            rewards=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ),
    )
    mean, source = read_rollout_ep_rew_mean(model)
    assert mean == 2.5
    assert source == "step_mean"


def test_record_rollout_metrics_skips_missing_reward():
    state = TrainProgressState()
    state.record_rollout_metrics(1024, {})
    snap = state.snapshot()
    assert snap["series"]["timesteps"] == []
