"""학습 진행 상태 UI 스냅샷 테스트."""
import json

from agent.train_progress import TrainProgressState, TRAIN_BUDGET_EPISODES, TRAIN_BUDGET_TIMESTEPS


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
