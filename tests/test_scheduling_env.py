"""LOT_CD/TEMP 액션·상환경 단위 테스트"""
import numpy as np

from config import CONFIG
from data.loader import load_data
from data.preprocessor import preprocess
from env.scheduling_env import SchedulingEnv, compute_obs_dim
from simulation.simulator import SchedulingSimulator, ToolTracker


def test_obs_dim_matches_env():
    raw = load_data()
    env_data = preprocess(raw)
    env = SchedulingEnv(env_data)
    obs, _ = env.reset()
    assert obs.shape == (compute_obs_dim(),)
    assert env.observation_space.contains(obs)


def test_action_masks_shape():
    raw = load_data()
    env_data = preprocess(raw)
    env = SchedulingEnv(env_data)
    env.reset()
    O, P, M = CONFIG.env.max_oper_count, CONFIG.env.max_prod_count, CONFIG.env.max_eqp_count
    mask = env.action_masks()
    assert mask.shape == (O * P + M,)
    assert mask.any()


def test_tool_tracker_capacity():
    cap = {("LC01", "A"): 1}
    tracker = ToolTracker(cap, {"EQP001": "A"})
    assert tracker.can_assign("LC01", "EQP001")
    tracker.occupy("LC01", "EQP001")
    assert not tracker.can_assign("LC01", "EQP001")
    tracker.release("LC01", "EQP001")
    assert tracker.can_assign("LC01", "EQP001")


def test_sim_horizon_1440():
    raw = load_data()
    env_data = preprocess(raw)
    assert env_data["sim_end_minutes"] == CONFIG.env.hard_horizon_minutes
    assert env_data["soft_cutoff_minutes"] == CONFIG.env.soft_cutoff_minutes


def test_heuristic_episode_completes():
    raw = load_data()
    env_data = preprocess(raw)
    from agent.minprogress_agent import MinProgressAgent

    env = SchedulingEnv(env_data, record_history=False)
    agent = MinProgressAgent(env_data)
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < 5000:
        action = agent.predict(env.sim)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done
    assert len(env.get_schedule()) > 0


if __name__ == "__main__":
    test_obs_dim_matches_env()
    test_action_masks_shape()
    test_tool_tracker_capacity()
    test_sim_horizon_1440()
    test_heuristic_episode_completes()
    print("all tests passed")
