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
    O, P = CONFIG.env.max_oper_count, CONFIG.env.max_prod_count
    mask = env.action_masks()
    assert mask.shape == (O * P,)
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


def test_pacing_shaping_reward_when_behind_plan():
    """계획 있는 (PPK, OPER)이 직선보다 뒤처졌을 때 pacing 보상 > 0."""
    raw = load_data()
    env_data = preprocess(raw)
    sim = SchedulingSimulator(env_data, record_history=False)
    ppk = env_data["prod_keys"][0]
    oper = env_data["oper_ids"][0]
    if not sim._has_plan(ppk, oper):
        return
    sim.current_time = sim.soft_cutoff // 2
    plan_qty = env_data["plan_meta"][(ppk, oper)]["d0_plan_qty"]
    sim.stats["completed_qty"][(ppk, oper)] = 0
    r = sim._pacing_shaping_reward(ppk, oper, wf_qty=min(25, plan_qty))
    assert r > 0, f"expected positive pacing reward when behind plan, got {r}"


def test_pacing_shaping_skipped_without_plan():
    """계획 없는 (PPK, OPER)은 pacing 보상 0."""
    sim = SchedulingSimulator(preprocess(load_data()), record_history=False)
    assert sim._pacing_shaping_reward("PPK_NO_PLAN", "OPER001", wf_qty=25) == 0.0


def test_same_prod_skipped_when_ppk_not_feasible():
    """이전 PPK에 feasible 조합이 없으면 same_prod 보너스 없음."""
    raw = load_data()
    env_data = preprocess(raw)
    sim = SchedulingSimulator(env_data, record_history=False)
    eqp_id = env_data["eqp_ids"][0]
    eqp = sim.eqps[eqp_id]
    eqp.prev_prod = "PPK_NO_FEASIBLE"
    r = sim._same_prod_reward(eqp, "PPK_NO_FEASIBLE")
    assert r == 0.0


def test_pacing_steady_scenario_preprocess_and_episode():
    """pacing_steady 샘플: 전처리·에피소드 완료·OPER002 abstract 유입 확인."""
    from data.generator import (
        _build_pacing_steady_sample,
        build_pacing_steady_abstract_arrange,
        build_split_rules,
    )

    discrete, plan, flow = _build_pacing_steady_sample()
    abstract = build_pacing_steady_abstract_arrange()
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": build_split_rules(flow),
        "lot_master": [],
        "tool_capacity": [],
    }
    env_data = preprocess(raw)
    routes = env_data["abstract_routes_by_ppk_oper"]
    assert ("PPK001", "OPER002") in routes
    assert ("PPK002", "OPER002") in routes
    assert env_data["plan_meta"][("PPK002", "OPER001")]["d0_plan_qty"] == 100
    assert env_data["abstract_inventory"], "abstract 템플릿이 있어야 함"
    wip = env_data["abstract_wip_init"]
    assert wip[("PPK001", "OPER001")]["wip_qty"] > 0
    assert wip[("PPK002", "OPER001")]["wip_qty"] > 0
    assert wip.get(("PPK001", "OPER002"), {}).get("wip_qty", 0) == 0

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
    completed = env.sim.stats["completed_qty"]
    assert completed.get(("PPK001", "OPER002"), 0) > 0, "PPK001 OPER002 생산이 있어야 함"


def test_step_resolves_invalid_ppk_on_current_eqp():
    """invalid bucket이어도 현재 EQP feasible 조합으로 할당."""
    raw = load_data()
    env_data = preprocess(raw)
    env = SchedulingEnv(env_data, record_history=False)
    env.reset()
    eqp_id = env.sim.current_idle_eqp()
    if eqp_id is None:
        return
    feasible = env.sim.get_feasible_ppk_oper(eqp_id)
    if not feasible:
        return
    target_flat = feasible[0]
    wrong_flat = (feasible[-1] + 1) % (CONFIG.env.max_oper_count * CONFIG.env.max_prod_count)
    if wrong_flat in feasible:
        wrong_flat = (wrong_flat + 1) % (CONFIG.env.max_oper_count * CONFIG.env.max_prod_count)
    obs, reward, term, trunc, _ = env.step(np.array([wrong_flat], dtype=np.int64))
    assert reward > 0 or env.sim.schedule, "feasible 있을 때 할당이 진행되어야 함"
    assert obs is not None
    assert not (term and trunc)


def test_rl_inference_makes_progress():
    """RL 추론이 max_steps까지 idle 반복하지 않고 스케줄을 생성."""
    from agent.rl_agent import SchedulingAgent
    from inference.runner import run_inference

    raw = load_data()
    env_data = preprocess(raw)
    if not SchedulingAgent().model_exists():
        return
    agent = SchedulingAgent.load()
    result = run_inference(env_data, agent=agent, record_history=True)
    assert len(result["schedule"]) > 1
    assert result["history"][-1]["time"] > 0 or len(result["schedule"]) >= len(env_data["lots"])


if __name__ == "__main__":
    test_obs_dim_matches_env()
    test_action_masks_shape()
    test_tool_tracker_capacity()
    test_sim_horizon_1440()
    test_heuristic_episode_completes()
    test_pacing_shaping_reward_when_behind_plan()
    test_pacing_shaping_skipped_without_plan()
    test_same_prod_skipped_when_ppk_not_feasible()
    test_pacing_steady_scenario_preprocess_and_episode()
    test_step_resolves_invalid_ppk_on_current_eqp()
    test_rl_inference_makes_progress()
    print("all tests passed")
