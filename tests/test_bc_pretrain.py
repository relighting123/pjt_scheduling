"""DedicationAgent 시연 기반 PPO 모방학습(BC) 워밍스타트 검증.

agent/rl_agent.py의 _collect_expert_transitions()/_behavior_clone_pretrain()가
- SchedulingRLEnv의 [bucket, size_level] 액션 형식에 맞게 전문가 시연을
  올바르게 모으는지(각 액션이 그 시점의 action_mask에서 허용된 버킷인지)
- 그 시연으로 정책을 지도학습하면 실제로 loss가 줄어드는지
를 확인한다. 대칭 2설비×2제품(각 제품 전담 설비에 홈 배정)의 작은 데이터셋을
써서 SYM_5x5 전체를 쓰지 않고도 빠르게 검증한다.
"""
import numpy as np

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from agent.rl_agent import SchedulingAgent, _collect_expert_transitions, _behavior_clone_pretrain

RULE_TIMEKEY = "20260712070000"


def _sym_2x2_raw():
    """EQP001<->PPK001, EQP002<->PPK002 전담 시 전환 0회가 최적인 대칭 케이스."""
    discrete = []
    plan = []
    flow = []
    abstract = []
    for i, (eqp, ppk) in enumerate([("EQP001", "PPK001"), ("EQP002", "PPK002")]):
        for c in range(2):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 2, "D1_PLAN_QTY": 2, "PLAN_PRIORITY": 1,
        })
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
            "EQP_MODEL_CD": "A", "ST": 30,
        })
    return {
        "discrete_arrange": discrete, "plan": plan, "flow": flow,
        "abstract_arrange": abstract,
    }


def _load_ed():
    ed = preprocess(_sym_2x2_raw(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 120
    ed["conversion_minutes"] = 30
    return ed


def test_collect_expert_transitions_actions_are_within_mask():
    ed = _load_ed()
    obs, action, mask = _collect_expert_transitions(ed, SchedulingRLEnv)

    assert obs.shape[0] > 0
    assert obs.shape[0] == action.shape[0] == mask.shape[0]

    n_bucket = SchedulingRLEnv(ed, record_history=False, record_event_log=False)._n_bucket
    for i in range(len(action)):
        bucket, size_level = action[i]
        assert mask[i, bucket], f"step {i}: bucket {bucket} not in mask"
        assert 0 <= bucket < n_bucket


def test_collect_expert_transitions_reaches_zero_conversions():
    """DedicationAgent 시연 자체가 이 대칭 데이터에서 전환 0회로 끝나야 한다
    (모방 대상이 실제로 정답이라는 전제 확인)."""
    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    _obs, _action, _mask = _collect_expert_transitions(ed, SchedulingRLEnv)
    # _collect_expert_transitions 내부에서 새 env를 만들어 굴리므로, 여기서는
    # 별도로 같은 로직을 한 번 더 돌려 최종 conversions만 확인한다.
    from agent.dedication_agent import DedicationAgent, HOLD_ACTION

    obs, _ = env.reset()
    expert = DedicationAgent(ed)
    for _ in range(2000):
        mask = env.action_masks()
        eqp_id = env.sim.current_idle_eqp()
        choice = int(expert.predict(env.sim)[0]) if eqp_id is not None else 0
        if choice == HOLD_ACTION:
            committed = expert._committed.get(eqp_id) if eqp_id is not None else None
            if committed is not None:
                choice = committed
            else:
                feasible = np.flatnonzero(mask[: env._n_bucket])
                choice = int(feasible[0]) if feasible.size else 0
        obs, _r, terminated, truncated, _info = env.step(
            np.array([choice, env._L - 1], dtype=np.int64)
        )
        if terminated or truncated:
            break
    assert env.sim.stats["conversions"] == 0
    assert sum(env.sim.stats["completed_qty"].values()) == 4


def test_behavior_clone_pretrain_reduces_loss():
    ed = _load_ed()
    CONFIG.rl.n_steps = 64
    CONFIG.rl.device = "cpu"
    CONFIG.rl.n_envs = 1

    agent = SchedulingAgent()
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from agent.rl_agent import _mask_fn

    def make_env():
        return Monitor(ActionMasker(
            SchedulingRLEnv(ed, record_history=False, record_event_log=False), _mask_fn,
        ))

    train_env = DummyVecEnv([make_env])
    model = MaskablePPO("MlpPolicy", train_env, n_steps=64, batch_size=8, device="cpu", verbose=0)

    obs, action, mask = _collect_expert_transitions(ed, SchedulingRLEnv)
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.long)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)

    model.policy.set_training_mode(True)
    with torch.no_grad():
        _v, log_prob_before, _e = model.policy.evaluate_actions(obs_t, action_t, mask_t)
    loss_before = -log_prob_before.mean().item()

    _behavior_clone_pretrain(model, [ed], SchedulingRLEnv, epochs=50, lr=1e-2, verbose=0)

    with torch.no_grad():
        _v, log_prob_after, _e = model.policy.evaluate_actions(obs_t, action_t, mask_t)
    loss_after = -log_prob_after.mean().item()

    assert loss_after < loss_before


def test_scheduling_agent_train_with_bc_pretrain_enabled_runs_end_to_end():
    """CONFIG.rl.bc_pretrain_epochs>0으로 SchedulingAgent.train()을 실제 경로
    그대로(옵트인 설정) 돌려도 에러 없이 끝나는지 확인 (배선 자체의 회귀 방지)."""
    ed = _load_ed()
    original = (
        CONFIG.rl.bc_pretrain_epochs, CONFIG.rl.bc_pretrain_lr,
        CONFIG.rl.total_timesteps, CONFIG.rl.n_steps, CONFIG.rl.device, CONFIG.rl.n_envs,
    )
    try:
        CONFIG.rl.bc_pretrain_epochs = 5
        CONFIG.rl.bc_pretrain_lr = 1e-2
        CONFIG.rl.total_timesteps = 64
        CONFIG.rl.n_steps = 64
        CONFIG.rl.device = "cpu"
        CONFIG.rl.n_envs = 1

        agent = SchedulingAgent()
        agent.train([ed], verbose=0, env_cls=SchedulingRLEnv)
        assert agent.model is not None
    finally:
        (
            CONFIG.rl.bc_pretrain_epochs, CONFIG.rl.bc_pretrain_lr,
            CONFIG.rl.total_timesteps, CONFIG.rl.n_steps, CONFIG.rl.device, CONFIG.rl.n_envs,
        ) = original
