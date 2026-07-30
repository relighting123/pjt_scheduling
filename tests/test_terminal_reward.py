"""SchedulingRLEnv – 종단(terminal) KPI 보상 · 결정 컨텍스트 관측 검증.

step reward가 전부 대리지표(same_setup / bulk_block_bonus / …)라
"보상 최고 = 벤치마크 KPI 최고"가 보장되지 않던 문제를, 에피소드 종료
시점에 실제 KPI(생산 carrier 수, 전환 총횟수)를 직접 보상으로 주어 메운다.
"""
import numpy as np
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_env import compute_obs_dim
from env.scheduling_rl_env import (
    RL_EXTRA_OBS_DIM,
    SchedulingRLEnv,
    rl_obs_dim,
)

RULE_TIMEKEY = "20260712070000"


def _sym_2x2_raw():
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate([("EQP001", "PPK001"), ("EQP002", "PPK002")]):
        for c in range(3):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": 3, "D1_PLAN_QTY": 3, "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                         "EQP_MODEL_CD": "A", "ST": 30})
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed():
    ed = preprocess(_sym_2x2_raw(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 180
    ed["conversion_minutes"] = 30
    return ed


@pytest.fixture
def reward_cfg():
    r = CONFIG.reward
    original = (r.w_terminal_throughput, r.w_terminal_conversion, r.terminal_reward_clip)
    yield r
    (r.w_terminal_throughput, r.w_terminal_conversion, r.terminal_reward_clip) = original


def _run_episode(ed):
    """첫 feasible 버킷을 계속 고르는 단순 정책으로 1 에피소드."""
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    env.reset()
    last_info, last_reward = {}, 0.0
    for _ in range(int(ed["sim_end_minutes"]) + 500):
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, env._L - 1],
                          dtype=np.int64)
        _obs, last_reward, terminated, truncated, last_info = env.step(action)
        if terminated or truncated:
            break
    return env, last_reward, last_info


def test_terminal_reward_only_at_episode_end(reward_cfg):
    reward_cfg.w_terminal_throughput = 30.0
    reward_cfg.w_terminal_conversion = -1.0

    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    env.reset()
    step_clip = CONFIG.reward.reward_clip

    done = False
    steps = 0
    while not done and steps < 500:
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, 0], dtype=np.int64)
        _o, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if not done:
            # 중간 스텝은 step clip 범위를 벗어날 수 없다
            assert abs(reward) <= step_clip + 1e-6
            assert "terminal_reward" not in info

    assert done
    assert "terminal_reward" in info


def test_terminal_reward_scales_with_production(reward_cfg):
    reward_cfg.w_terminal_throughput = 30.0
    reward_cfg.w_terminal_conversion = 0.0
    reward_cfg.terminal_reward_clip = 0.0

    ed = _load_ed()
    env, _reward, info = _run_episode(ed)

    expected = 30.0 * min(
        info["produced_carriers"] / max(info["producible_carriers"], 1), 1.0,
    )
    assert env._terminal_reward == pytest.approx(expected, abs=1e-6)


def test_terminal_reward_penalizes_conversions(reward_cfg):
    ed = _load_ed()

    reward_cfg.w_terminal_throughput = 0.0
    reward_cfg.w_terminal_conversion = 0.0
    reward_cfg.terminal_reward_clip = 0.0
    env0, _r, _i = _run_episode(ed)
    assert env0._terminal_reward == 0.0

    reward_cfg.w_terminal_conversion = -2.0
    env1, _r, _i = _run_episode(ed)
    assert env1._terminal_reward == pytest.approx(
        -2.0 * env1.sim.stats.get("conversions", 0), abs=1e-6,
    )


def test_terminal_reward_respects_clip(reward_cfg):
    reward_cfg.w_terminal_throughput = 1000.0
    reward_cfg.w_terminal_conversion = 0.0
    reward_cfg.terminal_reward_clip = 5.0

    ed = _load_ed()
    env, _reward, _info = _run_episode(ed)
    assert abs(env._terminal_reward) <= 5.0


def test_terminal_reward_disabled_by_zero_weights(reward_cfg):
    reward_cfg.w_terminal_throughput = 0.0
    reward_cfg.w_terminal_conversion = 0.0

    ed = _load_ed()
    env, _reward, _info = _run_episode(ed)
    assert env._terminal_reward == 0.0


# ── 결정 컨텍스트 관측 ───────────────────────────────────────────────────────

def test_obs_dim_is_base_plus_decision_context():
    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    obs, _ = env.reset()

    assert obs.shape[0] == compute_obs_dim() + RL_EXTRA_OBS_DIM
    assert rl_obs_dim() == obs.shape[0]
    assert env.observation_space.shape[0] == obs.shape[0]


def test_decision_context_carries_no_equipment_identity():
    """결정 컨텍스트는 호기 식별자를 담지 않아야 한다.

    `(장비 순번)/(장비 수)` 같은 채널을 넣으면 정책이 "몇 번 호기가 무슨 제품"을
    외워버리고, 장비가 추가·제외되면 인덱스가 밀려 그 매핑이 통째로 깨진다.
    같은 성질(같은 모델·같은 셋업 상태·블록 없음)의 장비들은 결정 컨텍스트가
    서로 구별되지 않아야 한다 — 구별은 시뮬레이터 상태(WIP·cover)가 한다.
    """
    ed = _load_ed()   # EQP001/EQP002 둘 다 모델 A, 초기 셋업 없음
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    env.reset()

    ctx = {e: env._decision_context_features(e).copy() for e in ed["eqp_ids"]}
    first = ctx[ed["eqp_ids"][0]]
    for eqp, vec in ctx.items():
        assert np.array_equal(vec, first), (
            f"{eqp}의 결정 컨텍스트가 다르다 — 호기 종속 채널이 섞였는지 확인: {vec}"
        )


def test_sequential_rollout_gives_distinct_observations_per_equipment():
    """호기 채널 없이도 장비별 관측이 실제로 갈리는지.

    결정은 순차적이라 앞 장비가 배정하면 WIP·cover가 바뀐다 — 대칭 케이스에서도
    뒤 장비는 이미 다른 상태를 본다. 이게 호기 인덱스를 넣지 않아도 되는 근거다.
    """
    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    obs, _ = env.reset()

    seen: list = []
    for _ in range(120):
        mask = env.action_masks()
        eqp = env.sim.current_idle_eqp()
        if eqp is not None:
            seen.append((eqp, obs.copy()))
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, env._L - 1],
                          dtype=np.int64)
        obs, _r, terminated, truncated, _i = env.step(action)
        if terminated or truncated:
            break

    assert len(seen) >= 2
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            if seen[i][0] != seen[j][0]:
                assert not np.array_equal(seen[i][1], seen[j][1]), (
                    f"{seen[i][0]}와 {seen[j][0]}의 관측이 동일 — 구분 불가"
                )


def test_decision_context_stays_in_unit_range_and_flags_block():
    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    obs, _ = env.reset()

    saw_decision_eqp = False
    for _ in range(200):
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, env._L - 1],
                          dtype=np.int64)
        obs, _r, terminated, truncated, info = env.step(action)
        if info.get("current_eqp") is not None:
            saw_decision_eqp = True
            assert obs[-RL_EXTRA_OBS_DIM] == 1.0     # 채널 0: 결정 장비 있음
        if terminated or truncated:
            break

    assert saw_decision_eqp
