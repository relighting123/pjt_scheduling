"""일반화 벤치마크의 공정·제품·모델 축이 RL 공간에 들어오는지 검증."""

from benchmark.tool_change_bench import META, load_ed
from config import CONFIG
from env.scheduling_env import compute_obs_dim
from env.scheduling_rl_env import SchedulingRLEnv


def test_default_axes_cover_all_tool_change_cases():
    assert CONFIG.env.max_oper_count >= 4
    assert CONFIG.env.max_prod_count >= 7
    assert CONFIG.env.max_model_count >= 6

    for meta in META:
        env = SchedulingRLEnv(load_ed(meta), record_history=False, record_event_log=False)
        assert env.observation_space.shape == (compute_obs_dim(),)
        assert env.action_space.nvec.tolist() == [
            CONFIG.env.max_oper_count * CONFIG.env.max_prod_count,
            env._L,
        ]
        env.close()
