"""SchedulingAgent.train()의 restore_best 옵션 검증.

PPO는 이미 좋은 정책을 찾은 뒤에도 계속 학습하면서 오히려 나빠질 수 있다는
게 실험으로 확인됐다(SYM_5x5: BC 워밍스타트로 1만 스텝에 전환 0회 도달 후
유지하다가 후반에 다시 나빠짐). train()은 학습 종료 후 EvalCallback이
저장해둔 best_model.zip이 있으면 그걸로 self.model을 되돌려야 한다
(restore_best=True, 기본값). restore_best=False면 학습이 끝난 시점의
모델을 그대로 둬야 한다.
"""
from pathlib import Path
from unittest.mock import patch

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from agent.rl_agent import SchedulingAgent

RULE_TIMEKEY = "20260712070000"


def _tiny_raw():
    return {
        "discrete_arrange": [{
            "EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 30,
            "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
        }],
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 1, "D1_PLAN_QTY": 1, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 30},
        ],
    }


def _load_ed():
    ed = preprocess(_tiny_raw(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 60
    ed["conversion_minutes"] = 30
    return ed


def _set_rl_config(tmp_path, eval_freq=32, total_timesteps=64):
    CONFIG.rl.total_timesteps = total_timesteps
    CONFIG.rl.n_steps = 32
    CONFIG.rl.eval_freq = eval_freq
    CONFIG.rl.device = "cpu"
    CONFIG.rl.n_envs = 1
    CONFIG.rl.bc_pretrain_epochs = 0
    CONFIG.path.model_dir = tmp_path


def test_restore_best_reloads_best_model_when_present(tmp_path):
    _set_rl_config(tmp_path)
    ed = _load_ed()

    from sb3_contrib import MaskablePPO
    original_load = MaskablePPO.load.__func__

    agent = SchedulingAgent()
    with patch("agent.rl_agent.MaskablePPO.load") as mocked_load:
        mocked_load.side_effect = lambda *a, **kw: original_load(MaskablePPO, *a, **kw)

        agent.train([ed], verbose=0, env_cls=SchedulingRLEnv, restore_best=True)

        best_path = tmp_path / "best" / "best_model.zip"
        if best_path.exists():
            mocked_load.assert_called_once()
            assert str(best_path) in str(mocked_load.call_args)
        else:
            # eval이 한 번도 안 돌았다면(운 나쁘게 타이밍이 안 맞으면) 호출 안 됨
            mocked_load.assert_not_called()


def test_restore_best_false_keeps_final_model(tmp_path):
    _set_rl_config(tmp_path)
    ed = _load_ed()

    agent = SchedulingAgent()
    with patch("agent.rl_agent.MaskablePPO.load") as mocked_load:
        agent.train([ed], verbose=0, env_cls=SchedulingRLEnv, restore_best=False)
        mocked_load.assert_not_called()
