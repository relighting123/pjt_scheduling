"""FAC_ID별 모델 저장/로드 분리 검증.

model_dir_for(fac_id)는 fac_id가 없으면 기존 공용 경로(CONFIG.path.model_dir)
그대로, 있으면 그 아래 FAC_ID별 하위 폴더를 반환해야 한다. SchedulingAgent의
save()/load()/model_exists()도 fac_id를 넘기면 그 폴더만 쓰고, 서로 다른
FAC_ID의 모델은 섞이지 않아야 한다.
"""
from config import CONFIG, model_dir_for
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from agent.rl_agent import SchedulingAgent

RULE_TIMEKEY = "20260712070000"


def test_model_dir_for_none_returns_legacy_shared_path():
    assert model_dir_for(None) == CONFIG.path.model_dir
    assert model_dir_for("") == CONFIG.path.model_dir


def test_model_dir_for_fac_id_returns_subfolder():
    assert model_dir_for("FAC001") == CONFIG.path.model_dir / "FAC001"
    assert model_dir_for("FAC002") == CONFIG.path.model_dir / "FAC002"


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


def test_save_and_load_are_isolated_per_fac_id(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG.path, "model_dir", tmp_path)
    CONFIG.rl.total_timesteps = 64
    CONFIG.rl.n_steps = 32
    CONFIG.rl.eval_freq = 1000  # 이번 테스트에선 eval/best 복원 여부는 무관
    CONFIG.rl.device = "cpu"
    CONFIG.rl.n_envs = 1
    CONFIG.rl.bc_pretrain_epochs = 0

    ed = _load_ed()

    agent_a = SchedulingAgent()
    agent_a.train([ed], verbose=0, env_cls=SchedulingRLEnv, restore_best=False)
    agent_a.save(fac_id="FAC001")

    # FAC001 하위에만 저장되고, 공용 경로(models/ 바로 아래)에는 없어야 한다
    assert (tmp_path / "FAC001" / f"{CONFIG.rl.model_name}.zip").exists()
    assert not (tmp_path / f"{CONFIG.rl.model_name}.zip").exists()

    # FAC001 모델은 FAC001로만 로드되고, FAC002로는 못 찾아야 한다
    loaded = SchedulingAgent.load(fac_id="FAC001")
    assert loaded.model is not None

    agent_probe = SchedulingAgent()
    assert agent_probe.model_exists(fac_id="FAC001") is True
    assert agent_probe.model_exists(fac_id="FAC002") is False
    assert agent_probe.model_exists() is False  # 공용 경로엔 저장한 적 없음

    # FAC002에 별도로 저장하면 FAC001과 섞이지 않아야 한다
    agent_b = SchedulingAgent()
    agent_b.train([ed], verbose=0, env_cls=SchedulingRLEnv, restore_best=False)
    agent_b.save(fac_id="FAC002")
    assert (tmp_path / "FAC002" / f"{CONFIG.rl.model_name}.zip").exists()
    assert agent_probe.model_exists(fac_id="FAC001") is True
    assert agent_probe.model_exists(fac_id="FAC002") is True
