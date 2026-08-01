"""TRAIN_POOL의 seed 재현성과 다공정 데이터 분리를 검증."""

import random

from benchmark import gen_tool_change_bench as multistage
from benchmark.gen_train_pool import (
    generate_multistage,
    sample_multistage_spec,
    sample_spec,
)
from config import CONFIG
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv


def test_train_specs_are_seed_reproducible_and_use_distinct_ids():
    a = random.Random(17)
    b = random.Random(17)
    assert [sample_spec(a, i) for i in range(8)] == [sample_spec(b, i) for i in range(8)]

    a = random.Random(17)
    b = random.Random(17)
    specs_a = [sample_multistage_spec(a, i) for i in range(4)]
    specs_b = [sample_multistage_spec(b, i) for i in range(4)]
    assert specs_a == specs_b
    assert all(s["id"].startswith("TP_MS_") for s in specs_a)
    assert all(s["stage1"]["oper"] != s["stage2"]["oper"] for s in specs_a)


def test_multistage_pool_case_builds_valid_rl_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(multistage, "SUITE_ROOT", tmp_path)
    spec = sample_multistage_spec(random.Random(3), 0)
    meta = generate_multistage(spec)

    raw = load_data(meta["dir"])
    env_data = preprocess(raw)
    env_data["sim_end_minutes"] = meta["sim"]
    env_data["conversion_minutes"] = meta["conv"]

    assert len(env_data["oper_ids"]) == 2
    assert not set(env_data["oper_ids"]) - {"OPER001", "OPER002"}
    assert len(env_data["prod_keys"]) <= CONFIG.env.max_prod_count
    env = SchedulingRLEnv(env_data, record_history=False, record_event_log=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.action_masks().any()
    env.close()
