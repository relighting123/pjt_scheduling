"""Takt pacing 시나리오 4종 + 파이프라인 smoke test."""
import numpy as np
import pytest

from config import CONFIG
from data.loader import validate_data
from data.preprocessor import normalize_raw, preprocess
from data.pacing_scenarios import TAKT_SCENARIOS, bootstrap_takt_suite
from env.scheduling_env import SchedulingEnv
from inference.runner import run_inference
from validation.pacing_metrics import pacing_metrics


def _raw_from_builder(scenario_id: str) -> dict:
    from data.generator import build_abstract_arrange

    meta = TAKT_SCENARIOS[scenario_id]
    discrete, plan, flow = meta["build"]()
    abstract_fn = meta.get("abstract_arrange")
    if callable(abstract_fn):
        abstract = abstract_fn()
    else:
        abstract = build_abstract_arrange(discrete, flow)
    return {
        "schedule": [],
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": [],
        "tool_capacity": [],
    }


@pytest.mark.parametrize("scenario_id", list(TAKT_SCENARIOS))
def test_takt_scenario_preprocess_and_episode(scenario_id: str):
    raw = normalize_raw(_raw_from_builder(scenario_id))
    assert not validate_data(raw), validate_data(raw)
    env_data = preprocess(raw)
    assert env_data["lots"]
    assert env_data["abstract_wip_init"]

    from agent.minprogress_agent import MinProgressAgent

    env = SchedulingEnv(env_data, record_history=False)
    agent = MinProgressAgent(env_data)
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < 5000:
        action = agent.predict(env.sim)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
    if scenario_id == "takt_2ppk":
        # PPK_B 재공 부족 시 is_done() 지연 가능 (WIP 잔존) — 배정 발생만 확인
        assert len(env.get_schedule()) >= 6
    else:
        assert done
    assert len(env.get_schedule()) > 0


@pytest.mark.parametrize("scenario_id", list(TAKT_SCENARIOS))
def test_takt_heuristic_pacing_metrics(scenario_id: str):
    env_data = preprocess(normalize_raw(_raw_from_builder(scenario_id)))
    result = run_inference(env_data, algorithm="minprogress", record_history=False)
    m = pacing_metrics(
        result["schedule"],
        env_data["plan"],
        horizon=env_data["soft_cutoff_minutes"],
    )
    assert m["mae"] >= 0
    if scenario_id == "takt_1p1o":
        assert m["final_gap"] >= 0.8, f"단순 케이스 달성률 낮음: {m}"


def test_bootstrap_takt_suite_paths():
    info = bootstrap_takt_suite(fac_id="FAC_TAKT_TEST")
    assert len(info["paths"]) == 4
