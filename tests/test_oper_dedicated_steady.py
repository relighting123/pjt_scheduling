"""1PPK·3OPER·3EQP 공정 전담 시나리오 — 데이터·학습·추론 검증."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pytest

from agent.rl_agent import SchedulingAgent
from config import CONFIG
from data.dedicated_scenarios import (
    DEDICATED_SCENARIOS,
    EQP_OPER,
    PLAN_QTY,
    bootstrap_dedicated_suite,
)
from data.loader import preprocess, validate_data
from env.scheduling_env import SchedulingEnv, compute_obs_dim
from inference.runner import run_inference


def _raw_from_builder() -> dict:
    from data.generator import build_abstract_arrange

    meta = DEDICATED_SCENARIOS["oper_dedicated_steady"]
    discrete, plan, flow = meta["build"]()
    abstract_fn = meta.get("abstract_arrange")
    abstract = abstract_fn() if callable(abstract_fn) else build_abstract_arrange(discrete, flow)
    eqp_init_fn = meta.get("eqp_initial_state")
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": [],
        "tool_capacity": [],
    }
    if callable(eqp_init_fn):
        raw["eqp_initial_state"] = eqp_init_fn()
    return raw


def oper_completed_qty(
    schedule: List[dict],
    *,
    horizon: int | None = None,
) -> Dict[Tuple[str, str], int]:
    """(PPK, OPER)별 완료 수량 (END_TM 기준)."""
    totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in schedule:
        end = int(row.get("END_TM", 0))
        if horizon is not None and end > horizon:
            continue
        key = (row["PLAN_PROD_ATTR_VAL"], row["OPER_ID"])
        totals[key] += int(row.get("WF_QTY", 0))
    return dict(totals)


def dedication_fraction(schedule: List[dict]) -> float:
    """장비가 전담 공정만 처리한 비율."""
    if not schedule:
        return 0.0
    ok = sum(
        1 for r in schedule
        if EQP_OPER.get(r["EQP_ID"]) == r["OPER_ID"]
    )
    return ok / len(schedule)


def plan_achievement_ratio(
    completed: Dict[Tuple[str, str], int],
    plan: List[dict],
) -> Dict[str, float]:
    """공정별 계획 대비 달성률."""
    ratios: Dict[str, float] = {}
    for p in plan:
        op = p.get("oper_id") or p.get("OPER_ID")
        ppk = p.get("plan_prod_attr_val") or p.get("PLAN_PROD_ATTR_VAL")
        target = max(int(p.get("d0_plan_qty", p.get("D0_PLAN_QTY", 0))), 1)
        done = completed.get((ppk, op), 0)
        ratios[op] = done / target
    return ratios


def test_oper_dedicated_steady_preprocess():
    raw = _raw_from_builder()
    assert not validate_data(raw), validate_data(raw)
    env_data = preprocess(raw)
    assert env_data["eqp_ids"] == ["EQP001", "EQP002", "EQP003"]
    assert len(env_data["oper_ids"]) >= 3
    wip_init = env_data.get("abstract_wip_init", {})
    oper1_wip = wip_init.get(("PPK001", "OPER001"), {}).get("wip_qty_init", 0)
    assert oper1_wip >= 6, f"OPER001 재공 LOT 부족: {oper1_wip}"
    oper1_wafers = oper1_wip * 25
    assert oper1_wafers >= 150, f"OPER001 재공 매수 부족: {oper1_wafers}"


def test_oper_dedicated_steady_obs_dim():
    env_data = preprocess(_raw_from_builder())
    env = SchedulingEnv(env_data)
    obs, _ = env.reset()
    assert obs.shape == (compute_obs_dim(),)


def test_oper_dedicated_heuristic_dedication_and_pacing():
    env_data = preprocess(_raw_from_builder())
    horizon = env_data["soft_cutoff_minutes"]
    result = run_inference(
        env_data, algorithm="minprogress", record_history=False,
        enable_wip_inflow=True,
    )
    sched = result["schedule"]
    assert len(sched) >= 6
    assert dedication_fraction(sched) >= 0.95, "장비가 전담 공정 외 배정"
    completed = oper_completed_qty(sched, horizon=horizon)
    ratios = plan_achievement_ratio(completed, env_data["plan"])
    for op in ("OPER001", "OPER002", "OPER003"):
        assert ratios.get(op, 0) >= 0.9, f"{op} 계획 달성률 낮음: {ratios}"


def test_oper_dedicated_scheduling_rl_train_and_infer():
    """RL 학습 후 전담·pacing 유지 확인."""
    raw = _raw_from_builder()
    env_data = preprocess(raw)
    horizon = env_data["soft_cutoff_minutes"]

    saved_ts = CONFIG.rl.total_timesteps
    saved_eval = CONFIG.rl.eval_freq
    CONFIG.rl.total_timesteps = 30_000
    CONFIG.rl.eval_freq = 10_000
    try:
        agent = SchedulingAgent()
        agent.train(env_data, verbose=0)
        result = run_inference(
            env_data,
            algorithm="scheduling_rl",
            agent=agent,
            record_history=False,
            deterministic=True,
            enable_wip_inflow=True,
        )
    finally:
        CONFIG.rl.total_timesteps = saved_ts
        CONFIG.rl.eval_freq = saved_eval

    sched = result["schedule"]
    assert len(sched) >= 6
    assert dedication_fraction(sched) >= 0.90
    completed = oper_completed_qty(sched, horizon=horizon)
    ratios = plan_achievement_ratio(completed, env_data["plan"])
    assert ratios.get("OPER001", 0) >= 0.8
    assert sum(ratios.values()) / max(len(ratios), 1) >= 0.8


def test_bootstrap_dedicated_suite_paths():
    info = bootstrap_dedicated_suite(fac_id="FAC_DED_TEST")
    assert "train" in info["paths"]
    assert "test" in info["paths"]
