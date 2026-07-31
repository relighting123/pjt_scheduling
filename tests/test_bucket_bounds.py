"""config max_oper_count × max_prod_count 밖 버킷 인덱스 방어."""
from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


def _discrete_row(eqp_id, ppk, oper_id, carrier_id, wf_qty=25, st=60, model="A"):
    return {
        "EQP_ID": eqp_id,
        "LOT_ID": f"LOT_{carrier_id}",
        "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": ppk,
        "OPER_ID": oper_id,
        "ST": st,
        "EQP_MODEL_CD": model,
        "WF_QTY": wf_qty,
        "SEQ": 1,
        "LOT_STAT_CD": "WAIT",
    }


def _raw_with_many_products(n_products: int):
    ppks = [f"PPK{i:03d}" for i in range(1, n_products + 1)]
    discrete = [
        _discrete_row("EQP001", ppk, "OPER001", f"CAR{i:03d}")
        for i, ppk in enumerate(ppks, start=1)
    ]
    plan = [
        {
            "PLAN_PROD_ATTR_VAL": ppk,
            "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 50,
            "D1_PLAN_QTY": 50,
            "PLAN_PRIORITY": 1,
        }
        for ppk in ppks
    ]
    flow = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"}
        for ppk in ppks
    ]
    abstract = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 75}
        for ppk in ppks
    ]
    return {
        "discrete_arrange": discrete,
        "plan": plan,
        "flow": flow,
        "abstract_arrange": abstract,
    }


@pytest.fixture
def restore_config_axes():
    O, P = CONFIG.env.max_oper_count, CONFIG.env.max_prod_count
    yield
    CONFIG.env.max_oper_count = O
    CONFIG.env.max_prod_count = P


def test_ppk_oper_flat_index_clamps_to_config_bounds(restore_config_axes):
    n_bucket = CONFIG.env.max_oper_count * CONFIG.env.max_prod_count
    raw = _raw_with_many_products(CONFIG.env.max_prod_count + 1)
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    first_ppk = data["prod_keys"][0]
    overflow_ppk = data["prod_keys"][-1]

    assert sim.ppk_oper_flat_index("OPER001", first_ppk) == 0
    assert sim.ppk_oper_flat_index("OPER001", overflow_ppk) == -1

    feasible = sim.get_feasible_ppk_oper("EQP001")
    assert all(0 <= flat < n_bucket for flat in feasible)
    assert sim.ppk_oper_flat_index("OPER001", overflow_ppk) not in feasible


def test_action_masks_no_index_error_with_overflow_products(restore_config_axes):
    raw = _raw_with_many_products(CONFIG.env.max_prod_count + 1)
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    env = SchedulingRLEnv(
        data,
        record_history=False,
        record_event_log=False,
        record_decision_log=False,
    )
    env.reset()
    mask = env.action_masks()
    assert mask.shape[0] == env._n_bucket + env._L
    assert mask[: env._n_bucket].any()

    # 강제로 O×P 밖 블록 상태를 넣어도 IndexError 없이 마스크 생성
    env._block["EQP001"] = [env._n_bucket, 2, 2]
    mask = env.action_masks()
    assert mask[: env._n_bucket].any()
    assert "EQP001" not in env._block
