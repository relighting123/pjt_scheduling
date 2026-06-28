"""
1제품·3공정·3장비 공정 전담 + 재공 충분 시나리오.

- PPK001 단일 제품, OPER001→OPER002→OPER003
- EQP001=OPER001, EQP002=OPER002, EQP003=OPER003 전용 (discrete eligibility 0)
- 계획 100매/공정, OPER001 초기 재공 150매 (계획 대비 충분)
- OPER002/003는 유입만 — 장비별 한 공정 전담·꾸준 생산(pacing) 검증용
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from data.generator import (
    _abstract_row,
    _discrete_row,
    build_batch_info_from_discrete,
    build_lot_master_from_discrete,
    build_split_rules,
    build_tool_capacity_from_lots,
    ensure_split_dirs,
    list_period_keys,
    validate_path_segment,
    write_json_bundle,
)

PPK = "PPK001"
OPERS = ("OPER001", "OPER002", "OPER003")
EQPS = ("EQP001", "EQP002", "EQP003")
EQP_OPER = dict(zip(EQPS, OPERS))
EQP_MODEL = {"EQP001": "A", "EQP002": "B", "EQP003": "C"}
WF_QTY = 25
ST = 4
PLAN_QTY = 100
OPER1_LOTS = 6  # 6 × 25 = 150 > PLAN_QTY


def _build_oper_dedicated_steady_sample() -> Tuple[List[dict], List[dict], List[dict]]:
    """1PPK·3OPER·3EQP 전담, OPER001 재공만 초기 보유."""
    flow = [
        {"PLAN_PROD_KEY": PPK, "OPER_SEQ": seq, "OPER_ID": op}
        for seq, op in enumerate(OPERS, start=1)
    ]
    plan = [
        {
            "PLAN_PROD_KEY": PPK,
            "OPER_ID": op,
            "D0_PLAN_QTY": PLAN_QTY,
            "D1_PLAN_QTY": PLAN_QTY,
            "PLAN_PRIORITY": 1,
        }
        for op in OPERS
    ]

    discrete: List[dict] = []
    for i in range(OPER1_LOTS):
        lot_id = f"LOT{i + 1:03d}"
        discrete.append(_discrete_row(
            "EQP001", lot_id, PPK, "OPER001", ST, WF_QTY,
            eqp_model="A", carrier_id=f"CAR{i + 1:03d}", seq=1,
        ))

    for eqp_id, oper_id in EQP_OPER.items():
        if eqp_id == "EQP001":
            continue
        cap_lot = f"CAP_{eqp_id}"
        seq = OPERS.index(oper_id) + 1
        discrete.append(_discrete_row(
            eqp_id, cap_lot, PPK, oper_id, ST, 1,
            eqp_model=EQP_MODEL[eqp_id],
            carrier_id=f"CAR_{eqp_id}", seq=seq,
        ))

    return discrete, plan, flow


def build_oper_dedicated_steady_abstract_arrange() -> List[dict]:
    return [
        _abstract_row(PPK, oper, EQP_MODEL[eqp], ST)
        for eqp, oper in EQP_OPER.items()
    ]


def build_oper_dedicated_steady_eqp_initial_state() -> List[dict]:
    return [
        {
            "eqp_id": eqp_id,
            "plan_prod_key": PPK,
            "oper_id": oper_id,
            "lot_cd": "LC001",
            "temp": "T650",
        }
        for eqp_id, oper_id in EQP_OPER.items()
    ]


def _raw_bundle() -> dict:
    discrete, plan, flow = _build_oper_dedicated_steady_sample()
    lot_master = build_lot_master_from_discrete(discrete)
    return {
        "discrete_arrange": discrete,
        "abstract_arrange": build_oper_dedicated_steady_abstract_arrange(),
        "plan": plan,
        "flow": flow,
        "split": build_split_rules(flow),
        "lot_master": lot_master,
        "batch_info": build_batch_info_from_discrete(discrete),
        "tool_capacity": build_tool_capacity_from_lots(lot_master),
        "eqp_initial_state": build_oper_dedicated_steady_eqp_initial_state(),
    }


DEDICATED_SCENARIOS: Dict[str, dict] = {
    "oper_dedicated_steady": {
        "name": "공정 전담 꾸준 생산 (1P×3O×3EQP)",
        "description": (
            "PPK001 단일, OPER001→002→003, 장비별 1공정 전담. "
            f"계획 {PLAN_QTY}매/공정, OPER001 재공 {OPER1_LOTS * WF_QTY}매."
        ),
        "build": _build_oper_dedicated_steady_sample,
        "configurable": False,
        "abstract_arrange": build_oper_dedicated_steady_abstract_arrange,
        "eqp_initial_state": build_oper_dedicated_steady_eqp_initial_state,
    },
}


def bootstrap_dedicated_suite(fac_id: str = "FAC_DEDICATED_TEST") -> dict:
    """train 1 + test 1 폴더 생성."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    train_key = list_period_keys(1)[0]
    test_key = list_period_keys(1, start_key=train_key)[0]
    paths = {}
    for split, per in (("train", train_key), ("test", test_key)):
        inp, out = ensure_split_dirs(fac_id, split, per)
        raw = _raw_bundle()
        write_json_bundle(
            inp,
            raw["discrete_arrange"],
            raw["plan"],
            raw["flow"],
            split=raw["split"],
            lot_master=raw["lot_master"],
            batch_info=raw["batch_info"],
            tool_capacity=raw["tool_capacity"],
            abstract_arrange=raw["abstract_arrange"],
            eqp_initial_state=raw["eqp_initial_state"],
        )
        paths[split] = {"input": str(inp), "output": str(out), "period": per}
    return {
        "fac_id": fac_id,
        "scenario": "oper_dedicated_steady",
        "paths": paths,
    }
