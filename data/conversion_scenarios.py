"""
data/conversion_scenarios.py – LOT_CD/TEMP conversion 검증용 시나리오

단일 공정, 2제품, EQP001은 PPK001(LC001/T650) 상태로 초기화,
PPK002(LC002/T700) 투입 시 conversion 필요.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from data.generator import (
    _abstract_row,
    _discrete_row,
    ensure_split_dirs,
    write_json_bundle,
)

CONV_SCENARIOS: Dict[str, dict] = {}


def build_conv_2ppk_1oper() -> Tuple[
    List[dict], List[dict], List[dict], List[dict], List[dict], List[dict],
]:
    """
    단일 OPER001, PPK001/PPK002, EQP001·EQP002 (MODEL A).
    EQP001 초기: LC001/T650 (PPK001 제품 상태).
    재공: 제품별 4 LOT × 25매 = 100매 (계획 100 충족).
    """
    oper = "OPER001"
    eqps = ("EQP001", "EQP002")
    st = 90

    lot_master: List[dict] = []
    discrete: List[dict] = []

    for i in range(1, 5):
        lot_id = f"LOT{i:03d}"
        lot_master.append({"LOT_ID": lot_id, "LOT_CD": "LC001", "TEMP": "T650"})
        for eqp in eqps:
            discrete.append(_discrete_row(
                eqp, lot_id, "PPK001", oper, st,
                carrier_id=f"CAR{i:03d}", seq=1, eqp_model="A",
            ))

    for i in range(5, 9):
        lot_id = f"LOT{i:03d}"
        lot_master.append({"LOT_ID": lot_id, "LOT_CD": "LC002", "TEMP": "T700"})
        for eqp in eqps:
            discrete.append(_discrete_row(
                eqp, lot_id, "PPK002", oper, st,
                carrier_id=f"CAR{i:03d}", seq=1, eqp_model="A",
            ))

    plan = [
        {
            "PLAN_PROD_KEY": "PPK001", "OPER_ID": oper,
            "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1,
        },
        {
            "PLAN_PROD_KEY": "PPK002", "OPER_ID": oper,
            "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1,
        },
    ]
    flow = [
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 1, "OPER_ID": oper},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 1, "OPER_ID": oper},
    ]
    abstract = [
        _abstract_row("PPK001", oper, "A", st),
        _abstract_row("PPK002", oper, "A", st),
    ]
    eqp_initial_state = [
        {
            "EQP_ID": "EQP001",
            "LOT_CD": "LC001",
            "TEMP": "T650",
            "PLAN_PROD_KEY": "PPK001",
            "OPER_ID": oper,
        },
    ]
    tool_capacity = [
        {"LOT_CD": "LC001", "EQP_MODEL": "A", "MAX_TOOL": 2},
        {"LOT_CD": "LC002", "EQP_MODEL": "A", "MAX_TOOL": 2},
    ]
    return discrete, plan, flow, lot_master, abstract, eqp_initial_state, tool_capacity


def bootstrap_conv_test_suite(
    fac_id: str = "FAC_CONV",
    *,
    period: str = "20260105070000",
) -> dict:
    """external/dataset/FAC_CONV/test/{period}/input 생성."""
    discrete, plan, flow, lot_master, abstract, eqp_init, tool_cap = build_conv_2ppk_1oper()
    inp, _ = ensure_split_dirs(fac_id, "test", period)
    write_json_bundle(
        inp,
        discrete,
        plan,
        flow,
        lot_master=lot_master,
        tool_capacity=tool_cap,
        abstract_arrange=abstract,
        eqp_initial_state=eqp_init,
    )
    return {
        "fac_id": fac_id,
        "period": period,
        "input_dir": str(inp),
        "folder": f"{fac_id}/test/{period}",
    }


def _build_conv_2ppk_1oper_sample() -> Tuple[List[dict], List[dict], List[dict]]:
    discrete, plan, flow, *_rest = build_conv_2ppk_1oper()
    return discrete, plan, flow


CONV_SCENARIOS["conv_2ppk_1oper"] = {
    "name": "Conversion 2PPK 1OPER",
    "description": (
        "단일 공정·2제품. EQP001은 PPK001(LC001/T650) 상태. "
        "PPK002 투입 시 conversion 검증."
    ),
    "build": _build_conv_2ppk_1oper_sample,
    "full_build": build_conv_2ppk_1oper,
    "abstract_arrange": lambda: build_conv_2ppk_1oper()[4],
}