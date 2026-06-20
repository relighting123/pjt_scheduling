"""
data/pacing_scenarios.py – Takt/선형 생산 검증용 단순 시나리오 4종

목표: x=시간, y=누적 생산량이 (PPK, OPER)별 D0 계획 직선에 근접.
재공이 부족하면 앞공정에서 WIP를 쌓은 뒤 후공정이 직선을 따라가도록 유도.
"""
from typing import Callable, Dict, List, Optional, Tuple

from config import PERIOD_SPLITS, validate_path_segment
from data.generator import (
    _abstract_row,
    _discrete_row,
    build_split_rules,
    ensure_split_dirs,
    write_json_bundle,
)

TAKT_PLAN_QTY = 100
TAKT_WF = 25
TAKT_SOFT = 1320  # config.env.soft_cutoff_minutes 와 동일


def _flow_1(ppk: str, oper: str = "OPER001") -> List[dict]:
    return [{"PLAN_PROD_KEY": ppk, "SEQ_ID": 1, "OPER_ID": oper}]


def _flow_2(ppk: str) -> List[dict]:
    return [
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]


def _plan(
    ppk: str,
    opers: List[str],
    qty: int = TAKT_PLAN_QTY,
    priority: int = 1,
) -> List[dict]:
    return [
        {
            "PLAN_PROD_KEY": ppk,
            "OPER_ID": oper,
            "D0_PLAN_QTY": qty,
            "D1_PLAN_QTY": qty,
            "PLAN_PRIORITY": priority,
        }
        for oper in opers
    ]


def _abstract_routes(
    ppks: List[str],
    opers: List[str],
    st: int = 90,
    model: str = "A",
) -> List[dict]:
    return [_abstract_row(p, o, model, st) for p in ppks for o in opers]


def _lots_on_oper(
    discrete: List[dict],
    lot_ids: List[str],
    ppk: str,
    oper_id: str,
    eqps: Tuple[str, ...],
    st: int,
    wf: int = TAKT_WF,
    seq: int = 1,
) -> None:
    for i, lot_id in enumerate(lot_ids):
        for eqp_id in eqps:
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk, oper_id, st, wf,
                carrier_id=f"CAR{lot_id[-3:]}", seq=seq,
            ))


def build_takt_1p1o() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Case 1 – 단일 제품·단일 공정·2 EQP
    WIP 100매, 계획 100매 / 1320분 → 모든 알고리즘이 직선에 가깝게 가능해야 함.
    """
    ppk = "PPK001"
    discrete: List[dict] = []
    lot_ids = [f"LOT{i:03d}" for i in range(1, 5)]
    _lots_on_oper(discrete, lot_ids, ppk, "OPER001", ("EQP001", "EQP002"), st=66)
    return discrete, _plan(ppk, ["OPER001"]), _flow_1(ppk)


def build_takt_2stage() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Case 2 – 1 PPK, OPER001→OPER002 파이프라인
    초기 WIP는 OPER001만. OPER002는 완료 유입 후 직선 추종.
  """
    ppk = "PPK001"
    discrete: List[dict] = []
    lot_ids = [f"LOT{i:03d}" for i in range(1, 5)]
    _lots_on_oper(discrete, lot_ids, ppk, "OPER001", ("EQP001", "EQP002"), st=90, seq=1)
    return discrete, _plan(ppk, ["OPER001", "OPER002"]), _flow_2(ppk)


def build_takt_wip_buffer() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Case 3 – 느린 upstream(1 EQP) → WIP 축적 후 downstream 가속
    OPER001 ST=120 (EQP001 단독), OPER002 ST=60 (EQP002).
    """
    ppk = "PPK001"
    discrete: List[dict] = []
    lot_ids = [f"LOT{i:03d}" for i in range(1, 5)]
    for lot_id in lot_ids:
        discrete.append(_discrete_row(
            "EQP001", lot_id, ppk, "OPER001", 120, TAKT_WF,
            carrier_id=f"CAR{lot_id[-3:]}", seq=1,
        ))
        discrete.append(_discrete_row(
            "EQP002", lot_id, ppk, "OPER002", 60, TAKT_WF,
            carrier_id=f"CAR{lot_id[-3:]}", seq=2,
        ))
    return discrete, _plan(ppk, ["OPER001", "OPER002"]), _flow_2(ppk)


def build_takt_2ppk() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Case 4 – 2 PPK 경쟁, PPK_B OPER001 재공 50매 vs 계획 100매
    전환·앞공정 우선 vs pacing 트레이드오프 검증.
    """
    ppk_a, ppk_b = "PPK001", "PPK002"
    discrete: List[dict] = []
    lots_a = [f"LOT{i:03d}" for i in range(1, 5)]
    lots_b = [f"LOT{101 + i}" for i in range(2)]
    _lots_on_oper(discrete, lots_a, ppk_a, "OPER001", ("EQP001", "EQP002"), st=90)
    _lots_on_oper(discrete, lots_b, ppk_b, "OPER001", ("EQP001", "EQP002"), st=90)
    flow = _flow_2(ppk_a) + _flow_2(ppk_b)
    plan = _plan(ppk_a, ["OPER001", "OPER002"]) + _plan(ppk_b, ["OPER001", "OPER002"])
    return discrete, plan, flow


def abstract_takt_2stage() -> List[dict]:
    return _abstract_routes(["PPK001"], ["OPER001", "OPER002"], st=90)


def abstract_takt_wip_buffer() -> List[dict]:
    return [
        _abstract_row("PPK001", "OPER001", "A", 120),
        _abstract_row("PPK001", "OPER002", "A", 60),
    ]


def abstract_takt_2ppk() -> List[dict]:
    return _abstract_routes(["PPK001", "PPK002"], ["OPER001", "OPER002"], st=90)


TAKT_SCENARIOS: Dict[str, dict] = {
    "takt_1p1o": {
        "name": "Takt 1제품 1공정",
        "description": "PPK001·OPER001·2EQP·WIP100 — 직선 생산 기본 검증",
        "build": build_takt_1p1o,
        "configurable": False,
    },
    "takt_2stage": {
        "name": "Takt 2단 파이프라인",
        "description": "OPER001 WIP → OPER002 유입 후 직선 추종",
        "build": build_takt_2stage,
        "abstract_arrange": abstract_takt_2stage,
        "configurable": False,
    },
    "takt_wip_buffer": {
        "name": "Takt WIP 축적",
        "description": "느린 upstream 1대 → WIP 쌓고 downstream pacing",
        "build": build_takt_wip_buffer,
        "abstract_arrange": abstract_takt_wip_buffer,
        "configurable": False,
    },
    "takt_2ppk": {
        "name": "Takt 2제품 경쟁",
        "description": "PPK_B 재공 부족(50/100) — 전환 vs pacing",
        "build": build_takt_2ppk,
        "abstract_arrange": abstract_takt_2ppk,
        "configurable": False,
    },
}

# train 3 + test 1
TAKT_SUITE_LAYOUT = [
    ("takt_1p1o", "train", "20260101070000"),
    ("takt_2stage", "train", "20260102070000"),
    ("takt_wip_buffer", "train", "20260103070000"),
    ("takt_2ppk", "test", "20260104070000"),
]


def bootstrap_takt_suite(fac_id: str = "FAC_TAKT") -> dict:
    """4개 takt 시나리오 JSON 생성 (train 3 + test 1)."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    paths = []
    for scenario_id, split, period in TAKT_SUITE_LAYOUT:
        meta = TAKT_SCENARIOS[scenario_id]
        discrete, plan, flow = meta["build"]()
        abstract_fn = meta.get("abstract_arrange")
        abstract = abstract_fn() if callable(abstract_fn) else None
        if split in PERIOD_SPLITS:
            inp, out = ensure_split_dirs(fac_id, split, period)
        else:
            inp, out = ensure_split_dirs(fac_id, split, None)
        write_json_bundle(
            inp, discrete, plan, flow,
            split=build_split_rules(flow, split_qty=25),
            abstract_arrange=abstract,
        )
        paths.append({
            "scenario": scenario_id,
            "split": split,
            "period": period,
            "folder": f"{fac_id}/{split}/{period}",
            "input": str(inp),
        })
    return {"fac_id": fac_id, "paths": paths}
