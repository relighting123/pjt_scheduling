"""
benchmark/optimal/cases.py — 증명 가능한 최적해를 가진 벤치마크 케이스

각 케이스는 "정답이 이 값임을 수학적으로 증명할 수 있는" 만큼 의도적으로
단순화된 시나리오다 (증명은 각 케이스 옆 주석 + benchmark/optimal/proofs.py
참고). data.generator의 저수준 row 빌더로 env_data를 메모리에서 직접
구성하므로 파일 I/O가 없고, agent/env/simulation/data 등 기존 코드는
전혀 수정하지 않는다 — run_inference()가 이미 받는 형태의 env_data만
만들어 넘길 뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from data.generator import (
    _abstract_row,
    _discrete_row,
    build_batch_info_from_discrete,
    build_lot_master_from_discrete,
    build_split_rules,
    build_tool_capacity_from_lots,
)
from data.loader.preprocess import preprocess

from benchmark.optimal.proofs import capacity_bound


@dataclass(frozen=True)
class OptimalTarget:
    production: int
    conversions: int
    proof: str


@dataclass(frozen=True)
class CaseMetrics:
    production: int
    conversions: int


@dataclass(frozen=True)
class OptimalCase:
    id: str
    description: str
    build: Callable[[], dict]
    optimal: OptimalTarget


def measure(result: dict, sim_end: int) -> CaseMetrics:
    """run_inference() 결과에서 케이스 채점에 필요한 지표만 추출."""
    in_time = [r for r in result["schedule"] if r.get("END_TM", 0) <= sim_end]
    return CaseMetrics(
        production=len(in_time),
        conversions=result["stats"].get("conversions", 0),
    )


def _bundle(
    discrete: list, plan: list, flow: list, abstract_arrange: list,
    eqp_initial_state: list | None = None,
) -> dict:
    lot_master = build_lot_master_from_discrete(discrete)
    return {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract_arrange,
        "plan": plan,
        "flow": flow,
        "split": build_split_rules(flow),
        "lot_master": lot_master,
        "batch_info": build_batch_info_from_discrete(discrete),
        "tool_capacity": build_tool_capacity_from_lots(lot_master),
        "eqp_initial_state": eqp_initial_state or [],
    }


# ── 케이스 1: 전담 배정 (일반화된 CONV_BENCH) ────────────────────────────────
# EQP N대 × PPK N종(각기 다른 LOT_CD), ST=Conv=60분, sim=480분 → EQP당 8건.
# 상한 = N × floor(480/60)는 케이스1과 같은 논리를 EQP마다 적용한 합이며,
# PPK-LOT_CD가 1:1이므로 "EQP 1대 = PPK 1종 전담"일 때만 전환 0회로
# 동시에 달성된다. 초기 discrete_arrange는 라운드로빈으로 PPK를 여러 EQP에
# 일부러 섞어 배정해두므로, 전담 전략을 스스로 찾지 못하는 알고리즘은
# 전환이 발생해 상한에 못 미친다.
_DED_N_EQP = 3
_DED_ST = 60
_DED_SIM = 480


def _build_dedicated_assignment() -> dict:
    n, st = _DED_N_EQP, _DED_ST
    cap = _DED_SIM // st
    eqps = [f"EQP{i + 1:03d}" for i in range(n)]
    ppks = [f"PPK{i + 1:03d}" for i in range(n)]
    lot_cds = {ppk: f"LC_{chr(65 + i)}" for i, ppk in enumerate(ppks)}
    oper = "OPER001"

    discrete = []
    for pi, ppk in enumerate(ppks):
        for ci in range(cap):
            home_eqp = eqps[(pi + ci) % n]  # 라운드로빈 홈 배정 — 일부러 섞음
            discrete.append(_discrete_row(
                home_eqp, f"LOT{pi:02d}{ci:02d}", ppk, oper, st, 1,
                carrier_id=f"CAR{pi:02d}{ci:02d}", seq=ci + 1,
            ))

    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper} for ppk in ppks]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": cap, "D1_PLAN_QTY": cap, "PLAN_PRIORITY": 1} for ppk in ppks]
    abstract = [_abstract_row(ppk, oper, "A", st) for ppk in ppks]

    raw = _bundle(discrete, plan, flow, abstract)
    ppk_by_lot = {d["LOT_ID"]: d["PLAN_PROD_ATTR_VAL"] for d in discrete}
    for row in raw["lot_master"]:
        row["LOT_CD"] = lot_cds[ppk_by_lot[row["LOT_ID"]]]
        row["TEMP"] = "T600"
    raw["batch_info"] = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "LOT_CD": lot_cds[ppk], "TEMP": "T600"}
        for ppk in ppks
    ]
    raw["tool_capacity"] = [
        {"LOT_CD": lc, "EQP_MODEL_CD": "A", "MAX_TOOL": 99} for lc in lot_cds.values()
    ]

    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = _DED_SIM
    ed["conversion_minutes"] = st
    return ed


DEDICATED_ASSIGNMENT_CASE = OptimalCase(
    id="dedicated_assignment",
    description=(
        f"EQP {_DED_N_EQP}대 × PPK {_DED_N_EQP}종(각기 다른 LOT_CD), "
        f"ST=Conv={_DED_ST}분, sim={_DED_SIM}분"
    ),
    build=_build_dedicated_assignment,
    optimal=OptimalTarget(
        production=_DED_N_EQP * (_DED_SIM // _DED_ST),
        conversions=0,
        proof=(
            f"EQP당 상한 floor({_DED_SIM}/{_DED_ST})={_DED_SIM // _DED_ST}건 × {_DED_N_EQP}대. "
            "PPK-LOT_CD가 1:1이므로 이 상한은 EQP 1대=PPK 1종 전담 배정일 때만 "
            "전환 0회로 동시에 달성된다."
        ),
    ),
)


# ── 케이스 2: 전환이 강제되는 EQP와 무료인 EQP가 섞인 다중 설비 ───────────────
# EQP 2대(모두 model A, 같은 PPK/OPER 처리 가능), 수요(carrier) 12건이
# EQP 1대의 용량(8건)을 넘어 반드시 두 대를 함께 써야 한다.
#   - EQP001: 초기 셋업 미지정 → 첫 배정은 항상 무료(전환 없음). 전환 없이
#     floor(480/60)=8건 처리 가능.
#   - EQP002: 초기 셋업이 이 제품과 다른 LOT_CD(LC_OTHER)로 이미 채워져 있어,
#     이 제품을 처리하려면 최초 1회 전환이 강제된다. 전환 후 남은 시간으로
#     floor((480-60)/60)=7건 처리 가능.
# 수요 12건은 EQP001 8건(무전환)+EQP002 4건(전환 1회, 용량 7 이내)으로 정확히
# 달성 가능하다. EQP001 단독으로는 8<12로 수요를 못 채우므로 EQP002를 반드시
# 써야 하고, EQP002는 초기 셋업이 다르므로 전환 1회는 회피 불가능한 하한이다
# (더 줄일 수 없고, 이후 같은 제품만 처리하므로 더 늘어날 이유도 없다).
# → 전환을 아예 피하려는 알고리즘은 EQP002를 못 써서 생산 손실을 보고,
#   전환을 아무데서나 남발하는 알고리즘은 conversions>1로 상한을 넘는다.
_MIX_N_EQP = 2
_MIX_ST = 60
_MIX_SIM = 480
_MIX_DEMAND = 12
_MIX_TARGET_LOT_CD = "LC_A"
_MIX_OTHER_LOT_CD = "LC_OTHER"


def _build_mixed_conversion() -> dict:
    ppk, oper = "PPK001", "OPER001"
    eqp_free, eqp_forced = "EQP001", "EQP002"

    discrete = [
        _discrete_row(
            eqp_free if i % 2 == 0 else eqp_forced,  # 홈 배정은 임의(적격성엔 영향 없음)
            f"LOT{i:03d}", ppk, oper, _MIX_ST, 1,
            carrier_id=f"CAR{i:03d}", seq=i + 1,
        )
        for i in range(_MIX_DEMAND)
    ]
    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper}]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": _MIX_DEMAND, "D1_PLAN_QTY": _MIX_DEMAND, "PLAN_PRIORITY": 1}]
    abstract = [_abstract_row(ppk, oper, "A", _MIX_ST)]

    eqp_initial_state = [{
        "EQP_ID": eqp_forced,
        "LOT_CD": _MIX_OTHER_LOT_CD,
        "TEMP": "T900",
        "PLAN_PROD_ATTR_VAL": "PPK000",
        "OPER_ID": oper,
    }]
    raw = _bundle(discrete, plan, flow, abstract, eqp_initial_state=eqp_initial_state)
    for row in raw["lot_master"]:
        row["LOT_CD"] = _MIX_TARGET_LOT_CD
        row["TEMP"] = "T600"
    raw["batch_info"] = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "LOT_CD": _MIX_TARGET_LOT_CD, "TEMP": "T600"},
    ]
    raw["tool_capacity"] = [
        {"LOT_CD": _MIX_TARGET_LOT_CD, "EQP_MODEL_CD": "A", "MAX_TOOL": 99},
        {"LOT_CD": _MIX_OTHER_LOT_CD, "EQP_MODEL_CD": "A", "MAX_TOOL": 99},
    ]

    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = _MIX_SIM
    ed["conversion_minutes"] = _MIX_ST
    return ed


_MIX_FREE_CAP = capacity_bound(_MIX_DEMAND, _MIX_ST, _MIX_SIM)
_MIX_FORCED_CAP = capacity_bound(_MIX_DEMAND, _MIX_ST, _MIX_SIM - _MIX_ST)

MIXED_CONVERSION_CASE = OptimalCase(
    id="mixed_conversion_two_eqp",
    description=(
        f"EQP {_MIX_N_EQP}대(1대는 무전환 가능·1대는 전환 강제), PPK 1종, "
        f"수요 {_MIX_DEMAND}건, ST=Conv={_MIX_ST}분, sim={_MIX_SIM}분"
    ),
    build=_build_mixed_conversion,
    optimal=OptimalTarget(
        production=_MIX_DEMAND,
        conversions=1,
        proof=(
            f"EQP001(초기 셋업 미지정)은 전환 없이 floor({_MIX_SIM}/{_MIX_ST})={_MIX_FREE_CAP}건, "
            f"EQP002(초기 셋업={_MIX_OTHER_LOT_CD})는 전환 1회 후 "
            f"floor(({_MIX_SIM}-{_MIX_ST})/{_MIX_ST})={_MIX_FORCED_CAP}건 처리 가능. "
            f"수요 {_MIX_DEMAND}건은 EQP001 {_MIX_FREE_CAP}건(무전환)+EQP002 "
            f"{_MIX_DEMAND - _MIX_FREE_CAP}건(전환 1회)으로 달성되며, "
            f"EQP001 단독으로는 {_MIX_FREE_CAP}<{_MIX_DEMAND}이라 EQP002 사용이 강제되고 "
            "그 초기 셋업이 다르므로 전환 1회는 회피 불가능한 하한이다."
        ),
    ),
)


CASES: List[OptimalCase] = [
    DEDICATED_ASSIGNMENT_CASE,
    MIXED_CONVERSION_CASE,
]
