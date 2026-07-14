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

from benchmark.optimal.proofs import capacity_bound, spt_max_completed


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


def _bundle(discrete: list, plan: list, flow: list, abstract_arrange: list) -> dict:
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
    }


# ── 케이스 1: 단일 EQP 용량 상한 ────────────────────────────────────────────
# EQP 1대, LOT_CD 1종(=전환 불가능), carrier 6개(ST=10분), sim=45분.
# 상한 = floor(45/10) = 4건, 처리 순서와 무관하게 항상 달성 가능.
_CAP_ST = 10
_CAP_SIM = 45
_CAP_N = 6


def _build_capacity_bound() -> dict:
    ppk, oper, eqp = "PPK001", "OPER001", "EQP001"
    discrete = [
        _discrete_row(eqp, f"LOT{i:03d}", ppk, oper, _CAP_ST, 1,
                       carrier_id=f"CAR{i:03d}", seq=i + 1)
        for i in range(_CAP_N)
    ]
    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper}]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": _CAP_N, "D1_PLAN_QTY": _CAP_N, "PLAN_PRIORITY": 1}]
    raw = _bundle(discrete, plan, flow, [_abstract_row(ppk, oper, "A", _CAP_ST)])
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = _CAP_SIM
    ed["conversion_minutes"] = 30
    return ed


CAPACITY_BOUND_CASE = OptimalCase(
    id="capacity_bound_single_eqp",
    description=f"EQP 1대, carrier {_CAP_N}개(ST={_CAP_ST}분), sim={_CAP_SIM}분 — 용량 상한",
    build=_build_capacity_bound,
    optimal=OptimalTarget(
        production=capacity_bound(_CAP_N, _CAP_ST, _CAP_SIM),
        conversions=0,
        proof=f"floor({_CAP_SIM}/{_CAP_ST})={_CAP_SIM // _CAP_ST}건이 EQP 1대의 물리적 시간 상한. "
              "LOT_CD가 단일이므로 전환은 애초에 발생 불가.",
    ),
)


# ── 케이스 2: 전담 배정 (일반화된 CONV_BENCH) ────────────────────────────────
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


# ── 케이스 3: 처리 순서가 성과를 가르는 단일 EQP (SPT) ───────────────────────
# EQP 1대, LOT_CD 1종(전환 없음), carrier별 ST가 서로 다르고 내림차순으로
# 배치되어 있다. 마감 sim 이내 최대 완료 건수는 ST가 작은 순으로 채울 때
# 최대화된다 (교환 논증, benchmark/optimal/proofs.spt_max_completed 참고).
# 입력 순서를 그대로 따르는(=ST 큰 것부터 처리) 알고리즘은 손해를 본다.
_SPT_STS = [45, 35, 25, 15, 5]
_SPT_SIM = 50


def _build_spt_ordering() -> dict:
    ppk, oper, eqp = "PPK001", "OPER001", "EQP001"
    discrete = [
        _discrete_row(eqp, f"LOT{i:03d}", ppk, oper, st, 1,
                       carrier_id=f"CAR{i:03d}", seq=i + 1)
        for i, st in enumerate(_SPT_STS)
    ]
    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper}]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": len(_SPT_STS), "D1_PLAN_QTY": len(_SPT_STS), "PLAN_PRIORITY": 1}]
    abstract = [_abstract_row(ppk, oper, "A", max(_SPT_STS))]
    raw = _bundle(discrete, plan, flow, abstract)
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = _SPT_SIM
    ed["conversion_minutes"] = 30
    return ed


SPT_ORDERING_CASE = OptimalCase(
    id="spt_ordering_single_eqp",
    description=f"EQP 1대, ST={_SPT_STS}분(내림차순 배치), sim={_SPT_SIM}분 — 처리 순서가 성과를 좌우",
    build=_build_spt_ordering,
    optimal=OptimalTarget(
        production=spt_max_completed(_SPT_STS, _SPT_SIM),
        conversions=0,
        proof=(
            f"ST 오름차순 누적합이 {_SPT_SIM}분을 넘기 직전까지 "
            f"{spt_max_completed(_SPT_STS, _SPT_SIM)}건 완료 가능 (교환 논증 SPT 최적, "
            "tests/test_optimal_bench.py에서 전수 탐색으로 교차 검증)."
        ),
    ),
)


CASES: List[OptimalCase] = [
    CAPACITY_BOUND_CASE,
    DEDICATED_ASSIGNMENT_CASE,
    SPT_ORDERING_CASE,
]
