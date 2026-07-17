"""
benchmark/optimal/cases.py — 증명 가능한 최적해를 가진 벤치마크 케이스

각 케이스는 "정답이 이 값임을 수학적으로 증명할 수 있는" 만큼 의도적으로
단순화된 시나리오다 (증명은 각 케이스 옆 주석 + benchmark/optimal/proofs.py
참고). data.generator의 저수준 row 빌더로 env_data를 메모리에서 직접
구성하므로 파일 I/O가 없고, agent/env/simulation/data 등 기존 코드는
전혀 수정하지 않는다 — run_inference()가 이미 받는 형태의 env_data만
만들어 넘길 뿐이다.

단일 공정(OPER) 케이스 3개(dedicated_assignment / mixed_conversion_two_eqp /
overflow_conversion_three_eqp)는 각각 "전담 배정" / "전환 강제·무료 혼합" /
"전담+오버플로" 3가지 기본 패턴이다. 다중 공정 케이스들은 이 3가지 패턴을
서로 다른 OPER에 독립적으로 배치해 조합한 것이다 — 공정별로 EQP를 완전히
분리하고 각 공정의 초기 재공을 충분히 채워두므로(파이프라인 재공 이어받기
타이밍은 검증하지 않음), 최적값은 두 공정의 증명된 최적값을 단순히 더한
값이다(생산은 합, 전환도 합 — 공정 간 상호작용이 없으므로 각자 독립적으로
최소/최대이면 전체도 최소/최대).

pipeline_wip_buildup_then_steady 케이스는 위에서 의도적으로 비워둔 자리 —
"파이프라인 재공 이어받기 타이밍"을 검증한다. EQP를 공정별로 분리하지 않고
OPER001·OPER002 겸용으로 공유시키며, OPER001만 초기 재공을 채우고 OPER002는
0에서 출발한다. LOT은 simulation/simulator.py의 flow_next(OPER_SEQ 기반)로
실제 완료 시각에 다음 공정 재공으로 전입하므로(정적 사전 배치가 아님),
최적해는 "초반엔 전량 OPER001에 투입해 재공을 쌓고, 완료되는 대로 EQP를
OPER002로 순차 재배치"하는 시점 배분 자체가 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from data.generator import (
    _abstract_row,
    _discrete_row,
    build_lot_master_from_discrete,
    build_split_rules,
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
    # 기존 케이스는 공정별 초기 재공을 이미 충분히 채워두고 있어(run_inference
    # 기본값 그대로) current-WIP-only 정적 배정으로도 증명된 최적값에 도달할 수
    # 있다. pipeline_wip_buildup_then_steady처럼 재공이 실제로 흘러들어야 하는
    # 케이스만 True로 켠다 — run_inference(enable_wip_inflow=...)로 전달된다.
    enable_wip_inflow: bool = False


def measure(result: dict, sim_end: int) -> CaseMetrics:
    """run_inference() 결과에서 케이스 채점에 필요한 지표만 추출."""
    in_time = [r for r in result["schedule"] if r.get("END_TM", 0) <= sim_end]
    return CaseMetrics(
        production=len(in_time),
        conversions=result["stats"].get("conversions", 0),
    )


_ST = 60
_SIM = 480
_CAP = _SIM // _ST  # 8


# ── 공정(스테이지) 빌더 ──────────────────────────────────────────────────────
# 각 함수는 EQP/PPK/LOT을 prefix로 네임스페이싱해 한 OPER 안의 독립된
# 부분 문제를 만든다. 여러 스테이지를 _combine_stages()로 합치면 다중
# 공정·다중 제품 케이스가 된다(스테이지끼리는 서로 다른 OPER·이름공간이라
# 상호작용이 없다).

@dataclass(frozen=True)
class StagePiece:
    discrete: list
    flow: list
    plan: list
    abstract: list
    eqp_initial_state: list
    lot_cd_overrides: dict  # LOT_ID -> (LOT_CD, TEMP)
    batch_info: list
    tool_capacity: list
    production: int
    conversions: int
    proof: str
    eqp_count: int = field(default=0)


def _stage_dedicated(oper: str, n: int, prefix: str, model: str = "A") -> StagePiece:
    """[oper] EQP n대 × PPK n종 전담 배정 — 증명된 최적: 생산 n*CAP, 전환 0."""
    eqps = [f"{prefix}EQP{i + 1:03d}" for i in range(n)]
    ppks = [f"{prefix}PPK{i + 1:03d}" for i in range(n)]
    lot_cds = {ppk: f"{prefix}LC_{chr(65 + i)}" for i, ppk in enumerate(ppks)}

    discrete = []
    lot_cd_overrides = {}
    for pi, ppk in enumerate(ppks):
        for ci in range(_CAP):
            home_eqp = eqps[(pi + ci) % n]  # 라운드로빈 홈 배정 — 일부러 섞음
            lot_id = f"{prefix}LOT{pi:02d}{ci:02d}"
            discrete.append(_discrete_row(
                home_eqp, lot_id, ppk, oper, _ST, 1, eqp_model=model,
                carrier_id=f"{prefix}CAR{pi:02d}{ci:02d}", seq=ci + 1,
            ))
            lot_cd_overrides[lot_id] = (lot_cds[ppk], "T600")

    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper} for ppk in ppks]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": _CAP, "D1_PLAN_QTY": _CAP, "PLAN_PRIORITY": 1} for ppk in ppks]
    abstract = [_abstract_row(ppk, oper, model, _ST) for ppk in ppks]
    batch_info = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "LOT_CD": lot_cds[ppk], "TEMP": "T600"}
        for ppk in ppks
    ]
    tool_capacity = [{"LOT_CD": lc, "EQP_MODEL_CD": model, "MAX_TOOL": 99} for lc in lot_cds.values()]

    return StagePiece(
        discrete=discrete, flow=flow, plan=plan, abstract=abstract,
        eqp_initial_state=[], lot_cd_overrides=lot_cd_overrides,
        batch_info=batch_info, tool_capacity=tool_capacity,
        production=n * _CAP, conversions=0,
        proof=(
            f"[{oper}] EQP당 상한 floor({_SIM}/{_ST})={_CAP}건 × {n}대. PPK-LOT_CD가 1:1이므로 "
            "이 상한은 EQP 1대=PPK 1종 전담 배정일 때만 전환 0회로 동시에 달성된다."
        ),
        eqp_count=n,
    )


def _stage_mixed(oper: str, prefix: str, model: str = "A") -> StagePiece:
    """[oper] EQP 2대(1대 무전환 가능·1대 전환 강제), PPK 1종, 수요 12건.

    증명된 최적: 생산 12, 전환 1(회피 불가능한 하한).
    """
    demand = 12
    target_lot_cd, other_lot_cd = f"{prefix}LC_A", f"{prefix}LC_OTHER"
    ppk = f"{prefix}PPK001"
    eqp_free, eqp_forced = f"{prefix}EQP001", f"{prefix}EQP002"

    discrete = []
    lot_cd_overrides = {}
    for i in range(demand):
        lot_id = f"{prefix}LOT{i:03d}"
        discrete.append(_discrete_row(
            eqp_free if i % 2 == 0 else eqp_forced,  # 홈 배정은 임의(적격성엔 영향 없음)
            lot_id, ppk, oper, _ST, 1, eqp_model=model,
            carrier_id=f"{prefix}CAR{i:03d}", seq=i + 1,
        ))
        lot_cd_overrides[lot_id] = (target_lot_cd, "T600")

    flow = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper}]
    plan = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
             "D0_PLAN_QTY": demand, "D1_PLAN_QTY": demand, "PLAN_PRIORITY": 1}]
    abstract = [_abstract_row(ppk, oper, model, _ST)]
    eqp_initial_state = [{
        "EQP_ID": eqp_forced, "LOT_CD": other_lot_cd, "TEMP": "T900",
        "PLAN_PROD_ATTR_VAL": f"{prefix}PPK000", "OPER_ID": oper,
    }]
    batch_info = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "LOT_CD": target_lot_cd, "TEMP": "T600"}]
    tool_capacity = [
        {"LOT_CD": target_lot_cd, "EQP_MODEL_CD": model, "MAX_TOOL": 99},
        {"LOT_CD": other_lot_cd, "EQP_MODEL_CD": model, "MAX_TOOL": 99},
    ]

    free_cap = capacity_bound(demand, _ST, _SIM)
    forced_cap = capacity_bound(demand, _ST, _SIM - _ST)
    return StagePiece(
        discrete=discrete, flow=flow, plan=plan, abstract=abstract,
        eqp_initial_state=eqp_initial_state, lot_cd_overrides=lot_cd_overrides,
        batch_info=batch_info, tool_capacity=tool_capacity,
        production=demand, conversions=1,
        proof=(
            f"[{oper}] {eqp_free}(초기 셋업 미지정)은 전환 없이 {free_cap}건, "
            f"{eqp_forced}(초기 셋업={other_lot_cd})는 전환 1회 후 {forced_cap}건 처리 가능. "
            f"수요 {demand}건은 {eqp_free} {free_cap}건(무전환)+{eqp_forced} "
            f"{demand - free_cap}건(전환 1회)으로 달성되며, {eqp_free} 단독으로는 "
            f"{free_cap}<{demand}이라 {eqp_forced} 사용이 강제되고 그 초기 셋업이 다르므로 "
            "전환 1회는 회피 불가능한 하한이다."
        ),
        eqp_count=2,
    )


def _stage_overflow(oper: str, prefix: str, model: str = "A") -> StagePiece:
    """[oper] EQP 3대(2대 전담 + 1대 오버플로 전용), PPK 2종.

    증명된 최적: 생산 20(=12+8), 전환 1(회피 불가능한 하한).
    """
    overflow = 4
    demand_a, demand_b = _CAP + overflow, _CAP
    lot_cd_a, lot_cd_b, lot_cd_other = f"{prefix}LC_A", f"{prefix}LC_B", f"{prefix}LC_OTHER"
    ppk_a, ppk_b = f"{prefix}PPK001", f"{prefix}PPK002"
    eqp_a, eqp_b, eqp_overflow = f"{prefix}EQP001", f"{prefix}EQP002", f"{prefix}EQP003"

    # 홈 EQP는 적격성에 영향을 주지 않지만(WAIT 상태는 자유 배정), 오버플로
    # EQP도 discrete_arrange에 최소 1건 등장해야 eqp_ids에 포함되어 배정 후보가 된다.
    discrete = []
    lot_cd_overrides = {}
    for i in range(demand_a):
        lot_id = f"{prefix}LOTA{i:03d}"
        discrete.append(_discrete_row(
            eqp_a if i < _CAP else eqp_overflow, lot_id, ppk_a, oper, _ST, 1, eqp_model=model,
            carrier_id=f"{prefix}CARA{i:03d}", seq=i + 1,
        ))
        lot_cd_overrides[lot_id] = (lot_cd_a, "T600")
    for i in range(demand_b):
        lot_id = f"{prefix}LOTB{i:03d}"
        discrete.append(_discrete_row(
            eqp_b, lot_id, ppk_b, oper, _ST, 1, eqp_model=model,
            carrier_id=f"{prefix}CARB{i:03d}", seq=i + 1,
        ))
        lot_cd_overrides[lot_id] = (lot_cd_b, "T600")

    flow = [
        {"PLAN_PROD_ATTR_VAL": ppk_a, "OPER_SEQ": 1, "OPER_ID": oper},
        {"PLAN_PROD_ATTR_VAL": ppk_b, "OPER_SEQ": 1, "OPER_ID": oper},
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": ppk_a, "OPER_ID": oper,
         "D0_PLAN_QTY": demand_a, "D1_PLAN_QTY": demand_a, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_ATTR_VAL": ppk_b, "OPER_ID": oper,
         "D0_PLAN_QTY": demand_b, "D1_PLAN_QTY": demand_b, "PLAN_PRIORITY": 1},
    ]
    abstract = [_abstract_row(ppk_a, oper, model, _ST), _abstract_row(ppk_b, oper, model, _ST)]
    eqp_initial_state = [{
        "EQP_ID": eqp_overflow, "LOT_CD": lot_cd_other, "TEMP": "T900",
        "PLAN_PROD_ATTR_VAL": f"{prefix}PPK000", "OPER_ID": oper,
    }]
    batch_info = [
        {"PLAN_PROD_ATTR_VAL": ppk_a, "OPER_ID": oper, "LOT_CD": lot_cd_a, "TEMP": "T600"},
        {"PLAN_PROD_ATTR_VAL": ppk_b, "OPER_ID": oper, "LOT_CD": lot_cd_b, "TEMP": "T600"},
    ]
    tool_capacity = [
        {"LOT_CD": lc, "EQP_MODEL_CD": model, "MAX_TOOL": 99}
        for lc in (lot_cd_a, lot_cd_b, lot_cd_other)
    ]

    return StagePiece(
        discrete=discrete, flow=flow, plan=plan, abstract=abstract,
        eqp_initial_state=eqp_initial_state, lot_cd_overrides=lot_cd_overrides,
        batch_info=batch_info, tool_capacity=tool_capacity,
        production=demand_a + demand_b, conversions=1,
        proof=(
            f"[{oper}] {eqp_a}({ppk_a} 전담)·{eqp_b}({ppk_b} 전담)는 초기 셋업 미지정이라 "
            f"무전환으로 각 {_CAP}건. {ppk_a}의 overflow {overflow}건은 초기 셋업이 다른(={lot_cd_other}) "
            f"{eqp_overflow}에서만 처리 가능하며, 그 첫 배정에 전환이 강제된다. 전환을 0회로 제한하면 "
            f"{eqp_overflow}은 아예 쓸 수 없어 생산 상한이 {_CAP}+{_CAP}={_CAP * 2}건으로 줄어 수요 "
            f"{demand_a + demand_b}건을 채울 수 없으므로, 전환 1회는 회피 불가능한 하한이며 그 이상은 불필요하다."
        ),
        eqp_count=3,
    )


def _combine_stages(*stages: StagePiece) -> dict:
    """서로 다른 OPER의 독립 스테이지들을 하나의 env_data로 병합."""
    discrete = [row for s in stages for row in s.discrete]
    flow = [row for s in stages for row in s.flow]
    plan = [row for s in stages for row in s.plan]
    abstract = [row for s in stages for row in s.abstract]
    eqp_initial_state = [row for s in stages for row in s.eqp_initial_state]

    lot_master = build_lot_master_from_discrete(discrete)
    overrides = {}
    for s in stages:
        overrides.update(s.lot_cd_overrides)
    for row in lot_master:
        lot_cd, temp = overrides[row["LOT_ID"]]
        row["LOT_CD"] = lot_cd
        row["TEMP"] = temp

    batch_info = [row for s in stages for row in s.batch_info]
    tool_capacity = []
    seen = set()
    for s in stages:
        for row in s.tool_capacity:
            key = (row["LOT_CD"], row["EQP_MODEL_CD"])
            if key in seen:
                continue
            seen.add(key)
            tool_capacity.append(row)

    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": build_split_rules(flow),
        "lot_master": lot_master,
        "batch_info": batch_info,
        "tool_capacity": tool_capacity,
        "eqp_initial_state": eqp_initial_state,
    }
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = _SIM
    ed["conversion_minutes"] = _ST
    return ed


def _combined_optimal(*stages: StagePiece) -> OptimalTarget:
    return OptimalTarget(
        production=sum(s.production for s in stages),
        conversions=sum(s.conversions for s in stages),
        proof=" / ".join(s.proof for s in stages) + (
            " 공정(OPER)별로 EQP·재공이 완전히 분리되어 있어 서로 영향을 주지 않으므로, "
            "전체 최적값은 공정별 최적값의 단순 합이다."
        ),
    )


def _single_stage_case(case_id: str, description: str, stage: StagePiece) -> OptimalCase:
    return OptimalCase(
        id=case_id, description=description,
        build=lambda: _combine_stages(stage),
        optimal=OptimalTarget(stage.production, stage.conversions, stage.proof),
    )


def _two_stage_case(case_id: str, description: str, stage1: StagePiece, stage2: StagePiece) -> OptimalCase:
    return OptimalCase(
        id=case_id, description=description,
        build=lambda: _combine_stages(stage1, stage2),
        optimal=_combined_optimal(stage1, stage2),
    )


# ── 단일 공정 케이스 3개 (기존) ───────────────────────────────────────────────
DEDICATED_ASSIGNMENT_CASE = _single_stage_case(
    "dedicated_assignment",
    f"EQP 3대 × PPK 3종(각기 다른 LOT_CD), ST=Conv={_ST}분, sim={_SIM}분",
    _stage_dedicated("OPER001", 3, ""),
)

MIXED_CONVERSION_CASE = _single_stage_case(
    "mixed_conversion_two_eqp",
    f"EQP 2대(1대는 무전환 가능·1대는 전환 강제), PPK 1종, 수요 12건, ST=Conv={_ST}분, sim={_SIM}분",
    _stage_mixed("OPER001", ""),
)

OVERFLOW_CONVERSION_CASE = _single_stage_case(
    "overflow_conversion_three_eqp",
    f"EQP 3대(2대 전담 + 1대 오버플로 전용), PPK 2종, 수요 A=12/B=8건, ST=Conv={_ST}분, sim={_SIM}분",
    _stage_overflow("OPER001", ""),
)


# ── 다중 공정(OPER) × 다중 제품 케이스 7개 ────────────────────────────────────
# 위 3가지 단일 공정 패턴(전담/혼합전환/오버플로)을 서로 다른 OPER(OPER001,
# OPER002)에 독립적으로 배치해 조합했다. 공정별로 EQP를 완전히 분리하고
# 각 공정의 초기 재공을 처음부터 충분히 채워두므로("공정별 독립 EQP" 설계 —
# 파이프라인 재공 이어받기 타이밍 자체는 검증하지 않음), 두 공정은 서로
# 영향을 주지 않아 최적값은 두 공정 최적값의 단순 합으로 증명된다.
#
# 스테이지별로 EQP_MODEL_CD를 다르게 주는 이유: abstract_arrange 적격성은
# (PPK, OPER, EQP_MODEL_CD) 조합으로 결정되므로, OPER만 다르고 모델이 같으면
# "다른 공정용" EQP도 모델이 일치한다는 이유로 서로의 제품 후보로 노출되어
# 스테이지 간 의도치 않은 교차 배정(및 그로 인한 불필요한 전환)이 발생한다.
# 스테이지1=model A, 스테이지2=model B로 분리해 이 누출을 원천 차단한다.
_PATTERNS = {
    "dedicated2": lambda oper, prefix, model: _stage_dedicated(oper, 2, prefix, model),
    "dedicated3": lambda oper, prefix, model: _stage_dedicated(oper, 3, prefix, model),
    "mixed": _stage_mixed,
    "overflow": _stage_overflow,
}

TWO_STAGE_DEDICATED_SMALL_CASE = _two_stage_case(
    "two_stage_dedicated_small",
    "OPER001: EQP2×PPK2 전담 / OPER002: EQP2×PPK2 전담 — 2공정 4제품, 전환 0회가 최적",
    _PATTERNS["dedicated2"]("OPER001", "S1_", "A"),
    _PATTERNS["dedicated2"]("OPER002", "S2_", "B"),
)

TWO_STAGE_DEDICATED_MIXED_CASE = _two_stage_case(
    "two_stage_dedicated_mixed",
    "OPER001: EQP3×PPK3 전담 / OPER002: 전환 강제·무료 혼합(EQP2·PPK1) — 2공정 4제품",
    _PATTERNS["dedicated3"]("OPER001", "S1_", "A"),
    _PATTERNS["mixed"]("OPER002", "S2_", "B"),
)

TWO_STAGE_MIXED_MIXED_CASE = _two_stage_case(
    "two_stage_mixed_mixed",
    "OPER001·OPER002 각각 전환 강제·무료 혼합(EQP2·PPK1) — 2공정 2제품, 공정마다 전환 1회 필수",
    _PATTERNS["mixed"]("OPER001", "S1_", "A"),
    _PATTERNS["mixed"]("OPER002", "S2_", "B"),
)

TWO_STAGE_DEDICATED_OVERFLOW_CASE = _two_stage_case(
    "two_stage_dedicated_overflow",
    "OPER001: EQP2×PPK2 전담 / OPER002: 전담2+오버플로1(EQP3·PPK2) — 2공정 4제품",
    _PATTERNS["dedicated2"]("OPER001", "S1_", "A"),
    _PATTERNS["overflow"]("OPER002", "S2_", "B"),
)

TWO_STAGE_OVERFLOW_OVERFLOW_CASE = _two_stage_case(
    "two_stage_overflow_overflow",
    "OPER001·OPER002 각각 전담2+오버플로1(EQP3·PPK2) — 2공정 4제품, 공정마다 전환 1회 필수",
    _PATTERNS["overflow"]("OPER001", "S1_", "A"),
    _PATTERNS["overflow"]("OPER002", "S2_", "B"),
)

TWO_STAGE_MIXED_OVERFLOW_CASE = _two_stage_case(
    "two_stage_mixed_overflow",
    "OPER001: 전환 강제·무료 혼합(EQP2·PPK1) / OPER002: 전담2+오버플로1(EQP3·PPK2) — 2공정 3제품",
    _PATTERNS["mixed"]("OPER001", "S1_", "A"),
    _PATTERNS["overflow"]("OPER002", "S2_", "B"),
)

TWO_STAGE_DEDICATED_LARGE_CASE = _two_stage_case(
    "two_stage_dedicated_large",
    "OPER001: EQP3×PPK3 전담 / OPER002: EQP2×PPK2 전담 — 2공정 5제품, 전환 0회가 최적",
    _PATTERNS["dedicated3"]("OPER001", "S1_", "A"),
    _PATTERNS["dedicated2"]("OPER002", "S2_", "B"),
)


# ── 파이프라인 재공 이어받기 케이스 (신규) ────────────────────────────────────
# 위 다중 공정 케이스들과 달리 EQP를 공정별로 분리하지 않는다: EQP 4대가
# OPER001·OPER002 겸용(같은 EQP_MODEL_CD)이다. LOT_CD는 전 구간 동일하지만
# TEMP가 공정마다 다르다(OPER001=T600, OPER002=T900) — _would_need_conversion
# (simulation/simulator.py)이 "LOT_CD 또는 TEMP 중 하나라도 바뀌면 전환"으로
# 판정하므로, 같은 LOT_CD라도 공정을 넘나드는 재배치엔 실제 전환 비용이 든다.
# OPER001은 초기 재공 8LOT(수요 전량)을 이미 보유("재고 많음"), OPER002는
# 재공 0("재공 없음")에서 출발 — OPER002 재공은 flow_next를 통해 LOT이
# 실제로 OPER001 처리를 마친 시각에만 생긴다.
def _build_pipeline_buildup_env() -> dict:
    """[OPER001→OPER002] EQP 4대(겸용) × LOT 8개, 1PPK. 후공정(OPER002) ST가
    전공정(OPER001)보다 짧고(40분 < 100분), 공정 전환 시 TEMP가 바뀌어 진짜
    전환 비용(60분)이 든다.

    증명된 최적: 생산 16(=8LOT×2공정), 전환 1, sim=340분(달성 가능한 최소 길이).
    benchmark/optimal 밖에서 exhaustive search로 재검증됨(아래 proof 참고) —
    "EQP 4대 전부가 각 1회씩 전환"하는 대칭적 배분(4전환)은 하한이 아니며,
    비대칭 배분(EQP 1대만 전환)이 동일한 최소 시간에 전환 수를 줄인다.
    """
    n_eqp, n_lots, model = 4, 8, "A"
    st1, st2 = 100, 40  # OPER001(전공정) > OPER002(후공정) — 후공정이 더 짧다
    conv_min = 60  # 공정 전환(TEMP 변경) 1회당 소요 시간
    ppk = "PPK001"
    oper1, oper2 = "OPER001", "OPER002"
    eqps = [f"EQP{i + 1:03d}" for i in range(n_eqp)]

    discrete = []
    for i in range(n_lots):
        lot_id = f"LOT{i:03d}"
        home_eqp = eqps[i % n_eqp]
        # seq를 지정하지 않으면 flow의 OPER_SEQ(=1)로 정확히 해석된다 — 이
        # 값을 잘못 지정하면(예: LOT 인덱스) flow_next 조회가 어긋나 OPER001
        # 완료 후 OPER002로 재공이 전입하지 않는다.
        discrete.append(_discrete_row(
            home_eqp, lot_id, ppk, oper1, st1, 1, eqp_model=model,
            carrier_id=f"CAR{i:03d}",
        ))

    flow = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": oper1},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 2, "OPER_ID": oper2},
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper1,
         "D0_PLAN_QTY": n_lots, "D1_PLAN_QTY": n_lots, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper2,
         "D0_PLAN_QTY": n_lots, "D1_PLAN_QTY": n_lots, "PLAN_PRIORITY": 1},
    ]
    # 두 공정 모두 model="A" 그대로 겹쳐 등록 — abstract_arrange_map은
    # (PPK,OPER,MODEL) 키로만 적격성을 판정하므로, EQP 4대 전부가 OPER001·
    # OPER002 양쪽에 자동으로 배정 후보가 된다(공정별 EQP 분리가 없다는 뜻).
    abstract = [
        _abstract_row(ppk, oper1, model, st1),
        _abstract_row(ppk, oper2, model, st2),
    ]
    lot_master = build_lot_master_from_discrete(discrete)
    lot_cd = lot_master[0]["LOT_CD"]  # 전 LOT 동일 코드 — LOT_CD만으로는 전환 안 생김
    batch_info = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper1, "LOT_CD": lot_cd, "TEMP": "T600"},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper2, "LOT_CD": lot_cd, "TEMP": "T900"},
    ]
    tool_capacity = [{"LOT_CD": lot_cd, "EQP_MODEL_CD": model, "MAX_TOOL": 99}]

    raw = {
        "discrete_arrange": discrete, "abstract_arrange": abstract,
        "plan": plan, "flow": flow, "split": build_split_rules(flow),
        "lot_master": lot_master, "batch_info": batch_info,
        "tool_capacity": tool_capacity, "eqp_initial_state": [],
    }
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 340  # 하한이자 상한(증명 참고, exhaustive search로 확인)
    ed["conversion_minutes"] = conv_min
    return ed


PIPELINE_BUILDUP_THEN_STEADY_CASE = OptimalCase(
    id="pipeline_wip_buildup_then_steady",
    description=(
        "1PPK·EQP4대(OPER001·OPER002 겸용, 동일 LOT_CD·다른 TEMP)·LOT8개. "
        "전공정(OPER001) ST=100분 > 후공정(OPER002) ST=40분, 전환 60분. "
        "OPER001 초기 재공 8LOT(수요 전량, '재고 많음') / OPER002 초기 재공 0. sim=340분."
    ),
    build=_build_pipeline_buildup_env,
    enable_wip_inflow=True,
    optimal=OptimalTarget(
        production=16, conversions=1,
        proof=(
            "달성 가능한 스케줄(하한과 정확히 일치, 전환 1회) — 4대를 비대칭으로 "
            "나눈다: EQP002·EQP003은 OPER001만 각 3LOT씩 [0,300) 동안 100분×3 "
            "연속 처리(전환 없음, 완료 시각 100·200·300). EQP004는 OPER001을 2LOT만 "
            "처리([0,200), 완료 100·200)한 뒤 t=200에 전환(TEMP 600→900, 60분, "
            "[200,260))해 OPER002로 넘어간다. EQP001은 OPER001을 전혀 하지 않고 "
            "(자신의 첫 배정이라 전환 없음) 첫 재공이 생기는 t=100부터 OPER002를 "
            "연속 처리한다. OPER001 완료(=OPER002 재공 전입) 시각은 누적으로 "
            "t=100에 3LOT(EQP002·003·004 각 1개), t=200에 3LOT(같은 세 대 각 1개, "
            "누적 6), t=300에 2LOT(EQP002·003의 마지막 1개씩, 누적 8)이다. EQP001은 "
            "[100,140)부터 40분 간격으로 연속 처리해 t=100/140/180/220의 재공(t=100 "
            "묶음 3개+t=200 묶음 중 2개)을 소진하고, EQP004는 전환이 끝나는 t=260에 "
            "합류해 t=200 묶음의 마지막 1개를 [260,300)에 처리한다. 이어 t=300 묶음 "
            "2개를 EQP001·EQP004가 [300,340)에 하나씩 나눠 처리하며 둘 다 t=340에 "
            "끝난다 — 이 순서는 실제 이산사건 시뮬레이션(그리디: 먼저 준비된 설비가 "
            "먼저 재공을 가져감)으로 검증했으며 두 설비 모두 대기 없이 연속 가동된다. "
            "OPER001 8LOT(=2+3+3)·OPER002 8LOT(=6+2)이 정확히 나뉘어 t=340에 16건 "
            "전부가 끝나고 전환은 EQP004의 1회뿐이다.\n"
            "이보다 짧게는 불가능하다: 전환 0회를 강제하면(모든 EQP가 OPER001 "
            "전용이거나 OPER002 전용) OPER002 담당 설비가 최소 1대 필요한데 그 "
            "설비는 t=100 이전엔 처리할 재공이 없고 8LOT×40분=320분을 혼자 감당해야 "
            "해 t=100+320=420에야 끝난다(0~420 구간에서 이보다 빠른 0전환 배분은 "
            "없음을 전수 확인). 반대로 EQP 4대 모두 대칭으로 OPER001 2LOT→전환→ "
            "OPER002 2LOT씩 맡는(전환 4회) 배분은 직접 시뮬레이션하면 t=280~300 "
            "구간에서 재공 공급이 일시적으로 수요를 못 따라가 t=340으로 늘어나 "
            "위 1전환 배분과 시간은 같으면서 전환만 더 든다. EQP 4대·OPER001 8LOT·"
            "OPER002 8LOT에 대해 가능한 모든 (EQP별 OPER001량, OPER002량) 정수 "
            "배분과 그 실제 재공 전입 타이밍을 전수 탐색(exhaustive search)한 결과 "
            "340분 미만을 달성하는 배분은 없고, 340분을 달성하는 배분 중 전환 "
            "횟수의 최솟값이 1이다 — 따라서 production=16·conversions=1·sim=340이 "
            "동시에 성립하는 최적이다. 요약하면 '전공정을 먼저 몰아 재공을 쌓고, "
            "그 재공이 생기는 대로 EQP 일부(4대 중 1대)를 후공정에 재배치'하되 "
            "재배치는 정확히 한 번만 하는 것이 정답이다."
        ),
    ),
)


CASES: List[OptimalCase] = [
    DEDICATED_ASSIGNMENT_CASE,
    MIXED_CONVERSION_CASE,
    OVERFLOW_CONVERSION_CASE,
    TWO_STAGE_DEDICATED_SMALL_CASE,
    TWO_STAGE_DEDICATED_MIXED_CASE,
    TWO_STAGE_MIXED_MIXED_CASE,
    TWO_STAGE_DEDICATED_OVERFLOW_CASE,
    TWO_STAGE_OVERFLOW_OVERFLOW_CASE,
    TWO_STAGE_MIXED_OVERFLOW_CASE,
    TWO_STAGE_DEDICATED_LARGE_CASE,
    PIPELINE_BUILDUP_THEN_STEADY_CASE,
]
