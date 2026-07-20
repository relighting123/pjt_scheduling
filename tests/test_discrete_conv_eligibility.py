"""
tests/test_discrete_conv_eligibility.py

WAIT LOT의 EQP 배정 자격은 "전환(conversion) 필요 여부"로 갈린다.

  - 전환 불필요(EQP에 이미 장착된 LOT_CD/TEMP와 목표가 같음): 이 특정
    carrier가 이 EQP에서 discrete(EQP×carrier 정밀 조합, proc_time_matrix)로
    입력 가능한 조합인지를 봐야 한다. 같은 (PPK,OPER)라도 carrier별로 실제
    가능한 EQP가 다를 수 있어(discrete_arrange가 그 근거) abstract(모델
    매칭)만으로는 불충분 — discrete 조합이 없으면 배정 불가.
  - 전환 필요(LOT_CD/TEMP 중 하나라도 다름): 기존 셋업 기준 discrete 조합은
    전환 후엔 더 이상 유효하지 않으므로 discrete를 참조하지 않고 abstract
    (모델 평균 ST)만으로 판단한다.

강제 배정 LOT(PROC/LOAD/RESV/SELE)과 CONFIG.env.discrete_wait_enabled=False는
이 제약과 무관하게 기존 동작(abstract 폴백)을 그대로 유지한다. 또한 EQP가
아직 한 번도 셋업된 적이 없는 "첫 배정"(prev_lot_cd=None)은 비교 대상이
없으므로 이 제약에서 제외된다(부트스트랩 케이스).
"""
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"

# PPK001 기본 LOT_CD/TEMP (data/loader/preprocess.py::_default_lot_cd/_default_temp 규칙)
PPK001_LOT_CD = "LC001"
PPK001_TEMP = "T700"


@pytest.fixture(autouse=True)
def _restore_discrete_wait_enabled():
    original = CONFIG.env.discrete_wait_enabled
    yield
    CONFIG.env.discrete_wait_enabled = original


def _discrete_row(eqp_id, lot_id, carrier_id, wf_qty=25, st=60, model="A", lot_stat_cd="WAIT"):
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": st,
        "EQP_MODEL_CD": model, "WF_QTY": wf_qty, "SEQ": 1,
        "LOT_STAT_CD": lot_stat_cd,
    }


def _base_raw(discrete):
    return {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 75},
        ],
    }


def _no_conv_eqp_initial_state(eqp_id):
    """대상 PPK001 기본 LOT_CD/TEMP와 동일하게 셋업 → 전환 불필요 상태."""
    return [{"EQP_ID": eqp_id, "LOT_CD": PPK001_LOT_CD, "TEMP": PPK001_TEMP,
              "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001"}]


def _conv_needed_eqp_initial_state(eqp_id):
    """PPK001과 다른 LOT_CD/TEMP로 셋업 → 전환 필요 상태."""
    return [{"EQP_ID": eqp_id, "LOT_CD": "LC999", "TEMP": "T999",
              "PLAN_PROD_ATTR_VAL": "PPK999", "OPER_ID": "OPER001"}]


def _extra_registration_row(eqp_id, tag):
    """discrete_arrange가 EQP 전체 목록의 유일한 출처라, PPK001/OPER001에는
    discrete 조합이 전혀 없는 채로 이 EQP를 등록만 시키기 위한 더미 행
    (다른 PPK/OPER 조합이라 실제 테스트 대상 버킷과 무관)."""
    return {
        "EQP_ID": eqp_id, "LOT_ID": f"LOT_{tag}", "CARRIER_ID": f"CAR_{tag}",
        "PLAN_PROD_ATTR_VAL": "PPK002", "OPER_ID": "OPER002", "ST": 50,
        "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": "WAIT",
    }


def test_no_conversion_picks_discrete_carrier_over_non_discrete():
    # CAR001: EQP001에 discrete 조합 있음. CAR002: EQP002에만 discrete 조합
    # (EQP001 기준으로는 abstract 매칭만 가능).
    discrete = [
        _discrete_row("EQP001", "LOT001", "CAR001", st=60),
        _discrete_row("EQP002", "LOT002", "CAR002", st=90),
    ]
    raw = _base_raw(discrete)
    raw["eqp_initial_state"] = _no_conv_eqp_initial_state("EQP001")
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    lots = sim.available_lots("EQP001")
    assert {l["lot_id"] for l in lots} == {"CAR001", "CAR002"}

    reward = sim.assign_ppk_oper("EQP001", "PPK001", "OPER001")
    assert reward != -1.0
    row = sim.schedule[0]
    assert row["CARRIER_ID"] == "CAR001"  # 전환 불필요 → discrete 조합 있는 carrier만 자격
    assert row["ABSTRACT"] is False
    assert row["ST"] == 60


def test_no_conversion_masks_bucket_when_no_discrete_carrier_exists():
    # CAR002만 존재 (EQP001엔 discrete 조합 없음) → EQP001은 전환도 필요 없는데
    # discrete 조합도 없으니 이 버킷 자체가 EQP001에서 마스킹돼야 한다.
    discrete = [
        _discrete_row("EQP002", "LOT002", "CAR002", st=90),
        _extra_registration_row("EQP001", "reg1"),
    ]
    raw = _base_raw(discrete)
    raw["eqp_initial_state"] = _no_conv_eqp_initial_state("EQP001") + _no_conv_eqp_initial_state("EQP002")
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)

    feasible = sim.get_feasible_ppk_oper("EQP001")
    flat_idx = sim.ppk_oper_flat_index("OPER001", "PPK001")
    assert flat_idx not in feasible, "discrete 조합 없는 carrier뿐이면 버킷이 마스킹돼야 한다"

    assert sim.assign_ppk_oper("EQP001", "PPK001", "OPER001") == -1.0

    # 반대로 CAR002 자신의 discrete EQP(EQP002)에서는 정상 배정 가능해야 한다.
    feasible2 = sim.get_feasible_ppk_oper("EQP002")
    assert flat_idx in feasible2
    reward = sim.assign_ppk_oper("EQP002", "PPK001", "OPER001")
    assert reward != -1.0


def test_conversion_needed_uses_abstract_even_if_discrete_exists():
    # CAR001은 EQP001에 discrete 조합(ST=60)이 있지만, EQP001이 다른
    # LOT_CD/TEMP로 셋업돼 있어 전환이 필요 → discrete 무시하고 abstract(ST=75) 사용.
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001", st=60)]
    raw = _base_raw(discrete)
    raw["eqp_initial_state"] = _conv_needed_eqp_initial_state("EQP001")
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    reward = sim.assign_ppk_oper("EQP001", "PPK001", "OPER001")
    assert reward != -1.0
    if not sim.schedule:
        # 전환이 필요해 즉시 커밋되지 않고 conv_end까지 진행 중 — 완료 시점까지 진행.
        sim._advance_to_next_decision()
    row = sim.schedule[0]
    assert row["CARRIER_ID"] == "CAR001"
    assert row["ABSTRACT"] is True
    assert row["ST"] == 75  # discrete ST(60)이 아니라 abstract 평균(75)


def test_discrete_wait_enabled_false_bypasses_new_gate():
    # discrete_wait_enabled=False면 전환 불필요 상태에서도 discrete 조합 없이
    # abstract만으로 배정 가능해야 한다(기존 옵션 동작 유지).
    CONFIG.env.discrete_wait_enabled = False
    discrete = [
        _discrete_row("EQP002", "LOT002", "CAR002", st=90),
        _extra_registration_row("EQP001", "reg3"),
    ]
    raw = _base_raw(discrete)
    raw["eqp_initial_state"] = _no_conv_eqp_initial_state("EQP001") + _no_conv_eqp_initial_state("EQP002")
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    feasible = sim.get_feasible_ppk_oper("EQP001")
    flat_idx = sim.ppk_oper_flat_index("OPER001", "PPK001")
    assert flat_idx in feasible

    reward = sim.assign_ppk_oper("EQP001", "PPK001", "OPER001")
    assert reward != -1.0
    row = sim.schedule[0]
    assert row["ABSTRACT"] is True
    assert row["ST"] == 75


def test_forced_lot_ignores_new_gate():
    # PROC(강제 배정) LOT은 discrete 조합이 없고 전환도 필요 없는 상태라도
    # 항상 배정된다(자유 스케줄링 대상이 아니므로).
    discrete = [_discrete_row("EQP002", "LOT002", "CAR002", st=90, lot_stat_cd="PROC")]
    raw = _base_raw(discrete)
    raw["eqp_initial_state"] = _no_conv_eqp_initial_state("EQP002")
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    # 강제 배정이라 reset() 중 t=0에 즉시 커밋돼야 한다 — discrete 조합 없음/
    # 전환 불필요 여부와 무관하게 새 게이트에 막히지 않아야 한다.
    assert len(sim.schedule) == 1
    assert sim.schedule[0]["CARRIER_ID"] == "CAR002"


def test_bootstrap_first_assignment_not_blocked_by_new_gate():
    # EQP가 아직 한 번도 셋업된 적 없으면(prev_lot_cd=None) discrete 조합이
    # 없어도 abstract로 첫 배정이 가능해야 한다(전환 비교 대상이 없는
    # 부트스트랩 케이스).
    discrete = [
        _discrete_row("EQP002", "LOT002", "CAR002", st=90),
        _extra_registration_row("EQP001", "reg2"),
    ]
    raw = _base_raw(discrete)  # eqp_initial_state 없음 → EQP001.prev_lot_cd is None
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    feasible = sim.get_feasible_ppk_oper("EQP001")
    flat_idx = sim.ppk_oper_flat_index("OPER001", "PPK001")
    assert flat_idx in feasible

    reward = sim.assign_ppk_oper("EQP001", "PPK001", "OPER001")
    assert reward != -1.0
