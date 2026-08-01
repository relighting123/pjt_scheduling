"""
tests/test_forced_assignment_bypasses_constraints.py

eqp_queue_init(이미 확정된 재공)은 물리적으로 이미 그 상태로 진행 중이므로,
자유 스케줄링에만 적용되어야 할 제약(전환 횟수 상한, abstract_arrange 누락)에
막히지 않고 무조건 배정되어야 한다. discrete_arrange에만 존재하던 이 항목들이
이제는 별도 입력(eqp_queue_init)으로 분리되어, discrete_arrange 자유배정
후보가 전혀 없는 EQP도 eqp_queue_init만으로 등록되고 즉시 반영돼야 한다.
"""
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


def _queue_init_row(eqp_id, lot_id, carrier_id, wf_qty=25,
                     start_tm="20260712070000", end_tm="20260712074500"):
    return {
        "EQP_ID": eqp_id, "SEQ_NO": 1, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "OPER_ID": "OPER001", "START_TM": start_tm, "END_TM": end_tm,
        "PLAN_PROD_ATTR_VAL": "PPK001", "WF_QTY": wf_qty,
    }


def _discrete_row(eqp_id, lot_id, carrier_id, st=45):
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": st,
        "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1,
    }


def _base_raw(discrete=None, abstract_arrange=None):
    return {
        "discrete_arrange": discrete or [],
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 1, "D1_PLAN_QTY": 1, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": abstract_arrange or [],
    }


def _conv_needed_eqp_initial_state(eqp_id):
    return [{"EQP_ID": eqp_id, "LOT_CD": "LC999", "TEMP": "T999",
              "PLAN_PROD_ATTR_VAL": "PPK999", "OPER_ID": "OPER001"}]


def test_fixed_queue_ignores_missing_abstract_arrange_row():
    # abstract_arrange에 (PPK001, OPER001, A) 조합이 전혀 없고 discrete_arrange도
    # 없음(EQP001은 eqp_queue_init만으로 등록됨) — WAIT였다면 배정 불가능한 상황.
    raw = _base_raw(abstract_arrange=[])
    raw["eqp_queue_init"] = [_queue_init_row("EQP001", "LOT001", "CAR001")]
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert len(sim.schedule) == 1
    assert sim.schedule[0]["LOT_ID"] == "LOT001"
    assert sim.schedule[0]["LOT_STAT_CD"] == "FIXED"


def test_fixed_queue_ignores_conversion_limit_zero():
    # EQP001은 다른 셋업이라 전환이 필요한 상황이지만, max_conversions=0/
    # max_conversions_per_eqp=0이어도 이미 확정된 큐는 막히지 않아야 한다.
    raw = _base_raw(abstract_arrange=[
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
    ])
    raw["eqp_queue_init"] = [_queue_init_row("EQP001", "LOT001", "CAR001")]
    raw["eqp_initial_state"] = _conv_needed_eqp_initial_state("EQP001")
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["max_conversions"] = 0
    data["max_conversions_per_eqp"] = 0

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert len(sim.schedule) == 1
    assert sim.schedule[0]["LOT_ID"] == "LOT001"


def test_wait_lot_still_blocked_by_conversion_limit_zero():
    # 회귀 방지: WAIT LOT은 여전히 전환 상한에 막혀야 한다(이미 확정된 큐만 예외).
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001")]
    raw = _base_raw(discrete, abstract_arrange=[
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
    ])
    raw["eqp_initial_state"] = _conv_needed_eqp_initial_state("EQP001")
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["max_conversions"] = 0
    data["max_conversions_per_eqp"] = 0

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert sim.assign_ppk_oper("EQP001", "PPK001", "OPER001") == -1.0
    assert len(sim.schedule) == 0
