"""
tests/test_forced_assignment_bypasses_constraints.py

강제 배정(PROC/LOAD/RESV/SELE) LOT은 물리적으로 이미 그 상태로 진행 중이므로,
자유 스케줄링에만 적용되어야 할 제약(전환 횟수 상한, abstract_arrange 누락)에
막히지 않고 무조건 배정되어야 한다.
"""
from config import CONFIG
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


def _discrete_row(eqp_id, lot_id, carrier_id, lot_stat_cd, st=45):
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": st,
        "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": lot_stat_cd,
    }


def _base_raw(discrete, abstract_arrange=None):
    return {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 1, "D1_PLAN_QTY": 1, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": abstract_arrange or [],
    }


def _conv_needed_eqp_initial_state(eqp_id):
    return [{"EQP_ID": eqp_id, "LOT_CD": "LC999", "TEMP": "T999",
              "PLAN_PROD_ATTR_VAL": "PPK999", "OPER_ID": "OPER001"}]


def test_forced_load_ignores_missing_abstract_arrange_row():
    # abstract_arrange에 (PPK001, OPER001, A) 조합이 전혀 없음 — WAIT였다면 배정 불가.
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001", "LOAD")]
    raw = _base_raw(discrete, abstract_arrange=[])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert len(sim.schedule) == 1
    assert sim.schedule[0]["LOT_ID"] == "LOT001"
    assert sim.schedule[0]["LOT_STAT_CD"] == "LOAD"


def test_forced_load_ignores_conversion_limit_zero():
    # EQP001은 다른 셋업이라 전환이 필요한 상황이지만, max_conversions=0/
    # max_conversions_per_eqp=0이어도 강제 배정은 막히지 않아야 한다.
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001", "LOAD")]
    raw = _base_raw(discrete, abstract_arrange=[
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
    ])
    raw["eqp_initial_state"] = _conv_needed_eqp_initial_state("EQP001")
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["max_conversions"] = 0
    data["max_conversions_per_eqp"] = 0

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert len(sim.schedule) == 1
    assert sim.schedule[0]["LOT_ID"] == "LOT001"


def test_wait_lot_still_blocked_by_conversion_limit_zero():
    # 회귀 방지: WAIT LOT은 여전히 전환 상한에 막혀야 한다(강제 배정만 예외).
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001", "WAIT")]
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


def test_staged_forced_queue_skips_permanently_failing_head(monkeypatch):
    # 큐 맨 앞 항목(들)이 영원히 실패해도, 뒤에 대기 중인 다른 강제 carrier는
    # 여전히 시도되고 배정돼야 한다 (선입 실패가 뒤를 막아선 안 됨).
    discrete = [_discrete_row("EQP001", "LOT001", "CAR001", "LOAD")]
    raw = _base_raw(discrete, abstract_arrange=[
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    sim.eqps["EQP001"].status = "idle"
    sim._eqp_staged_forced["EQP001"] = ["PHANTOM_A", "PHANTOM_B", "REAL_LOT"]

    calls = []

    def fake_assign_lot(eqp_id, lot_id):
        calls.append(lot_id)
        return -1.0 if lot_id.startswith("PHANTOM") else 5.0

    monkeypatch.setattr(sim, "assign_lot", fake_assign_lot)

    ok = sim._try_assign_staged_forced("EQP001")
    assert ok is True
    assert calls == ["PHANTOM_A", "PHANTOM_B", "REAL_LOT"]
