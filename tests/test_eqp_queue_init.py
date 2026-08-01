"""
tests/test_eqp_queue_init.py

eqp_queue_init: AI 스케줄러가 배정을 결정하지 않는, 이미 투입이 확정된 EQP별
가공 큐(초기 상태) 입력. discrete_arrange의 옛 LOT_STAT_CD(PROC/LOAD/RESV/SELE)
강제배정을 대체한다.

스키마: EQP_ID/SEQ_NO/LOT_ID/CARRIER_ID/OPER_ID/START_TM/END_TM/
PLAN_PROD_ATTR_VAL/WF_QTY. ST 없이 START_TM/END_TM을 그대로 스케줄에 반영한다.
"""
from config import CONFIG
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"  # 07:00:00 = sim_base_time, minute 0


def _queue_row(eqp_id, seq_no, lot_id, carrier_id, start_tm, end_tm, wf_qty=25,
               ppk="PPK001", oper_id="OPER001"):
    return {
        "EQP_ID": eqp_id, "SEQ_NO": seq_no, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "OPER_ID": oper_id, "START_TM": start_tm, "END_TM": end_tm,
        "PLAN_PROD_ATTR_VAL": ppk, "WF_QTY": wf_qty,
    }


def _base_raw(**overrides):
    raw = {
        "discrete_arrange": [],
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
        ],
    }
    raw.update(overrides)
    return raw


def test_normalize_groups_by_eqp_and_sorts_by_seq_no():
    # 입력 순서를 일부러 SEQ_NO 역순으로 섞어, 정규화 단계가 EQP별 그룹핑 +
    # SEQ_NO 오름차순 정렬까지 끝내는지 확인한다(시뮬레이터가 재정렬하지 않음).
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP001", 2, "LOT_B", "CAR_B", "20260712080000", "20260712083000"),
        _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    queue = data["eqp_fixed_queue"]["EQP001"]
    assert [row["lot_id"] for row in queue] == ["CAR_A", "CAR_B"]
    assert [row["seq_no"] for row in queue] == [1, 2]
    assert queue[0]["start_min"] == 0
    assert queue[0]["end_min"] == 45
    assert queue[1]["start_min"] == 60
    assert queue[1]["end_min"] == 90


def test_eqp_registered_from_queue_init_alone():
    # discrete_arrange에 전혀 없는 EQP도 eqp_queue_init만으로 등록돼야 한다
    # (이미 확정 큐로 꽉 차 자유배정 후보가 하나도 없는 장비의 실제 상황).
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP_ONLY", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    assert "EQP_ONLY" in data["eqp_ids"]


def test_start_tm_at_or_before_now_activates_with_given_times_not_recomputed():
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    assert len(sim.schedule) == 1
    row = sim.schedule[0]
    assert row["CARRIER_ID"] == "CAR_A"
    assert row["START_TM"] == 0
    assert row["END_TM"] == 45
    assert row["LOT_STAT_CD"] == "FIXED"


def test_future_start_tm_keeps_input_start_time_not_clamped_to_zero():
    # start_min=60(미래) — 즉시 t=0으로 당겨지지 않고 입력된 START_TM 그대로
    # 활성화돼야 한다(무제약 큐이므로 결국 자동으로 소비되지만 시각은 보존).
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712080000", "20260712083000"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    assert len(sim.schedule) == 1
    row = sim.schedule[0]
    assert row["START_TM"] == 60
    assert row["END_TM"] == 90


def test_sequential_consumption_in_seq_no_order():
    # SEQ_NO 1이 끝나면 곧바로(다음 idle 시점에) SEQ_NO 2가 이어서 활성화돼야 한다.
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
        _queue_row("EQP001", 2, "LOT_B", "CAR_B", "20260712074500", "20260712081500"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    assert [row["CARRIER_ID"] for row in sim.schedule] == ["CAR_A", "CAR_B"]
    assert sim.schedule[1]["START_TM"] == 45
    assert sim.schedule[1]["END_TM"] == 75


def test_fixed_queue_lot_never_appears_in_free_lot_pool():
    raw = _base_raw(eqp_queue_init=[
        _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    assert all(lot["lot_id"] != "CAR_A" for lot in data["lots"])

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert "CAR_A" not in sim.lot_pool
    assert sim.available_lots("EQP001") == []


def test_fixed_queue_lot_not_split_even_when_split_rule_matches():
    # lot_info/_apply_wafer_lot_split에 애초에 들어가지 않으므로 SPLIT_QTY
    # 규칙과 무관하게 WF_QTY가 그대로 유지돼야 한다.
    raw = _base_raw(
        eqp_queue_init=[
            _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500", wf_qty=30),
        ],
        split=[{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                "EQP_MODEL_CD": "A", "SPLIT_QTY": 10}],
    )
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    queue = data["eqp_fixed_queue"]["EQP001"]
    assert len(queue) == 1
    assert queue[0]["wf_qty"] == 30


def test_discrete_wait_enabled_false_does_not_affect_fixed_queue():
    original = CONFIG.env.discrete_wait_enabled
    CONFIG.env.discrete_wait_enabled = False
    try:
        raw = _base_raw(eqp_queue_init=[
            _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
        ])
        data = preprocess(raw, period_key=RULE_TIMEKEY)
        sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
        assert len(sim.schedule) == 1
        assert sim.schedule[0]["CARRIER_ID"] == "CAR_A"
    finally:
        CONFIG.env.discrete_wait_enabled = original
