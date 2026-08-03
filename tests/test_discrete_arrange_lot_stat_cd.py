"""
tests/test_discrete_arrange_lot_stat_cd.py

discrete_arrange의 LOT_STAT_CD(PROC/LOAD/SELE/RESV) 강제배정 — eqp_queue_init
파이프라인을 별도로 못 넣는 소스를 위한 대체 입력 경로. discrete_arrange 행에
LOT_STAT_CD가 채워지면 그 carrier는 AI가 결정하지 않고 같은 EQP_ID에
FORCED_LOT_STAT_ORDER(PROC→LOAD→RESV→SELE) 순으로 강제 배정되며, 내부적으로
eqp_queue_init과 동일한 eqp_fixed_queue로 합류한다.
"""
import pytest

from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"  # 07:00:00 = sim_base_time, minute 0


def _discrete_row(eqp_id, lot_id, carrier_id, lot_stat_cd=None, st=45, wf_qty=25,
                   ppk="PPK001", oper_id="OPER001"):
    row = {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id, "ST": st,
        "EQP_MODEL_CD": "A", "WF_QTY": wf_qty,
    }
    if lot_stat_cd is not None:
        row["LOT_STAT_CD"] = lot_stat_cd
    return row


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


def test_forced_rows_priority_order_proc_load_resv_sele():
    # 입력 순서를 우선순위와 다르게 섞어도 PROC -> LOAD -> RESV -> SELE 순으로
    # 재정렬돼야 한다.
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_SELE", "CAR_SELE", lot_stat_cd="SELE"),
        _discrete_row("EQP001", "LOT_PROC", "CAR_PROC", lot_stat_cd="PROC"),
        _discrete_row("EQP001", "LOT_RESV", "CAR_RESV", lot_stat_cd="RESV"),
        _discrete_row("EQP001", "LOT_LOAD", "CAR_LOAD", lot_stat_cd="LOAD"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    queue = data["eqp_fixed_queue"]["EQP001"]
    assert [row["lot_id"] for row in queue] == ["CAR_PROC", "CAR_LOAD", "CAR_RESV", "CAR_SELE"]


def test_forced_rows_same_priority_keep_input_order():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_B", "CAR_B", lot_stat_cd="PROC"),
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="PROC"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    queue = data["eqp_fixed_queue"]["EQP001"]
    assert [row["lot_id"] for row in queue] == ["CAR_B", "CAR_A"]


def test_forced_row_start_end_min_computed_from_st_times_wf_qty():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="PROC", st=2, wf_qty=25),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    queue = data["eqp_fixed_queue"]["EQP001"]
    assert queue[0]["start_min"] == 0
    assert queue[0]["end_min"] == 50  # 2분/장 * 25장


def test_forced_row_activates_immediately_in_simulator():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="PROC", st=2, wf_qty=25),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    assert len(sim.schedule) == 1
    row = sim.schedule[0]
    assert row["CARRIER_ID"] == "CAR_A"
    assert row["START_TM"] == 0
    assert row["END_TM"] == 50
    assert row["LOT_STAT_CD"] == "FIXED"


def test_forced_rows_excluded_from_free_scheduling_pool():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="PROC"),
        _discrete_row("EQP001", "LOT_B", "CAR_B"),  # LOT_STAT_CD 없음 -> WAIT
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    assert all(lot["lot_id"] != "CAR_A" for lot in data["lots"])
    assert any(lot["lot_id"] == "CAR_B" for lot in data["lots"])
    assert "CAR_A" not in data["eqp_lot_map"].get("EQP001", [])
    assert "CAR_B" in data["eqp_lot_map"].get("EQP001", [])

    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)
    assert "CAR_A" not in sim.lot_pool
    assert "CAR_A" not in sim.available_lots("EQP001")


def test_forced_rows_continue_after_existing_eqp_queue_init():
    # 같은 EQP에 eqp_queue_init 큐가 이미 있으면 그 마지막 END_TM부터 이어서
    # 강제배정돼야 한다(이중 예약 방지).
    raw = _base_raw(
        discrete_arrange=[
            _discrete_row("EQP001", "LOT_FORCED", "CAR_FORCED", lot_stat_cd="PROC", st=2, wf_qty=25),
        ],
        eqp_queue_init=[
            _queue_row("EQP001", 1, "LOT_A", "CAR_A", "20260712070000", "20260712074500"),
        ],
    )
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    queue = data["eqp_fixed_queue"]["EQP001"]
    assert [row["lot_id"] for row in queue] == ["CAR_A", "CAR_FORCED"]
    assert queue[0]["end_min"] == 45
    assert queue[1]["start_min"] == 45
    assert queue[1]["end_min"] == 95


def test_missing_or_wait_lot_stat_cd_stays_free_scheduling_candidate():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="WAIT"),
        _discrete_row("EQP001", "LOT_B", "CAR_B"),  # 필드 자체가 없음
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    assert data.get("eqp_fixed_queue", {}).get("EQP001", []) == []
    assert {"CAR_A", "CAR_B"} <= set(data["eqp_lot_map"].get("EQP001", []))


def test_unknown_lot_stat_cd_raises():
    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="BOGUS"),
    ])
    with pytest.raises(ValueError):
        preprocess(raw, period_key=RULE_TIMEKEY)


def test_forced_row_not_split_even_when_split_rule_matches():
    raw = _base_raw(
        discrete_arrange=[
            _discrete_row("EQP001", "LOT_A", "CAR_A", lot_stat_cd="PROC", wf_qty=30),
        ],
        split=[{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                "EQP_MODEL_CD": "A", "SPLIT_QTY": 10}],
    )
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    queue = data["eqp_fixed_queue"]["EQP001"]
    assert len(queue) == 1
    assert queue[0]["wf_qty"] == 30
