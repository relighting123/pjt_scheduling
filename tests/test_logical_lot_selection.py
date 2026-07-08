"""논리 LOT(LOT_ID) 단위 carrier 우선 선택 — 진행 중 LOT의 남은 carrier 우선."""
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

ST, WF = 5, 25


def _disc(eqp, lot, carrier=None, lot_stat_cd=None):
    row = {
        "EQP_ID": eqp,
        "LOT_ID": lot,
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "OPER_ID": "OPER001",
        "ST": ST,
        "EQP_MODEL_CD": "A",
        "WF_QTY": WF,
        "CARRIER_ID": carrier or lot,
    }
    if lot_stat_cd is not None:
        row["LOT_STAT_CD"] = lot_stat_cd
    return row


def _raw(discrete):
    return {
        "discrete_arrange": discrete,
        "plan": [{
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1,
        }],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
    }


def test_preprocess_multiple_carriers_per_logical_lot():
    discrete = [
        _disc("EQP001", "LOT001", "CAR001"),
        _disc("EQP001", "LOT001", "CAR002"),
        _disc("EQP002", "LOT002", "CAR003"),
    ]
    env_data = preprocess(_raw(discrete))
    lots = {l["lot_id"]: l for l in env_data["lots"]}
    assert set(lots) == {"CAR001", "CAR002", "CAR003"}
    assert lots["CAR001"]["logical_lot_id"] == "LOT001"
    assert lots["CAR002"]["logical_lot_id"] == "LOT001"
    assert lots["CAR003"]["logical_lot_id"] == "LOT002"
    assert env_data["eqp_forced_queue"] == {}


def _bare_sim():
    sim = SchedulingSimulator.__new__(SchedulingSimulator)
    sim.current_time = 0
    sim.eqps = {"EQP001": type("E", (), {"prev_lot_cd": "", "prev_temp": ""})()}
    sim._conversion_minutes = 0
    sim._in_flight = {}
    sim._eqp_forced_queue = {}
    sim._eqp_pending_assign = {}
    sim.lot_pool = {}
    sim._wip_lot_meta = {}
    return sim


def test_auto_select_lot_prefers_sibling_of_in_flight_logical_lot():
    sim = _bare_sim()
    sim._in_flight = {
        "CAR001": {
            "PLAN_PROD_ATTR_VAL": "PPK001",
            "oper_id": "OPER001",
            "logical_lot_id": "LOT001",
        },
    }

    peer = dict(
        PLAN_PROD_ATTR_VAL="PPK001",
        oper_id="OPER001",
        oper_in_time=0,
        processing_time=30,
        is_abstract=False,
        is_initial_wip=True,
        logical_lot_id="LOT001",
    )
    other = dict(
        PLAN_PROD_ATTR_VAL="PPK001",
        oper_id="OPER001",
        oper_in_time=0,
        processing_time=10,
        is_abstract=False,
        is_initial_wip=True,
        logical_lot_id="LOT999",
    )
    candidates = [
        {**other, "lot_id": "CAR999"},
        {**peer, "lot_id": "CAR002"},
    ]
    assert sim._auto_select_lot("EQP001", candidates) == "CAR002"


def test_auto_select_lot_prefers_sibling_in_forced_queue():
    sim = _bare_sim()
    sim._eqp_forced_queue = {"EQP001": ["CAR001"]}
    sim._wip_lot_meta = {
        "CAR001": {
            "PLAN_PROD_ATTR_VAL": "PPK001",
            "oper_id": "OPER001",
            "logical_lot_id": "LOT001",
        },
    }

    candidates = [
        {
            "lot_id": "CAR999",
            "logical_lot_id": "LOT999",
            "PLAN_PROD_ATTR_VAL": "PPK001",
            "oper_id": "OPER001",
            "oper_in_time": 0,
            "processing_time": 10,
            "is_abstract": False,
            "is_initial_wip": True,
        },
        {
            "lot_id": "CAR002",
            "logical_lot_id": "LOT001",
            "PLAN_PROD_ATTR_VAL": "PPK001",
            "oper_id": "OPER001",
            "oper_in_time": 0,
            "processing_time": 30,
            "is_abstract": False,
            "is_initial_wip": True,
        },
    ]
    assert sim._auto_select_lot("EQP001", candidates) == "CAR002"
