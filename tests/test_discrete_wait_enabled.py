"""
tests/test_discrete_wait_enabled.py

CONFIG.env.discrete_wait_enabled(기본 True)이 False면 LOT_STAT_CD=WAIT인
discrete_arrange 행의 discrete 정보(특정 EQP 고정, 실측 ST)를 배정 로직에서
쓰지 않고 abstract 매칭 경로(모델 평균 ST, 모델이 맞는 아무 장비)만 태운다.
LOT의 수량/제품/공정 정체성(WIP 카운트)은 그대로 유지되고, PROC/LOAD/RESV/SELE
(강제 배정) LOT은 이 옵션과 무관하게 항상 discrete 그대로 유지된다.

이 과정에서 발견된 별도 버그도 함께 고정한다: _rebuild_eqp_oper_cap()이
r["LOT_ID"](논리 LOT_ID)로 lot_info(내부 carrier 키로 색인)를 조회해서,
CARRIER_ID가 LOT_ID와 다른 정상적인 1:N 케이스에서 eqp_oper_cap이 항상
비어 있었다.
"""
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess

RULE_TIMEKEY = "20260712070000"


@pytest.fixture(autouse=True)
def _restore_discrete_wait_enabled():
    original = CONFIG.env.discrete_wait_enabled
    yield
    CONFIG.env.discrete_wait_enabled = original


def _base_inputs():
    discrete = [
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 60,
         "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": "WAIT"},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT002", "CARRIER_ID": "CAR002",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 90,
         "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": "PROC"},
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
    ]
    flow = [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}]
    abstract_arrange = [
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 75},
    ]
    return {
        "discrete_arrange": discrete, "plan": plan, "flow": flow,
        "abstract_arrange": abstract_arrange,
    }


def test_default_discrete_wait_enabled_is_true():
    assert CONFIG.env.discrete_wait_enabled is True


def test_eqp_oper_cap_includes_carrier_distinct_wait_lot_by_default():
    # 회귀 테스트: _rebuild_eqp_oper_cap()이 CARRIER_ID != LOT_ID인 경우에도
    # 정상적으로 eqp_oper_cap을 채워야 한다 (이전엔 항상 비어 있었음).
    data = preprocess(_base_inputs(), period_key=RULE_TIMEKEY)
    assert data["eqp_oper_cap"] == {
        "EQP001": ["OPER001"],
        "EQP002": ["OPER001"],
    }
    assert data["proc_time_matrix"][("CAR001", "EQP001", "OPER001")] == 60


def test_discrete_wait_disabled_strips_discrete_info_for_wait_lot_only():
    CONFIG.env.discrete_wait_enabled = False
    data = preprocess(_base_inputs(), period_key=RULE_TIMEKEY)

    # WAIT(CAR001): discrete 정보(EQP 고정, ST, eqp_oper_cap 기여)가 전부 빠져야 함.
    assert ("CAR001", "EQP001", "OPER001") not in data["proc_time_matrix"]
    assert "CAR001" not in data["eqp_lot_map"].get("EQP001", [])
    assert data["eqp_oper_cap"] == {"EQP002": ["OPER001"]}

    # PROC(CAR002, 강제 배정): 옵션과 무관하게 discrete 정보 그대로 유지.
    assert data["proc_time_matrix"][("CAR002", "EQP002", "OPER001")] == 90
    assert "CAR002" in data["eqp_lot_map"]["EQP002"]


def test_discrete_wait_disabled_preserves_lot_identity_and_wip_count():
    CONFIG.env.discrete_wait_enabled = False
    data = preprocess(_base_inputs(), period_key=RULE_TIMEKEY)

    lot_ids = {lot["lot_id"]: lot for lot in data["lots"]}
    assert set(lot_ids) == {"CAR001", "CAR002"}
    assert lot_ids["CAR001"]["wf_qty"] == 25
    assert lot_ids["CAR001"]["PLAN_PROD_ATTR_VAL"] == "PPK001"
    assert lot_ids["CAR001"]["oper_id"] == "OPER001"

    wip = data["abstract_wip_init"][("PPK001", "OPER001")]
    assert wip["wip_qty"] == 2
    assert set(wip["lot_ids"]) == {"CAR001", "CAR002"}


def test_discrete_wait_disabled_schedules_wait_lot_via_abstract_average_st():
    from simulation.simulator import SchedulingSimulator

    CONFIG.env.discrete_wait_enabled = False
    raw = _base_inputs()
    # PROC(CAR002)이 자기 EQP를 곧바로 점유하니, WAIT(CAR001)만 남기고 단순화한다.
    raw["discrete_arrange"] = [raw["discrete_arrange"][0]]
    data = preprocess(raw, period_key=RULE_TIMEKEY)

    sim = SchedulingSimulator(data)
    assert not sim.is_done()
    eqp_id = sim.current_idle_eqp()
    lots = sim.available_lots(eqp_id)
    assert lots, "WAIT LOT은 abstract 매칭으로 여전히 배정 가능해야 한다"
    sim.assign_lot(eqp_id, lots[0]["lot_id"])

    row = sim.schedule[0]
    assert row["ABSTRACT"] is True
    assert row["ST"] == 75  # discrete ST(60)이 아니라 abstract 평균(75) 사용
