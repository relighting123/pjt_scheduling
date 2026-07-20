"""
tests/test_wip_inflow_discrete_eligibility.py

enable_wip_inflow=True + discrete_wait_enabled=True 조합에서, 다음 공정으로
흘러들어온(유입) carrier는 그 공정 조합으로 discrete(실측) 데이터가 존재한
적이 없다. discrete 자격 요건을 그대로 적용하면, 이미 그 공정에 필요한
셋업으로 세팅된 EQP조차 후보에서 배제되고 셋업이 다른 EQP가 불필요한 전환을
해가며 대신 가져가는 결과가 된다 — 유입 재공은 이 요건에서 제외해야 한다.
"""
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


@pytest.fixture(autouse=True)
def _restore_discrete_wait_enabled():
    original = CONFIG.env.discrete_wait_enabled
    yield
    CONFIG.env.discrete_wait_enabled = original


def _build_two_stage_env_data():
    """OPER001(LC_A/T600) -> OPER002(LC_B/T700) 2공정 흐름, carrier 1건.

    EQP002는 처음부터 OPER002가 요구하는 셋업(LC_B/T700)으로 세팅돼 있어
    전환 없이 바로 받을 수 있어야 한다. EQP002를 eqp_ids에 등록하기 위한
    무관한 더미 discrete 행도 함께 넣는다(EQP002 자체는 이 조합에 실측 데이터가
    없다는 게 테스트의 핵심).
    """
    discrete = [
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 30,
         "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": "WAIT"},
        {"EQP_ID": "EQP002", "LOT_ID": "LOTDUMMY", "CARRIER_ID": "CARDUMMY",
         "PLAN_PROD_ATTR_VAL": "PPKZZZ", "OPER_ID": "OPERZZZ", "ST": 30,
         "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": 1, "LOT_STAT_CD": "WAIT"},
    ]
    raw = {
        "discrete_arrange": discrete,
        "plan": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "D0_PLAN_QTY": 1, "D1_PLAN_QTY": 1, "PLAN_PRIORITY": 1},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002",
             "D0_PLAN_QTY": 1, "D1_PLAN_QTY": 1, "PLAN_PRIORITY": 1},
        ],
        "flow": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 2, "OPER_ID": "OPER002"},
        ],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 30},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002", "EQP_MODEL_CD": "A", "ST": 30},
        ],
        "batch_info": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "LOT_CD": "LC_A", "TEMP": "T600"},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002", "LOT_CD": "LC_B", "TEMP": "T700"},
        ],
        "eqp_initial_state": [
            {"EQP_ID": "EQP002", "LOT_CD": "LC_B", "TEMP": "T700",
             "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002"},
        ],
    }
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["enable_wip_inflow"] = True
    return data


def test_inflow_carrier_eligible_on_already_matching_eqp_without_discrete():
    data = _build_two_stage_env_data()
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    sim.assign_lot("EQP001", "CAR001")  # OPER001 완료까지 배정
    sim._advance_to_next_decision()  # t=750: OPER001 완료, OPER002로 유입

    assert sim.current_time == 750
    lots = sim.available_lots("EQP002")
    assert len(lots) == 1
    inflowed = lots[0]
    assert inflowed["oper_id"] == "OPER002"
    assert inflowed["is_abstract"] is True  # 유입 재공은 discrete 조합이 없음

    # 셋업이 이미 딱 맞는(전환 불필요) EQP002가 배제되면 안 된다.
    assert sim._lot_conv_discrete_eligible("EQP002", inflowed) is True
    assert ("PPK001", "OPER002") in sim._eqp_feasible_bucket_keys("EQP002")


def test_inflow_carrier_prefers_no_conversion_eqp_over_conversion_eqp():
    data = _build_two_stage_env_data()
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    sim.assign_lot("EQP001", "CAR001")
    sim._advance_to_next_decision()

    assert sorted(sim.get_idle_eqps()) == ["EQP001", "EQP002"]
    pick = sim.pick_earliest_st_assignment()
    assert pick is not None
    eqp_id, lot_id, _ = pick
    # EQP002(전환 불필요)가 EQP001(전환 필요)보다 우선 선택돼야 한다.
    assert eqp_id == "EQP002"
    assert lot_id == "CAR001"

    reward = sim.assign_lot(eqp_id, lot_id)
    assert reward != -1.0
    assert sim.stats["conversions"] == 0
