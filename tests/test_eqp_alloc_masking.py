"""
tests/test_eqp_alloc_masking.py

preprocess()가 채우는 env_data["eqp_alloc_map"](allocation.eqp_allocator 결과)이
시뮬레이터의 action mask(get_feasible_ppk_oper)를 실제로 좁히는지, 그리고
CONFIG.env.eqp_alloc_enabled 옵션이 하위 호환 폴백으로 동작하는지 검증한다.
"""
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260810070000"


@pytest.fixture(autouse=True)
def _restore_eqp_alloc_enabled():
    original = CONFIG.env.eqp_alloc_enabled
    yield
    CONFIG.env.eqp_alloc_enabled = original


def _raw_two_models_one_bucket():
    """(PPK001,OPER001)에 M1/M2 둘 다 가능. EQP001=M1, EQP002=M2."""
    discrete = [
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 30,
         "EQP_MODEL_CD": "M1", "WF_QTY": 25},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT002", "CARRIER_ID": "CAR002",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 30,
         "EQP_MODEL_CD": "M2", "WF_QTY": 25},
    ]
    return {
        "discrete_arrange": discrete,
        "plan": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
        ],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M1", "ST": 30},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M2", "ST": 30},
        ],
    }


def test_alloc_map_restricts_model_not_allocated():
    CONFIG.env.eqp_alloc_enabled = False  # 직접 eqp_alloc_map을 주입해 마스킹 로직만 검증
    data = preprocess(_raw_two_models_one_bucket(), period_key=RULE_TIMEKEY)
    data["eqp_alloc_map"] = {("PPK001", "OPER001"): {"M1": 1}}

    sim = SchedulingSimulator(data)
    bucket = ("PPK001", "OPER001")
    assert bucket in sim._eqp_feasible_bucket_keys("EQP001")  # M1 → 할당됨
    assert bucket not in sim._eqp_feasible_bucket_keys("EQP002")  # M2 → 할당 안 됨


def test_alloc_map_missing_key_is_unrestricted_fail_open():
    CONFIG.env.eqp_alloc_enabled = False
    data = preprocess(_raw_two_models_one_bucket(), period_key=RULE_TIMEKEY)
    data["eqp_alloc_map"] = {}  # 이번 회차 할당 계산 대상 아님(예: 계획 누락)

    sim = SchedulingSimulator(data)
    bucket = ("PPK001", "OPER001")
    assert bucket in sim._eqp_feasible_bucket_keys("EQP001")
    assert bucket in sim._eqp_feasible_bucket_keys("EQP002")


def test_eqp_alloc_enabled_populates_map_end_to_end():
    CONFIG.env.eqp_alloc_enabled = True
    data = preprocess(_raw_two_models_one_bucket(), period_key=RULE_TIMEKEY)
    assert data["eqp_alloc_rows"]
    assert ("PPK001", "OPER001") in data["eqp_alloc_map"]


def test_eqp_alloc_disabled_leaves_map_empty():
    CONFIG.env.eqp_alloc_enabled = False
    data = preprocess(_raw_two_models_one_bucket(), period_key=RULE_TIMEKEY)
    assert data["eqp_alloc_rows"] == []
    assert data["eqp_alloc_map"] == {}

    # 비활성화 시 기존(모델 매칭만) 동작 그대로 — 두 모델 EQP 모두 feasible.
    sim = SchedulingSimulator(data)
    bucket = ("PPK001", "OPER001")
    assert bucket in sim._eqp_feasible_bucket_keys("EQP001")
    assert bucket in sim._eqp_feasible_bucket_keys("EQP002")
