"""
PLAN_PRIORITY(null 허용, null=최하위) + OVER_PRODUCTION_YN(초과생산 허용 여부) 테스트.
"""
import math

import pytest

from agent.minprogress_agent import MinProgressAgent
from data.loader.fetch import validate_data
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator


def _raw():
    """PPK_A(우선순위1,Y) / PPK_B(우선순위5,N) / PPK_C(우선순위 null,Y) — 모두 OPER001, EQP001(모델 A)."""
    ppks = ["PPK_A", "PPK_B", "PPK_C"]
    discrete_arrange = [
        {
            "EQP_ID": "EQP001", "LOT_ID": f"LOT_{ppk}",
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
            "ST": 10, "EQP_MODEL_CD": "A", "WF_QTY": 25,
        }
        for ppk in ppks
    ]
    abstract_arrange = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 10}
        for ppk in ppks
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": "PPK_A", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1, "OVER_PRODUCTION_YN": "Y"},
        {"PLAN_PROD_ATTR_VAL": "PPK_B", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 5, "OVER_PRODUCTION_YN": "N"},
        {"PLAN_PROD_ATTR_VAL": "PPK_C", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 100, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": None},
    ]
    flow = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"}
        for ppk in ppks
    ]
    return {
        "discrete_arrange": discrete_arrange,
        "abstract_arrange": abstract_arrange,
        "plan": plan,
        "flow": flow,
    }


def _sim() -> SchedulingSimulator:
    raw = _raw()
    assert validate_data(raw) == []
    env_data = preprocess(raw)
    return SchedulingSimulator(env_data, record_history=False)


# ── PLAN_PRIORITY: null 허용 + null-last ────────────────────────────────────

def test_validate_data_allows_null_plan_priority():
    assert validate_data(_raw()) == []


def test_preprocess_keeps_null_priority_distinct_from_default():
    env_data = preprocess(_raw())
    plan_meta = env_data["plan_meta"]
    assert plan_meta[("PPK_A", "OPER001")]["priority"] == 1
    assert plan_meta[("PPK_B", "OPER001")]["priority"] == 5
    assert plan_meta[("PPK_C", "OPER001")]["priority"] is None


def test_simulator_plan_priority_null_is_none():
    sim = _sim()
    assert sim._plan_priority("PPK_A", "OPER001") == 1
    assert sim._plan_priority("PPK_B", "OPER001") == 5
    assert sim._plan_priority("PPK_C", "OPER001") is None


def test_minprogress_agent_sorts_null_priority_last():
    env_data = preprocess(_raw())
    agent = MinProgressAgent(env_data)
    a = agent._plan_priority("PPK_A", "OPER001")
    b = agent._plan_priority("PPK_B", "OPER001")
    c = agent._plan_priority("PPK_C", "OPER001")
    assert c == math.inf
    assert a < b < c
    assert min([a, b, c]) == a  # 작은 값(=1)이 우선


# ── 신규 w_priority 리워드 ──────────────────────────────────────────────────

def test_priority_reward_favors_smaller_priority_value():
    sim = _sim()
    sim._reward_cfg.w_priority = 2.0
    r_a = sim._priority_reward("PPK_A", "OPER001")  # priority=1
    r_b = sim._priority_reward("PPK_B", "OPER001")  # priority=5
    r_c = sim._priority_reward("PPK_C", "OPER001")  # priority=null
    assert r_a == pytest.approx(2.0)
    assert r_b == pytest.approx(0.4)
    assert r_c == 0.0
    assert r_a > r_b > r_c


def test_priority_reward_disabled_when_weight_zero():
    sim = _sim()
    sim._reward_cfg.w_priority = 0.0
    assert sim._priority_reward("PPK_A", "OPER001") == 0.0


# ── 초과생산 허용 여부(OVER_PRODUCTION_YN) ──────────────────────────────────

def test_achievable_qty_extends_past_quota_only_when_overproduction_allowed():
    sim = _sim()
    sim.stats["completed_qty"][("PPK_A", "OPER001")] = 100  # 계획(Y) 달성
    sim.stats["completed_qty"][("PPK_B", "OPER001")] = 100  # 계획(N) 달성
    sim._wip_wafers = lambda _p, _o: 25  # type: ignore[method-assign]

    # Y: 계획 상한을 넘어 done+reachable까지 목표 확장(takt 기준 계속 생산)
    assert sim._achievable_qty("PPK_A", "OPER001") == 125
    # N: 계획 상한에서 멈춤(더 생산해도 추가 목표 없음)
    assert sim._achievable_qty("PPK_B", "OPER001") == 100


def test_overproduction_blocked_only_while_y_flagged_wip_pending():
    sim = _sim()
    sim.stats["completed_qty"][("PPK_B", "OPER001")] = 100  # N, 계획 달성(초과생산 구간)

    # PPK_A(Y) 또는 PPK_C(Y)에 ready WIP이 남아있는 한 PPK_B(N) 배정은 지연
    sim._ready_wip_qty = lambda p, o: 25  # type: ignore[method-assign]
    assert sim._overproduction_blocked("PPK_B", "OPER001") is True

    # Y-플래그 재공이 모두 소진되면 그제서야 N의 초과생산 허용
    sim._ready_wip_qty = lambda p, o: (0 if p in ("PPK_A", "PPK_C") else 25)  # type: ignore[method-assign]
    assert sim._overproduction_blocked("PPK_B", "OPER001") is False

    # Y-플래그 버킷은 계획 달성 여부와 무관하게 절대 차단되지 않음
    assert sim._overproduction_blocked("PPK_A", "OPER001") is False


def test_feasible_ppk_oper_excludes_over_quota_n_bucket_while_y_pending():
    sim = _sim()
    eqp_id = "EQP001"
    sim.stats["completed_qty"][("PPK_B", "OPER001")] = 100  # N, 계획 달성
    sim._invalidate_caches()  # get_feasible_ppk_oper는 state_version 캐시 사용

    feasible = [sim.ppk_oper_from_flat(f) for f in sim.get_feasible_ppk_oper(eqp_id)]
    assert ("PPK_A", "OPER001") in feasible
    assert ("PPK_C", "OPER001") in feasible
    assert ("PPK_B", "OPER001") not in feasible  # Y 재공이 남아 있어 지연

    # Y-플래그 재공이 모두 소진되면 PPK_B도 feasible로 복귀
    sim._wip_pool[("PPK_A", "OPER001")]["wip_qty"] = 0
    sim._wip_pool[("PPK_C", "OPER001")]["wip_qty"] = 0
    sim._invalidate_caches()
    feasible_after = [sim.ppk_oper_from_flat(f) for f in sim.get_feasible_ppk_oper(eqp_id)]
    assert ("PPK_B", "OPER001") in feasible_after
