"""
tests/test_eqp_allocator.py

allocation/eqp_allocator.py – PPK/OPER × EQP_MODEL_CD 사전 할당(장비 댓수) 산출
알고리즘 단위 테스트. raw 입력 포맷은 data/loader/preprocess.preprocess()와
동일(SQL 원본 컬럼명)하다.
"""
import pytest

from allocation.eqp_allocator import build_eqp_alloc_map, compute_eqp_allocation
from config import CONFIG

RULE_TIMEKEY = "20260810070000"
WINDOW = 600  # 테스트 편의를 위한 짧은 창(분) — capa = 600/ST


@pytest.fixture(autouse=True)
def _restore_conversion_config():
    orig_max = CONFIG.env.max_conversions
    orig_per_eqp = CONFIG.env.max_conversions_per_eqp
    yield
    CONFIG.env.max_conversions = orig_max
    CONFIG.env.max_conversions_per_eqp = orig_per_eqp


def _discrete_row(eqp_id, lot_id, ppk, oper_id, model, wf_qty=10, carrier_id=None):
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id or lot_id,
        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id, "EQP_MODEL_CD": model,
        "WF_QTY": wf_qty, "ST": 60,
    }


def _plan_row(ppk, oper_id, d0, priority=1, d1=0):
    return {
        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id,
        "D0_PLAN_QTY": d0, "D1_PLAN_QTY": d1, "PLAN_PRIORITY": priority,
    }


def _flow_row(ppk, seq, oper_id):
    return {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": seq, "OPER_ID": oper_id}


def _abstract_row(ppk, oper_id, model, st):
    return {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id, "EQP_MODEL_CD": model, "ST": st}


def test_target_qty_is_min_of_plan_and_wip():
    # capa/unit = 600/60 = 10장. WIP=100장, 계획=45장 → target=45 → 5대 필요(ceil(45/10)).
    raw = {
        "abstract_arrange": [_abstract_row("PPK001", "OPER001", "M1", 60)],
        "discrete_arrange": [
            _discrete_row(f"EQP{i:03d}", f"LOT{i:03d}", "PPK001", "OPER001", "M1", wf_qty=10)
            for i in range(1, 11)  # 10장 x 10 lot = 100장 WIP, EQP 10대 보유
        ],
        "plan": [_plan_row("PPK001", "OPER001", d0=45)],
        "flow": [_flow_row("PPK001", 1, "OPER001")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["WIP_QTY"] == 100
    assert row["PLAN_QTY"] == 45
    assert row["TARGET_QTY"] == 45
    assert row["ALLOC_EQP_CNT"] == 5
    assert row["IS_EXCLUSIVE_MODEL"] == "Y"
    assert row["TOTAL_EQP_CNT"] == 10


def test_target_qty_capped_by_wip_when_wip_scarce():
    raw = {
        "abstract_arrange": [_abstract_row("PPK001", "OPER001", "M1", 60)],
        "discrete_arrange": [_discrete_row("EQP001", "LOT001", "PPK001", "OPER001", "M1", wf_qty=8)],
        "plan": [_plan_row("PPK001", "OPER001", d0=500)],
        "flow": [_flow_row("PPK001", 1, "OPER001")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    assert rows[0]["TARGET_QTY"] == 8  # WIP(8) < 계획(500)


def test_no_allocation_row_when_no_plan():
    raw = {
        "abstract_arrange": [_abstract_row("PPK001", "OPER001", "M1", 60)],
        "discrete_arrange": [_discrete_row("EQP001", "LOT001", "PPK001", "OPER001", "M1", wf_qty=8)],
        "plan": [],  # 계획 없음 → target=0 → 행 없음(마스킹 fail-open)
        "flow": [_flow_row("PPK001", 1, "OPER001")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    assert rows == []
    assert build_eqp_alloc_map(rows) == {}


def test_priority_order_higher_priority_ppk_claims_scarce_model_first():
    # M1 총 2대뿐. PPK001(priority=1)이 4대 필요, PPK002(priority=2)도 4대 필요.
    # 둘 다 전용 모델(M1)만 가능 → 우선순위 낮은 값(1)이 먼저 2대를 모두 가져가고
    # PPK002는 0대.
    raw = {
        "abstract_arrange": [
            _abstract_row("PPK001", "OPER001", "M1", 60),
            _abstract_row("PPK002", "OPER001", "M1", 60),
        ],
        "discrete_arrange": [
            _discrete_row("EQP001", "LOT1", "PPK001", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP002", "LOT1", "PPK001", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP001", "LOT2", "PPK002", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP002", "LOT2", "PPK002", "OPER001", "M1", wf_qty=100),
        ],
        "plan": [
            _plan_row("PPK001", "OPER001", d0=1000, priority=1),
            _plan_row("PPK002", "OPER001", d0=1000, priority=2),
        ],
        "flow": [_flow_row("PPK001", 1, "OPER001"), _flow_row("PPK002", 1, "OPER001")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    by_ppk = {r["PLAN_PROD_ATTR_VAL"]: r for r in rows}
    assert by_ppk["PPK001"]["ALLOC_EQP_CNT"] == 2
    assert "PPK002" not in by_ppk  # 남은 대수가 없어 배정 자체가 없음


def test_exclusive_model_claims_before_shared_demand():
    # M1: 총 2대. PPK001/OPER001은 M1만 가능(전용, priority=2, 나중 순위).
    # PPK002/OPER001은 M1·M2 둘 다 가능(공용, priority=1, 먼저 순위)하지만 M2로도
    # 충분히 커버 가능 → 전용 수요(PPK001)가 M1을 먼저 확보해야 한다.
    raw = {
        "abstract_arrange": [
            _abstract_row("PPK001", "OPER001", "M1", 60),
            _abstract_row("PPK002", "OPER001", "M1", 60),
            _abstract_row("PPK002", "OPER001", "M2", 60),
        ],
        "discrete_arrange": [
            _discrete_row("EQP001", "L1", "PPK001", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP002", "L1", "PPK001", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP001", "L2", "PPK002", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP002", "L2", "PPK002", "OPER001", "M1", wf_qty=100),
            _discrete_row("EQP003", "L3", "PPK002", "OPER001", "M2", wf_qty=100),
        ],
        "plan": [
            _plan_row("PPK001", "OPER001", d0=1000, priority=2),
            _plan_row("PPK002", "OPER001", d0=1000, priority=1),
        ],
        "flow": [_flow_row("PPK001", 1, "OPER001"), _flow_row("PPK002", 1, "OPER001")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    m1_by_ppk = {r["PLAN_PROD_ATTR_VAL"]: r for r in rows if r["EQP_MODEL_CD"] == "M1"}
    assert m1_by_ppk["PPK001"]["ALLOC_EQP_CNT"] == 2  # 전용 수요가 M1 전량 확보
    assert "PPK002" not in m1_by_ppk  # PPK002는 M1을 못 받음
    m2_by_ppk = {r["PLAN_PROD_ATTR_VAL"]: r for r in rows if r["EQP_MODEL_CD"] == "M2"}
    assert m2_by_ppk["PPK002"]["ALLOC_EQP_CNT"] == 1  # 대신 M2로 커버


def test_shared_model_balanced_across_opers_of_same_ppk():
    # 같은 PPK, OPER001/OPER002 둘 다 공용모델(M1) 사용 가능, WIP·계획 동일 →
    # 두 OPER에 균등 배분되어야 한다(한쪽에 몰리면 안 됨).
    raw = {
        "abstract_arrange": [
            _abstract_row("PPK001", "OPER001", "M1", 60),
            _abstract_row("PPK001", "OPER001", "M2", 60),
            _abstract_row("PPK001", "OPER002", "M1", 60),
            _abstract_row("PPK001", "OPER002", "M2", 60),
        ],
        "discrete_arrange": [
            _discrete_row(f"EQP{i:03d}", f"L{i}", "PPK001",
                          "OPER001" if i <= 5 else "OPER002",
                          "M1" if i % 2 == 0 else "M2", wf_qty=50)
            for i in range(1, 9)
        ] + [
            # 각 EQP가 두 OPER 모두 자격이 있다고 가정할 필요는 없음. WIP만 채움.
        ],
        "plan": [
            _plan_row("PPK001", "OPER001", d0=1000, priority=1),
            _plan_row("PPK001", "OPER002", d0=1000, priority=1),
        ],
        "flow": [_flow_row("PPK001", 1, "OPER001"), _flow_row("PPK001", 2, "OPER002")],
    }
    # 재공(WIP)을 두 OPER에 동일하게 크게 채워 target이 대칭이 되도록 별도 구성
    raw["discrete_arrange"] = [
        _discrete_row("EQP001", "L1", "PPK001", "OPER001", "M1", wf_qty=1000),
        _discrete_row("EQP002", "L2", "PPK001", "OPER001", "M2", wf_qty=1000),
        _discrete_row("EQP003", "L3", "PPK001", "OPER002", "M1", wf_qty=1000),
        _discrete_row("EQP004", "L4", "PPK001", "OPER002", "M2", wf_qty=1000),
    ]
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    total_by_oper = {}
    for r in rows:
        total_by_oper.setdefault(r["OPER_ID"], 0)
        total_by_oper[r["OPER_ID"]] += r["ALLOC_EQP_CNT"]
    assert total_by_oper["OPER001"] == total_by_oper["OPER002"]


def test_wip_skew_no_longer_triggers_special_release():
    # 과거에는 WIP가 압도적으로 몰린 OPER에 균등 상한 없는 예외(1.5배 release)가
    # 적용됐지만, 그 로직을 제거한 뒤로는 몰림 여부와 무관하게 항상 라운드로빈
    # 균등 배분만 적용되어야 한다.
    raw = {
        "abstract_arrange": [
            _abstract_row("PPK001", "OPER001", "M1", 60),
            _abstract_row("PPK001", "OPER001", "M2", 60),
            _abstract_row("PPK001", "OPER002", "M1", 60),
            _abstract_row("PPK001", "OPER002", "M2", 60),
        ],
        "discrete_arrange": [
            _discrete_row("EQP001", "L1", "PPK001", "OPER001", "M1", wf_qty=900),
            _discrete_row("EQP002", "L2", "PPK001", "OPER001", "M2", wf_qty=900),
            _discrete_row("EQP003", "L3", "PPK001", "OPER002", "M1", wf_qty=50),
            _discrete_row("EQP004", "L4", "PPK001", "OPER002", "M2", wf_qty=50),
        ],
        "plan": [
            _plan_row("PPK001", "OPER001", d0=1000, priority=1),
            _plan_row("PPK001", "OPER002", d0=1000, priority=1),
        ],
        "flow": [_flow_row("PPK001", 1, "OPER001"), _flow_row("PPK001", 2, "OPER002")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    total_by_oper = {}
    for r in rows:
        total_by_oper.setdefault(r["OPER_ID"], 0)
        total_by_oper[r["OPER_ID"]] += r["ALLOC_EQP_CNT"]
    assert total_by_oper.get("OPER001", 0) == total_by_oper.get("OPER002", 0) == 2


def test_build_eqp_alloc_map_filters_zero_and_groups_by_ppk_oper():
    rows = [
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M1", "ALLOC_EQP_CNT": 3},
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M2", "ALLOC_EQP_CNT": 0},
    ]
    m = build_eqp_alloc_map(rows)
    assert m == {("PPK001", "OPER001"): {"M1": 3}}


# ── 도달 가능 재공(유입) 반영 ─────────────────────────────────────────────────

def test_downstream_oper_target_includes_upstream_reachable_wip():
    # OPER002는 discrete_arrange에 재공이 전혀 없지만(현재 재공 0), 선행 공정
    # OPER001에 900장이 쌓여 있다 — 이 창(WINDOW) 안에 흘러들어올 수 있으므로
    # OPER002의 TARGET_QTY에도 반영돼야 한다(그렇지 않으면 target=0으로 아예
    # 할당 대상에서 빠짐).
    raw = {
        "abstract_arrange": [
            _abstract_row("PPK001", "OPER001", "M1", 60),
            _abstract_row("PPK001", "OPER002", "M1", 60),
        ],
        "discrete_arrange": [
            _discrete_row("EQP001", "L1", "PPK001", "OPER001", "M1", wf_qty=900),
        ] + [
            # 풀을 넉넉히(30대) 둬서 재공 반영 여부만 경쟁 없이 순수하게 검증한다.
            _discrete_row(f"EQP{i:03d}", f"LX{i}", "PPK001", "OPER001", "M1", wf_qty=0)
            for i in range(2, 31)
        ],
        "plan": [
            # 계획을 작게 둬서(target=100→10대) 물리 대수 경쟁 없이 둘 다 완전히
            # 채워지게 하고, target 값 자체(inflow 반영 여부)만 확인한다.
            _plan_row("PPK001", "OPER001", d0=100, priority=1),
            _plan_row("PPK001", "OPER002", d0=100, priority=1),
        ],
        "flow": [_flow_row("PPK001", 1, "OPER001"), _flow_row("PPK001", 2, "OPER002")],
    }
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    by_oper = {r["OPER_ID"]: r for r in rows}
    # OPER002는 현재 재공이 0이지만 선행 공정(OPER001)의 900장이 도달 가능
    # 재공으로 잡혀 target이 plan(100)까지 온전히 인정된다.
    assert by_oper["OPER002"]["WIP_QTY"] == 900
    assert by_oper["OPER002"]["TARGET_QTY"] == 100
    assert by_oper["OPER002"]["ALLOC_EQP_CNT"] == 10
    # OPER001 자신의 값은 상류가 없으니 그대로.
    assert by_oper["OPER001"]["WIP_QTY"] == 900


# ── 전환(conversion) 인지 배정 ───────────────────────────────────────────────

def _batch_info_row(ppk, oper_id, lot_cd, temp):
    return {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id, "LOT_CD": lot_cd, "TEMP": temp}


def _eqp_initial_state_row(eqp_id, lot_cd, temp):
    return {"EQP_ID": eqp_id, "LOT_CD": lot_cd, "TEMP": temp}


def _conv_test_raw(target_d0):
    # PPK001/OPER001 전용모델 M1, 물리 대수 4대: EQP001/002는 이미 목표 배치
    # (LC1/T1)로 셋업(전환 불필요), EQP003은 다른 배치(전환 필요),
    # EQP004는 초기 셋업 정보 자체가 없음(첫 배정 = 전환 불필요로 취급).
    return {
        "abstract_arrange": [_abstract_row("PPK001", "OPER001", "M1", 60)],
        "discrete_arrange": [
            _discrete_row("EQP001", "L1", "PPK001", "OPER001", "M1", wf_qty=1000),
            _discrete_row("EQP002", "L2", "PPK001", "OPER001", "M1", wf_qty=1000),
            _discrete_row("EQP003", "L3", "PPK001", "OPER001", "M1", wf_qty=1000),
            _discrete_row("EQP004", "L4", "PPK001", "OPER001", "M1", wf_qty=1000),
        ],
        "plan": [_plan_row("PPK001", "OPER001", d0=target_d0, priority=1)],
        "flow": [_flow_row("PPK001", 1, "OPER001")],
        "batch_info": [_batch_info_row("PPK001", "OPER001", "LC1", "T1")],
        "eqp_initial_state": [
            _eqp_initial_state_row("EQP001", "LC1", "T1"),   # 무전환
            _eqp_initial_state_row("EQP002", "LC1", "T1"),   # 무전환
            _eqp_initial_state_row("EQP003", "LC2", "T2"),   # 전환 필요
            # EQP004: 초기 셋업 정보 없음 → 무전환 취급
        ],
    }


def test_prefers_already_setup_equipment_over_conversion_needed():
    # target=20 → capa/대=10 → 2대만 필요. 무전환 후보(EQP001,002,004)가 3대나
    # 있으니 전환이 필요한 EQP003은 전혀 안 써야 한다.
    raw = _conv_test_raw(target_d0=20)
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    row = rows[0]
    assert row["ALLOC_EQP_CNT"] == 2
    assert row["NEEDS_CONV_EQP_CNT"] == 0


def test_draws_conversion_needed_unit_when_free_pool_insufficient():
    # target=40 → 4대 필요. 무전환 후보는 3대뿐이라 1대는 전환이 필요한
    # EQP003에서 끌어와야 한다(전환 예산 무제한 기본값).
    raw = _conv_test_raw(target_d0=40)
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    row = rows[0]
    assert row["ALLOC_EQP_CNT"] == 4
    assert row["NEEDS_CONV_EQP_CNT"] == 1


def test_max_conversions_budget_caps_conversion_needed_allocation():
    CONFIG.env.max_conversions = 0  # 전체 전환 예산 0 → 전환이 필요한 장비는 아예 못 씀
    raw = _conv_test_raw(target_d0=40)
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    row = rows[0]
    assert row["ALLOC_EQP_CNT"] == 3  # 무전환 후보(EQP001,002,004)만
    assert row["NEEDS_CONV_EQP_CNT"] == 0


def test_max_conversions_per_eqp_zero_forbids_any_conversion():
    CONFIG.env.max_conversions_per_eqp = 0
    raw = _conv_test_raw(target_d0=40)
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    row = rows[0]
    assert row["ALLOC_EQP_CNT"] == 3
    assert row["NEEDS_CONV_EQP_CNT"] == 0


def test_no_batch_info_or_initial_state_is_backward_compatible():
    # batch_info/eqp_initial_state를 아예 안 주면(기존 입력) 전환 판단 자체를
    # 생략하고 모든 장비를 무전환으로 취급한다 — 기존 동작 그대로.
    raw = _conv_test_raw(target_d0=40)
    del raw["batch_info"]
    del raw["eqp_initial_state"]
    rows = compute_eqp_allocation(raw, fac_id="FAC001", rule_timekey=RULE_TIMEKEY, window_minutes=WINDOW)
    row = rows[0]
    assert row["ALLOC_EQP_CNT"] == 4
    assert row["NEEDS_CONV_EQP_CNT"] == 0
