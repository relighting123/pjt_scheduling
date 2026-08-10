"""
tests/test_eqp_allocator.py

allocation/eqp_allocator.py – PPK/OPER × EQP_MODEL_CD 사전 할당(장비 댓수) 산출
알고리즘 단위 테스트. raw 입력 포맷은 data/loader/preprocess.preprocess()와
동일(SQL 원본 컬럼명)하다.
"""
from allocation.eqp_allocator import build_eqp_alloc_map, compute_eqp_allocation

RULE_TIMEKEY = "20260810070000"
WINDOW = 600  # 테스트 편의를 위한 짧은 창(분) — capa = 600/ST


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


def test_wip_skew_exception_releases_equal_cap():
    # OPER001에 재공이 압도적으로 몰려 있으면(임계 1.5배 이상) 균등 상한 없이
    # 그 OPER을 우선 풀어줘야 한다.
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
    # WIP가 몰린 OPER001이 OPER002보다 확실히 더 많이 배정받아야 한다
    # (극단적으로 몰리면 OPER002는 0대일 수도 있음 — 그것도 정상).
    assert total_by_oper.get("OPER001", 0) > total_by_oper.get("OPER002", 0)


def test_build_eqp_alloc_map_filters_zero_and_groups_by_ppk_oper():
    rows = [
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M1", "ALLOC_EQP_CNT": 3},
        {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "M2", "ALLOC_EQP_CNT": 0},
    ]
    m = build_eqp_alloc_map(rows)
    assert m == {("PPK001", "OPER001"): {"M1": 3}}
