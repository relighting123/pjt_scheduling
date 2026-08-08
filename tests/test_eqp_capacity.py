"""
tests/test_eqp_capacity.py

제품(PPK)·공정(OPER)별 적정 장비 대수 산출과 정원 마스킹 검증.

  1. 필요 대수 = 마감(soft_cutoff와 sim_end 중 이른 쪽)까지의 잔여량을 모델 ST로
     나눈 값 — 계획이 작으면 적은 대수, 크면 처리 가능 장비 전부까지 늘어난다.
  2. 재공이 그 대수를 먹이지 못하면(carrier 부족 / 작업량 부족) 정원이 줄고,
     재공이 아예 없으면 구동 불가(RUNNABLE_YN='N')로 남는다.
  3. 정원이 정해지면 이미 그 버킷을 돌고 있거나 마지막 셋업이 같은 장비는 계속
     통과하고, 그 외 신규 장비만 정원 초과 시 마스크에서 빠진다.
  4. 마스킹을 켜도 재공이 남은 버킷은 최소 1대가 열려 있어 재공이 통째로 묶이지 않는다.
  5. 산출 결과와 실제 배치 대수가 RTS_EQPCAPA_INF/HIS 스키마로 나간다.
"""
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from data.writer.rts_json import build_rts_output
from data.writer.rts_sql import build_writer_sql_scripts
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"
ST = 60


@pytest.fixture(autouse=True)
def _restore_env_config():
    original = (
        CONFIG.env.eqp_capacity_mask_enabled,
        CONFIG.env.capacity_min_run_minutes,
        CONFIG.env.capa_output_enabled,
    )
    yield
    (
        CONFIG.env.eqp_capacity_mask_enabled,
        CONFIG.env.capacity_min_run_minutes,
        CONFIG.env.capa_output_enabled,
    ) = original


def _inputs(*, n_eqp=4, n_carrier=8, plan_qty=200, st=ST):
    """EQP n대 × 모델 A, carrier n_carrier개(각 25매)의 단일 공정 데이터."""
    discrete = []
    for i in range(1, n_carrier + 1):
        for e in range(1, n_eqp + 1):
            # 전건 자유배정(WAIT) — 모든 carrier가 모든 EQP에 투입 가능
            discrete.append({
                "EQP_ID": f"EQP{e:03d}", "LOT_ID": f"LOT{i:03d}",
                "CARRIER_ID": f"CAR{i:03d}", "PLAN_PROD_ATTR_VAL": "PPK001",
                "OPER_ID": "OPER001", "ST": st, "EQP_MODEL_CD": "A",
                "WF_QTY": 25, "SEQ": i,
            })
    return {
        "discrete_arrange": discrete,
        "plan": [{
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
            "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1,
        }],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [{
            "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
            "EQP_MODEL_CD": "A", "ST": st,
        }],
    }


def _sim(*, mask=False, **kwargs):
    CONFIG.env.eqp_capacity_mask_enabled = mask
    data = preprocess(_inputs(**kwargs), period_key=RULE_TIMEKEY)
    data["termination_mode"] = "current_wip_assigned"
    data["enable_wip_inflow"] = False
    return SchedulingSimulator(data, record_history=False)


# ── 1. 필요 대수 산출 ────────────────────────────────────────────────────────

def test_required_count_follows_remaining_qty_over_deadline():
    """필요 대수는 마감까지의 잔여량 ÷ (남은시간/ST)으로 커진다."""
    # 마감 1320분, ST 60분/매 → 1대가 마감까지 처리 가능한 양 = 22매
    sim = _sim(plan_qty=20)
    assert sim.capacity_planner.target_for("PPK001", "OPER001").req_cnt == 1

    sim = _sim(plan_qty=60)
    assert sim.capacity_planner.target_for("PPK001", "OPER001").req_cnt == 3

    # 계획이 처리 가능 장비의 총 capa를 넘으면 있는 장비 전부(4대)가 상한이고,
    # 다 붙여도 마감까지 못 끝낸다는 사실이 사유로 남는다.
    sim = _sim(plan_qty=1000, n_carrier=40)
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.req_cnt == 4
    assert cap.reason == "EQP"


def test_deadline_is_earlier_of_soft_cutoff_and_sim_end():
    """sim_end를 짧게 잡은 데이터에서 soft_cutoff를 그대로 쓰면 필요 대수가 과소 산출된다."""
    CONFIG.env.eqp_capacity_mask_enabled = False
    data = preprocess(_inputs(plan_qty=100), period_key=RULE_TIMEKEY)
    data["sim_end_minutes"] = 480          # soft_cutoff(1320)보다 이른 마감
    sim = SchedulingSimulator(data, record_history=False)
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.horizon_min == 480
    # 480분 / ST 60 = 1대당 8매 → 100매엔 4대(있는 장비 전부)가 필요
    assert cap.req_cnt == 4


def test_plan_qty_capped_by_available_wip():
    """계획이 아무리 커도 만들 수 있는 재공 이상은 잔여량으로 잡지 않는다."""
    sim = _sim(plan_qty=10_000, n_carrier=2)   # 재공 2 carrier = 50매
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.remain_qty == 50


# ── 2. 재공 적정성 사전 체크 ─────────────────────────────────────────────────

def test_target_reduced_when_wip_cannot_feed_required_count():
    """ST상 3대가 필요해도 재공이 2 carrier뿐이면 정원은 2대로 줄고 사유가 남는다."""
    sim = _sim(plan_qty=10_000, n_carrier=2)
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    # 잔여량 50매(2 carrier) ÷ 마감 1320분 → 1대(1/60매·분)로는 부족해 3대가 필요
    assert cap.req_cnt == 3
    assert cap.target_cnt == 2
    assert cap.reason == "WIP"
    assert cap.runnable is True


def test_min_run_minutes_limits_target():
    """한 대당 최소 가동시간을 늘리면 같은 재공으로 지탱 가능한 대수가 줄어든다."""
    CONFIG.env.capacity_min_run_minutes = 60
    sim = _sim(plan_qty=10_000, n_carrier=4)
    assert sim.capacity_planner.target_for("PPK001", "OPER001").target_cnt == 4

    # 재공 작업량 = 100매 × 60분 = 6000분. 대당 3000분을 요구하면 2대까지만 인정.
    CONFIG.env.capacity_min_run_minutes = 3000
    sim = _sim(plan_qty=10_000, n_carrier=4)
    assert sim.capacity_planner.target_for("PPK001", "OPER001").target_cnt == 2


def test_not_runnable_without_wip():
    """재공이 없는 버킷은 구동 불가 — 정원 0대."""
    sim = _sim(plan_qty=200, n_carrier=8)
    sim._wip_pool[("PPK001", "OPER001")]["wip_qty"] = 0
    sim._wip_pool[("PPK001", "OPER001")]["lot_ids"] = []
    sim._invalidate_caches()
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.runnable is False
    assert cap.target_cnt == 0
    assert cap.reason == "NO_WIP"


# ── 3. 정원 마스킹 ───────────────────────────────────────────────────────────

def test_mask_blocks_extra_eqp_beyond_target():
    """정원 1대인 버킷은 첫 장비가 잡은 뒤 나머지 장비의 마스크에서 빠진다."""
    sim = _sim(mask=True, plan_qty=20)
    assert sim.capacity_planner.target_for("PPK001", "OPER001").target_cnt == 1

    first = sim.current_idle_eqp()
    assert sim.assign_ppk_oper(first, "PPK001", "OPER001") >= 0

    others = [e for e in sim._env_data["eqp_ids"] if e != first]
    for eqp_id in others:
        assert sim.get_feasible_ppk_oper(eqp_id) == [], eqp_id
        assert sim.capacity_planner.blocks(eqp_id, "PPK001", "OPER001")
    assert not sim.capacity_planner.blocks(first, "PPK001", "OPER001")


def test_mask_off_leaves_all_eqps_feasible():
    """마스킹을 끄면 같은 조건에서 모든 장비가 그대로 후보로 남는다(기존 동작)."""
    sim = _sim(mask=False, plan_qty=20)
    first = sim.current_idle_eqp()
    sim.assign_ppk_oper(first, "PPK001", "OPER001")
    others = [e for e in sim._env_data["eqp_ids"] if e != first]
    assert all(sim.get_feasible_ppk_oper(e) for e in others)


def test_incumbent_eqp_keeps_slot_after_finishing():
    """가공을 마치고 idle이 된 '하던 장비'는 정원 안에 남아 계속 그 버킷을 받는다."""
    sim = _sim(mask=True, plan_qty=20)
    first = sim.current_idle_eqp()
    sim.assign_ppk_oper(first, "PPK001", "OPER001")

    # 가공 종료까지 시간 전진 — first는 idle이 되지만 마지막 셋업이 그대로 남는다
    sim._advance_to_next_decision()
    while sim.eqps[first].status == "busy" and sim._event_q:
        sim._advance_to_next_decision()

    assert sim.eqps[first].prev_prod == "PPK001"
    assert first in sim.capacity_planner.occupancy("PPK001", "OPER001")
    assert not sim.capacity_planner.blocks(first, "PPK001", "OPER001")


def test_down_eqp_does_not_hold_a_slot():
    """다운된 장비는 마지막 셋업이 같아도 정원을 차지하지 않는다."""
    sim = _sim(mask=True, plan_qty=20)
    first = sim.current_idle_eqp()
    sim.assign_ppk_oper(first, "PPK001", "OPER001")
    assert sim.capacity_planner.occupancy("PPK001", "OPER001") == [first]

    sim.eqps[first].status = "down"
    sim._invalidate_caches()
    assert sim.capacity_planner.occupancy("PPK001", "OPER001") == []


def test_mask_never_locks_out_all_eqps_while_wip_remains():
    """계획을 이미 채웠어도 재공이 남아 있으면 정원은 최소 1대 — 재공이 묶이지 않는다."""
    sim = _sim(mask=True, plan_qty=20)
    sim.stats["completed_qty"][("PPK001", "OPER001")] = 10_000   # 계획 초과 달성
    sim._invalidate_caches()

    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.remain_qty == 0
    assert cap.reason == "DONE"
    assert cap.target_cnt == 1
    assert any(sim.get_feasible_ppk_oper(e) for e in sim._env_data["eqp_ids"])


# ── 4. 배치 우선순위 ─────────────────────────────────────────────────────────

def test_target_prefers_eqp_already_set_up_for_the_bucket():
    """정원 선정은 마지막 셋업이 같은 장비를 먼저 세운다(가급적 하던 장비 그대로)."""
    sim = _sim(plan_qty=20)
    incumbent = "EQP003"
    sim.eqps[incumbent].prev_prod = "PPK001"
    sim.eqps[incumbent].prev_oper = "OPER001"
    sim._invalidate_caches()

    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.target_eqps[0] == incumbent


# ── 5. 출력(DB 스키마) ───────────────────────────────────────────────────────

def test_capacity_plan_reports_required_and_allocated_counts():
    sim = _sim(mask=True, plan_qty=20)
    first = sim.current_idle_eqp()
    sim.assign_ppk_oper(first, "PPK001", "OPER001")

    rows = sim.get_eqp_capacity_plan()
    row = next(r for r in rows if r["EQP_MODEL_CD"] == "A")
    assert row["PLAN_PROD_ATTR_VAL"] == "PPK001"
    assert row["OPER_ID"] == "OPER001"
    assert row["REQ_EQP_CNT"] == 1
    assert row["PLAN_EQP_CNT"] == 1
    assert row["ALLOC_EQP_CNT"] == 1
    assert row["ALLOC_EQP_LVAL"] == first
    assert row["RUNNABLE_YN"] == "Y"
    assert row["MASK_APPLY_YN"] == "Y"


def test_rts_eqpcapa_payload_and_sql():
    sim = _sim(mask=True, plan_qty=20)
    first = sim.current_idle_eqp()
    sim.assign_ppk_oper(first, "PPK001", "OPER001")

    result = {
        "schedule": sim.schedule,
        "conversion_plans": [],
        "capacity_plan": sim.get_eqp_capacity_plan(),
        "algorithm": "minprogress",
    }
    payload = build_rts_output(
        result, sim._env_data, fac_id="FAC001", rule_timekey=RULE_TIMEKEY,
    )
    rows = payload["RTS_EQPCAPA_INF"]
    assert rows and rows[0]["FAC_ID"] == "FAC001"
    assert rows[0]["RULE_TIMEKEY"] == RULE_TIMEKEY
    assert rows[0]["FUNCTION_NM"] == "minprogress"
    assert rows[0]["REQ_EQP_CNT"] == 1
    assert rows[0]["ALLOC_EQP_CNT"] == 1

    scripts = build_writer_sql_scripts(payload, include_history=True)
    inf_sql = scripts["rts_eqpcapa_inf.sql"]
    assert "DELETE FROM RTS_EQPCAPA_INF" in inf_sql
    assert f"WHERE FAC_ID = 'FAC001' AND RULE_TIMEKEY = '{RULE_TIMEKEY}'" in inf_sql
    assert "INSERT INTO RTS_EQPCAPA_INF" in inf_sql

    his_sql = scripts["rts_eqpcapa_his.sql"]
    assert "DELETE FROM RTS_EQPCAPA_HIS" not in his_sql   # HIS는 누적만
    assert "EXEC_TIMEKEY" in his_sql


def test_capa_output_disabled_emits_no_rows():
    CONFIG.env.capa_output_enabled = False
    sim = _sim(plan_qty=20)
    result = {
        "schedule": sim.schedule,
        "conversion_plans": [],
        "capacity_plan": sim.get_eqp_capacity_plan(),
        "algorithm": "minprogress",
    }
    payload = build_rts_output(
        result, sim._env_data, fac_id="FAC001", rule_timekey=RULE_TIMEKEY,
    )
    assert payload["RTS_EQPCAPA_INF"] == []
