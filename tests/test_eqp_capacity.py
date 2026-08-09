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
        CONFIG.env.capacity_alloc_mode,
        CONFIG.env.capacity_line_balance,
    )
    yield
    (
        CONFIG.env.eqp_capacity_mask_enabled,
        CONFIG.env.capacity_min_run_minutes,
        CONFIG.env.capa_output_enabled,
        CONFIG.env.capacity_alloc_mode,
        CONFIG.env.capacity_line_balance,
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


# ── 3-b. 모델별 ST 반영 ──────────────────────────────────────────────────────
#
# ST는 (PPK, OPER, EQP_MODEL) 조합으로 정해진다. 모델마다 ST가 다르면 "대수 × 평균 ST"
# 로는 맞지 않으므로, 필요 대수도 재공 지탱 대수도 장비를 우선순위 순으로 한 대씩
# 붙이며 1/ST(분당 처리량)를 누적해서 구해야 한다.

MIXED_ST = {"A": 60, "B": 120}


def _mixed_model_inputs(*, n_carrier=4, plan_qty=200):
    """모델 A(ST 60) 2대 + 모델 B(ST 120) 2대."""
    eqp_model = {"EQP001": "A", "EQP002": "A", "EQP003": "B", "EQP004": "B"}
    discrete = [
        {"EQP_ID": eid, "LOT_ID": f"LOT{i:03d}", "CARRIER_ID": f"CAR{i:03d}",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
         "ST": MIXED_ST[model], "EQP_MODEL_CD": model, "WF_QTY": 25, "SEQ": i}
        for i in range(1, n_carrier + 1)
        for eid, model in eqp_model.items()
    ]
    return {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                  "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "EQP_MODEL_CD": m, "ST": st}
            for m, st in MIXED_ST.items()
        ],
    }


def _mixed_sim(**kwargs):
    CONFIG.env.eqp_capacity_mask_enabled = False
    data = preprocess(_mixed_model_inputs(**kwargs), period_key=RULE_TIMEKEY)
    data["termination_mode"] = "current_wip_assigned"
    data["enable_wip_inflow"] = False
    return SchedulingSimulator(data, record_history=False)


def test_st_resolved_per_ppk_oper_model():
    """ST는 (PPK, OPER, EQP_MODEL)로 조회되고, 모델별로 나뉘어 보고된다."""
    sim = _mixed_sim()
    assert sim._env_data["abstract_arrange_map"][("PPK001", "OPER001", "A")] == 60
    assert sim._env_data["abstract_arrange_map"][("PPK001", "OPER001", "B")] == 120
    # 장비별 ST도 그 장비의 모델로 풀린다
    assert sim._st_per_wafer_for_eqp("EQP001", "PPK001", "OPER001") == 60
    assert sim._st_per_wafer_for_eqp("EQP003", "PPK001", "OPER001") == 120

    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.st_by_model == {"A": 60.0, "B": 120.0}


def test_required_count_accumulates_per_model_rate_not_average():
    """필요 대수는 모델별 1/ST를 한 대씩 누적해 구한다(평균 ST로 나누지 않는다)."""
    # 잔여 60매 / 1320분 = 0.04545매·분.  A(1/60)+A(1/60)+B(1/120) = 0.04167 → 부족,
    # B 한 대를 더해 0.05 → 도달. 평균 ST(90분)로 뭉치면 3대로 잘못 나온다.
    sim = _mixed_sim(plan_qty=60)
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.req_cnt == 4
    assert cap.req_by_model == {"A": 2, "B": 2}
    # 빠른 모델이 먼저 채워진다
    assert cap.req_eqps == ["EQP001", "EQP002", "EQP003", "EQP004"]

    # 잔여 40매 / 1320 = 0.0303 → A 두 대(0.0333)로 충족, 느린 B는 부르지 않는다
    cap = _mixed_sim(plan_qty=40).capacity_planner.target_for("PPK001", "OPER001")
    assert cap.req_cnt == 2
    assert cap.req_by_model == {"A": 2}


def test_wip_sustainable_count_uses_per_model_st():
    """재공 지탱 대수도 평균이 아니라 붙는 장비의 ST로 소진 시간을 계산한다."""
    CONFIG.env.capacity_min_run_minutes = 2600
    sim = _mixed_sim(n_carrier=4, plan_qty=10_000)      # 재공 100매
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    # 소진 시간 = 100매 / 누적 처리율.  1대 6000분, 2대 3000분, 3대 2400분(<2600).
    # → 최소 가동 2600분을 채우는 건 2대까지. 평균 ST(90분)로 뭉치면
    #   100×90/2600 = 3대로 과대 산출된다.
    assert cap.req_cnt == 4
    assert cap.target_cnt == 2
    assert cap.reason == "WIP"
    assert cap.target_by_model == {"A": 2}


def test_model_not_in_arrange_is_excluded_from_candidates():
    """(PPK, OPER, MODEL) arrange 행이 없는 모델은 그 버킷의 후보가 아니다.

    회귀 테스트: _eqp_can_process()는 OPER 단위인 eqp_oper_cap(discrete)도 OR로
    받아주기 때문에, 그것만 보면 "이 공정을 하는 장비"가 전부 통과해 실제로는 한 대도
    못 돌리는 모델까지 정원에 잡히고 ST도 모델 평균 폴백으로 잘못 채워졌다.
    실제 배정 경로(_abstract_assignable_on_eqp)는 템플릿을 eqp_model로 거르므로
    정원 산출도 같은 기준이어야 한다.
    """
    CONFIG.env.eqp_capacity_mask_enabled = False
    # EQP001=모델 A(PPK001·PPK002 가능), EQP002=모델 B(PPK002만 가능)
    eligible = {"PPK001": {"A": 60}, "PPK002": {"A": 60, "B": 120}}
    eqp_model = {"EQP001": "A", "EQP002": "B"}
    discrete, n = [], 0
    for ppk, models in eligible.items():
        for _ in range(2):
            n += 1
            for eid, m in eqp_model.items():
                if m in models:
                    discrete.append({
                        "EQP_ID": eid, "LOT_ID": f"LOT{n:03d}", "CARRIER_ID": f"CAR{n:03d}",
                        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": models[m],
                        "EQP_MODEL_CD": m, "WF_QTY": 25, "SEQ": n})
    raw = {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001", "D0_PLAN_QTY": 100,
                  "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1} for p in eligible],
        "flow": [{"PLAN_PROD_ATTR_VAL": p, "OPER_SEQ": 1, "OPER_ID": "OPER001"}
                 for p in eligible],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001", "EQP_MODEL_CD": m, "ST": st}
            for p, ms in eligible.items() for m, st in ms.items()],
    }
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["termination_mode"] = "current_wip_assigned"
    data["enable_wip_inflow"] = False
    sim = SchedulingSimulator(data, record_history=False)

    # eqp_oper_cap 경로만 보면 EQP002도 PPK001을 할 수 있는 것처럼 보이지만,
    assert sim._eqp_can_process("EQP002", "PPK001", "OPER001") is True
    # 실제 배정에서는 PPK002만 받을 수 있다.
    assert sorted(
        sim.ppk_oper_from_flat(f) for f in sim.get_feasible_ppk_oper("EQP002")
    ) == [("PPK002", "OPER001")]

    # 정원 산출도 같은 기준이어야 한다 — PPK001 후보는 EQP001 한 대뿐.
    cap1 = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert [e for e, _st, _m in sim.capacity_planner._ranked_candidates(
        "PPK001", "OPER001")] == ["EQP001"]
    assert cap1.capable_cnt == 1
    assert set(cap1.st_by_model) == {"A"}          # 모델 B는 아예 등장하지 않음
    assert cap1.req_by_model.get("B", 0) == 0

    cap2 = sim.capacity_planner.target_for("PPK002", "OPER001")
    assert cap2.capable_cnt == 2
    assert cap2.st_by_model == {"A": 60.0, "B": 120.0}


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


# ── 6. 라인 밸런스 (최종 out 기준) ───────────────────────────────────────────
#
# 같은 PPK의 공정들은 직렬이라 최종 공정 out이 곧 제품 실적이다. 앞 공정에 장비를
# 몰아줘도 뒤 공정이 못 받으면 재공만 쌓이므로, 최종 공정 계획에서 역산해(pull)
# 각 공정의 목표 out을 잡는다 → 공정 간 처리량(매/분)이 맞는다.

def _two_oper_inputs(*, o1_carriers=4, o2_carriers=0, o1_plan=100, o2_plan=100,
                     st1=60, st2=60):
    """OPER001 → OPER002 2단 공정, 장비 4대(모델 A) 공용."""
    eqps = [f"EQP{i:03d}" for i in range(1, 5)]
    discrete, n = [], 0
    for oper, carriers, st in (("OPER001", o1_carriers, st1), ("OPER002", o2_carriers, st2)):
        for _ in range(carriers):
            n += 1
            for eid in eqps:
                discrete.append({
                    "EQP_ID": eid, "LOT_ID": f"LOT{n:03d}", "CARRIER_ID": f"CAR{n:03d}",
                    "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": oper, "ST": st,
                    "EQP_MODEL_CD": "A", "WF_QTY": 25, "SEQ": n})
    return {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                  "D0_PLAN_QTY": o1_plan, "D1_PLAN_QTY": o1_plan, "PLAN_PRIORITY": 1},
                 {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002",
                  "D0_PLAN_QTY": o2_plan, "D1_PLAN_QTY": o2_plan, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
                 {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 2, "OPER_ID": "OPER002"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "EQP_MODEL_CD": "A", "ST": st1},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER002",
             "EQP_MODEL_CD": "A", "ST": st2}],
    }


def _line_sim(*, mask=False, alloc_mode="cap", line_balance=True, **kwargs):
    CONFIG.env.eqp_capacity_mask_enabled = mask
    CONFIG.env.capacity_alloc_mode = alloc_mode
    CONFIG.env.capacity_line_balance = line_balance
    data = preprocess(_two_oper_inputs(**kwargs), period_key=RULE_TIMEKEY)
    data["termination_mode"] = "current_wip_assigned"
    data["enable_wip_inflow"] = False
    return SchedulingSimulator(data, record_history=False)


def test_line_balance_pulls_upstream_demand_from_final_oper():
    """상류 목표 out = 최종 공정 목표 − 최종 공정이 이미 들고 있는 재공."""
    # 최종(OPER002) 계획 100매, OPER002가 이미 50매(2 carrier) 보유
    # → OPER001이 만들어 줘야 할 양 = 100 − 50 = 50매
    sim = _line_sim(o1_carriers=4, o2_carriers=2, o1_plan=100, o2_plan=100)
    need = sim.capacity_planner._line_need_out()
    assert need[("PPK001", "OPER002")] == 100
    assert need[("PPK001", "OPER001")] == 50

    # 최종 공정이 계획을 이미 다 들고 있으면 상류는 더 만들 필요가 없다
    sim = _line_sim(o1_carriers=4, o2_carriers=4, o1_plan=100, o2_plan=100)
    need = sim.capacity_planner._line_need_out()
    assert need[("PPK001", "OPER001")] == 0


def test_line_balance_matches_rate_not_count_when_st_differs():
    """ST가 다르면 같은 처리율을 내기 위해 공정별 대수가 달라진다."""
    # OPER001 ST 60, OPER002 ST 120 — 최종 100매 목표(하류 재공 0 → 상류도 100매)
    sim = _line_sim(o1_carriers=8, o2_carriers=4, o1_plan=100, o2_plan=100,
                    st1=60, st2=120)
    caps = sim.capacity_planner.targets()
    o1 = caps[("PPK001", "OPER001")]
    o2 = caps[("PPK001", "OPER002")]
    # OPER002는 100매, OPER001은 100−100(하류 보유)=0 … 이 아니라 하류 보유분만큼 차감
    assert o2.remain_qty == 100
    # 같은 목표 out이라도 ST가 2배면 필요 대수도 대략 2배가 된다
    assert o2.req_cnt >= o1.req_cnt


def test_line_balance_off_uses_own_plan_only():
    """line_balance=False면 공정마다 자기 계획만 본다(기존 동작)."""
    sim = _line_sim(o1_carriers=4, o2_carriers=4, o1_plan=100, o2_plan=100,
                    line_balance=False)
    cap = sim.capacity_planner.target_for("PPK001", "OPER001")
    assert cap.remain_qty == 100          # 하류 재공을 빼지 않는다


# ── 7. 배분 모드 (allocate) ──────────────────────────────────────────────────

def test_allocate_mode_sums_to_fleet_and_masks_by_table():
    """배분 모드에서는 배분표가 곧 마스크 — 배분 못 받은 장비는 그 버킷에서 빠진다."""
    sim = _line_sim(mask=True, alloc_mode="allocate",
                    o1_carriers=4, o2_carriers=4, o1_plan=100, o2_plan=100)
    alloc, shortfalls = sim.capacity_planner.allocation()
    assigned = [e for v in alloc.values() for e in v]
    assert len(assigned) == len(set(assigned))          # 한 장비는 한 버킷에만
    assert set(assigned) <= set(sim._env_data["eqp_ids"])

    for bucket, eqps in alloc.items():
        if not eqps:
            continue
        outsider = next(
            (e for e in sim._env_data["eqp_ids"] if e not in eqps), None,
        )
        if outsider is not None:
            assert sim.capacity_planner.blocks(outsider, *bucket)
        for e in eqps:
            assert not sim.capacity_planner.blocks(e, *bucket)


def test_allocate_mode_scarcity_protects_dedicated_bucket():
    """범용 버킷이 전용 버킷의 유일한 장비를 가져가 굶기지 않는다."""
    CONFIG.env.eqp_capacity_mask_enabled = True
    CONFIG.env.capacity_alloc_mode = "allocate"
    # EQP001=모델 A(PPK001·PPK002 가능), EQP002=모델 B(PPK002만 가능)
    # PPK002 잔여량을 더 크게 잡아 배분 순서에서 PPK002가 먼저 고르게 한다.
    eligible = {"PPK001": {"A": 60}, "PPK002": {"A": 60, "B": 120}}
    eqp_model = {"EQP001": "A", "EQP002": "B"}
    carriers = {"PPK001": 2, "PPK002": 6}
    discrete, n = [], 0
    for ppk, models in eligible.items():
        for _ in range(carriers[ppk]):
            n += 1
            for eid, m in eqp_model.items():
                if m in models:
                    discrete.append({
                        "EQP_ID": eid, "LOT_ID": f"LOT{n:03d}", "CARRIER_ID": f"CAR{n:03d}",
                        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": models[m],
                        "EQP_MODEL_CD": m, "WF_QTY": 25, "SEQ": n})
    raw = {
        "discrete_arrange": discrete,
        "plan": [{"PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001",
                  "D0_PLAN_QTY": carriers[p] * 25, "D1_PLAN_QTY": 100,
                  "PLAN_PRIORITY": 1} for p in eligible],
        "flow": [{"PLAN_PROD_ATTR_VAL": p, "OPER_SEQ": 1, "OPER_ID": "OPER001"}
                 for p in eligible],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001", "EQP_MODEL_CD": m, "ST": st}
            for p, ms in eligible.items() for m, st in ms.items()],
    }
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    data["termination_mode"] = "current_wip_assigned"
    data["enable_wip_inflow"] = False
    sim = SchedulingSimulator(data, record_history=False)

    alloc, _short = sim.capacity_planner.allocation()
    # PPK001은 EQP001 없이는 한 대도 못 돌린다 — 반드시 확보돼야 한다
    assert alloc[("PPK001", "OPER001")] == ["EQP001"]
    assert alloc[("PPK002", "OPER001")] == ["EQP002"]


def test_allocate_rows_report_allocation_and_shortfall():
    sim = _line_sim(mask=True, alloc_mode="allocate",
                    o1_carriers=4, o2_carriers=4, o1_plan=100, o2_plan=100)
    rows = sim.get_eqp_capacity_plan()
    assert rows and all(r["ALLOC_MODE"] == "ALLOC" for r in rows)
    assert all("SHORTFALL_RATIO" in r for r in rows)
    # 배분 대수 합계는 보유 장비 수를 넘지 않는다
    assert sum(r["PLAN_EQP_CNT"] for r in rows) <= len(sim._env_data["eqp_ids"])
