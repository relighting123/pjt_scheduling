"""validation/output_checks.py – 스케줄링 결과 정합성 검증 테스트."""
from data.loader.fetch import validate_data
from data.loader.preprocess import preprocess
from utils.helpers import effective_proc_time
from validation.output_checks import (
    check_completeness,
    check_eligibility,
    check_processing_time,
    validate_schedule_output,
)


def _build_env():
    """
    LOT001 → EQP001(모델 A), LOT002 → EQP002(모델 B) : PPK001/OPER001
    LOT003 → EQP003(모델 C) : PPK002/OPER002 (다른 제품/공정, 모델 C는 PPK001/OPER001에는 무자격)
    """
    raw = {
        "discrete_arrange": [
            {
                "EQP_ID": "EQP001", "LOT_ID": "LOT001",
                "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                "ST": 10, "EQP_MODEL_CD": "A", "WF_QTY": 25,
            },
            {
                "EQP_ID": "EQP002", "LOT_ID": "LOT002",
                "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                "ST": 12, "EQP_MODEL_CD": "B", "WF_QTY": 25,
            },
            {
                "EQP_ID": "EQP003", "LOT_ID": "LOT003",
                "PLAN_PROD_ATTR_VAL": "PPK002", "OPER_ID": "OPER002",
                "ST": 5, "EQP_MODEL_CD": "C", "WF_QTY": 10,
            },
        ],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 10},
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "B", "ST": 12},
            {"PLAN_PROD_ATTR_VAL": "PPK002", "OPER_ID": "OPER002", "EQP_MODEL_CD": "C", "ST": 5},
        ],
        "plan": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
            {"PLAN_PROD_ATTR_VAL": "PPK002", "OPER_ID": "OPER002", "D0_PLAN_QTY": 10, "D1_PLAN_QTY": 10, "PLAN_PRIORITY": 1},
        ],
        "flow": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
            {"PLAN_PROD_ATTR_VAL": "PPK002", "OPER_SEQ": 1, "OPER_ID": "OPER002"},
        ],
    }
    assert validate_data(raw) == []
    return preprocess(raw)


def _schedule_row(env, lot_id, eqp_id, *, wf_qty=None, proc_time=None):
    lot = next(l for l in env["lots"] if l["lot_id"] == lot_id)
    wf_qty = lot["wf_qty"] if wf_qty is None else wf_qty
    if proc_time is None:
        st = env["proc_time_matrix"].get((lot_id, eqp_id, lot["oper_id"]), lot["processing_time"])
        proc_time = effective_proc_time(st, wf_qty)
    return {
        "EQP_ID": eqp_id,
        "LOT_ID": lot_id,
        "PLAN_PROD_ATTR_VAL": lot["plan_prod_attr_val"],
        "OPER_ID": lot["oper_id"],
        "WF_QTY": wf_qty,
        "ST": env["proc_time_matrix"].get((lot_id, eqp_id, lot["oper_id"]), lot["processing_time"]),
        "START_TM": 0,
        "END_TM": proc_time,
    }


def test_check_eligibility_allows_discrete_and_abstract_matches():
    env = _build_env()
    schedule = [
        _schedule_row(env, "LOT001", "EQP001"),  # discrete match
        _schedule_row(env, "LOT001", "EQP002"),  # PPK001/OPER001 모델 B는 abstract로 허용
    ]
    assert check_eligibility(schedule, env) == []


def test_check_eligibility_flags_ineligible_equipment():
    env = _build_env()
    # EQP003(모델 C)은 PPK001/OPER001에 discrete/abstract 어디에도 없음
    schedule = [_schedule_row(env, "LOT001", "EQP003", proc_time=100)]
    violations = check_eligibility(schedule, env)
    assert len(violations) == 1
    assert violations[0]["lot_id"] == "LOT001"
    assert violations[0]["eqp_id"] == "EQP003"


def test_check_processing_time_matches_input_data():
    env = _build_env()
    schedule = [_schedule_row(env, "LOT001", "EQP001")]
    assert check_processing_time(schedule, env) == []


def test_check_processing_time_flags_mismatch():
    env = _build_env()
    # 기대 처리시간(ST10×WF25=250분)과 다르게 END_TM 조작
    row = _schedule_row(env, "LOT001", "EQP001")
    row["END_TM"] = row["START_TM"] + 999
    mismatches = check_processing_time([row], env)
    assert len(mismatches) == 1
    assert mismatches[0]["expected_proc_time"] == 250
    assert mismatches[0]["actual_proc_time"] == 999


def test_check_completeness_all_assigned():
    env = _build_env()
    schedule = [
        _schedule_row(env, "LOT001", "EQP001"),
        _schedule_row(env, "LOT002", "EQP002"),
        _schedule_row(env, "LOT003", "EQP003"),
    ]
    assert check_completeness(schedule, env, stats={}) == []


def test_check_completeness_flags_silently_dropped_lot():
    env = _build_env()
    schedule = [_schedule_row(env, "LOT001", "EQP001")]
    missing = check_completeness(schedule, env, stats={})
    missing_ids = {m["lot_id"] for m in missing}
    assert missing_ids == {"LOT002", "LOT003"}


def test_check_completeness_explained_by_remaining_wip_not_flagged():
    env = _build_env()
    schedule = [_schedule_row(env, "LOT001", "EQP001")]
    stats = {"remaining_current_wip": {"PPK001|OPER001": 25, "PPK002|OPER002": 10}}
    missing = check_completeness(schedule, env, stats)
    assert missing == []


def test_validate_schedule_output_end_to_end_ok():
    env = _build_env()
    result = {
        "schedule": [
            _schedule_row(env, "LOT001", "EQP001"),
            _schedule_row(env, "LOT002", "EQP002"),
            _schedule_row(env, "LOT003", "EQP003"),
        ],
        "stats": {},
    }
    report = validate_schedule_output(result, env)
    assert report["ok"] is True
    assert report["summary"]["total_scheduled"] == 3


def test_validate_schedule_output_end_to_end_detects_all_issue_types():
    env = _build_env()
    bad_time_row = _schedule_row(env, "LOT002", "EQP002")
    bad_time_row["END_TM"] = bad_time_row["START_TM"] + 1
    result = {
        "schedule": [
            _schedule_row(env, "LOT001", "EQP003", proc_time=100),  # 무자격 배정
            bad_time_row,  # 처리시간 불일치
            # LOT003은 결과에서 누락 (배정 완전성 위반)
        ],
        "stats": {},
    }
    report = validate_schedule_output(result, env)
    assert report["ok"] is False
    assert report["summary"]["eligibility_violation_count"] == 1
    assert report["summary"]["proc_time_mismatch_count"] == 1
    assert report["summary"]["unassigned_lot_count"] == 1
