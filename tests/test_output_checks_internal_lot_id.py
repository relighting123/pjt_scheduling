"""
tests/test_output_checks_internal_lot_id.py

validation/output_checks.py는 schedule의 row["LOT_ID"]를 논리(비즈니스) LOT_ID로,
env_data["lots"]/proc_time_matrix/eqp_forced_queue는 여전히 내부 carrier 단위
lot_id로 색인한다. CARRIER_ID가 LOT_ID와 다른(1:N) 정상적인 경우에도
처리시간불일치/미배정으로 잘못 잡히면 안 된다.
"""
from validation.output_checks import (
    check_completeness,
    check_forced_placement,
    check_processing_time,
    validate_schedule_output,
)

PPK = "PPK001"
OPER = "OPER001"


def _env_data(**overrides):
    data = {
        "lots": [
            {"lot_id": "CAR001", "PLAN_PROD_ATTR_VAL": PPK, "oper_id": OPER, "wf_qty": 25},
        ],
        "proc_time_matrix": {("CAR001", "EQP001", OPER): 30},
        # 일부러 실제(30)와 다른 abstract 평균(99)을 넣어, discrete 매칭이 깨지면
        # 이 값으로 잘못 대체돼 mismatch가 나는지 확인한다.
        "abstract_arrange_map": {(PPK, OPER, "A"): 99},
        "eqp_model_map": {"EQP001": "A"},
        "eqp_oper_cap": {},
        "eqp_forced_queue": {},
    }
    data.update(overrides)
    return data


def _schedule_row(**overrides):
    row = {
        "LOT_ID": "LOT001",       # 논리(비즈니스) LOT_ID — CARRIER_ID와 다름
        "CARRIER_ID": "CAR001",   # 내부 lot_id와 동일
        "EQP_ID": "EQP001",
        "OPER_ID": OPER,
        "PLAN_PROD_ATTR_VAL": PPK,
        "WF_QTY": 25,
        "START_TM": 0,
        "END_TM": 750,  # 30 * 25 = 정확히 discrete ST 기준 기대 처리시간
    }
    row.update(overrides)
    return row


def test_processing_time_uses_carrier_id_not_logical_lot_id():
    schedule = [_schedule_row()]
    mismatches = check_processing_time(schedule, _env_data())
    assert mismatches == []


def test_processing_time_still_flags_real_mismatch():
    schedule = [_schedule_row(END_TM=999)]  # discrete 기대(750)와 실제 다름
    mismatches = check_processing_time(schedule, _env_data())
    assert len(mismatches) == 1
    assert mismatches[0]["lot_id"] == "LOT001"
    assert mismatches[0]["carrier_id"] == "CAR001"


def test_completeness_matches_scheduled_lot_by_carrier_id():
    schedule = [_schedule_row()]
    missing = check_completeness(schedule, _env_data(), stats={})
    assert missing == []


def test_completeness_reports_real_gap_with_business_lot_id():
    # 아무 것도 스케줄되지 않았고 잔여 재공 통계도 없음 → 진짜 누락.
    missing = check_completeness([], _env_data(), stats={})
    assert len(missing) == 1
    assert missing[0]["lot_id"] == "CAR001"  # logical_lot_id 없으면 내부 id로 대체
    assert missing[0]["carrier_id"] == "CAR001"


def test_forced_placement_matches_by_carrier_id():
    schedule = [_schedule_row()]
    env_data = _env_data(eqp_forced_queue={"EQP001": ["CAR001"]})
    violations = check_forced_placement(schedule, env_data)
    assert violations == []


def test_validate_schedule_output_reports_ok_for_normal_carrier_split():
    result = {"schedule": [_schedule_row()], "stats": {}}
    validation = validate_schedule_output(result, _env_data())
    assert validation["ok"] is True
    assert validation["summary"]["proc_time_mismatch_count"] == 0
    assert validation["summary"]["unassigned_lot_count"] == 0
