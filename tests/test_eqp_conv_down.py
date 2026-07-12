"""
tests/test_eqp_conv_down.py

외부 확정 EQP 전환 계획(eqp_conv_plan)과 다운타임(eqp_down) 입력이
시뮬레이터에 올바르게 반영되는지 검증한다.
  - START_TM/DOWN_START_TM이 RULE_TIMEKEY 이전/동일이면 reset과 동시에 즉시 적용
  - 미래 시각이면 해당 시각에 적용되고, 그 전까지는 정상적으로 idle 배정이 진행됨
  - DOWN_END_TM이 없으면(무제한 다운) 해당 EQP는 이후 영구히 배정 대상에서 제외됨
  - 반영된 내역은 conversion_plans / down_windows 로 노출되어 Gantt(conv/down)에 쓰인다
"""
import json
from pathlib import Path

import pytest

from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import _normalize_eqp_conv_plan, _normalize_eqp_down, preprocess
from datetime import datetime
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"
BASE_TIME = datetime(2026, 7, 12, 7, 0, 0)


def _next_decision_eqp(sim: SchedulingSimulator):
    """env/scheduling_env.py::_ensure_decision_eqp()와 동일한 시간 전진 규칙.

    current_idle_eqp()가 None이어도(동시각 idle 후보 소진) pending 가공/전환/다운이
    남아있으면 이벤트 큐를 다음 시각까지 전진시켜야 한다.
    """
    if sim.current_idle_eqp() is not None:
        return sim.current_idle_eqp()
    while sim.get_idle_eqps() and sim.current_idle_eqp() is None:
        sim._select_same_time_next_eqp()
        if sim.current_idle_eqp() is not None:
            return sim.current_idle_eqp()
    if sim._has_pending_processing() or sim._down_queue or sim._forced_conv_queue:
        sim._advance_to_next_decision()
        return sim.current_idle_eqp()
    return None


def _run_to_completion(sim: SchedulingSimulator, max_steps: int = 5000) -> None:
    steps = 0
    while not sim.is_done() and steps < max_steps:
        eqp_id = _next_decision_eqp(sim)
        if eqp_id is None:
            break
        lots = sim.available_lots(eqp_id)
        if not lots:
            break
        sim.assign_lot(eqp_id, lots[0]["lot_id"])
        steps += 1


@pytest.fixture()
def env_data_factory(tmp_path: Path):
    def _build(eqp_conv_plan=None, eqp_down=None):
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        generate_sample_data(scenario="default", output_dir=input_dir)
        if eqp_conv_plan is not None:
            (input_dir / "eqp_conv_plan.json").write_text(
                json.dumps(eqp_conv_plan), encoding="utf-8",
            )
        if eqp_down is not None:
            (input_dir / "eqp_down.json").write_text(
                json.dumps(eqp_down), encoding="utf-8",
            )
        raw = load_data(input_dir)
        return preprocess(raw, period_key=RULE_TIMEKEY)
    return _build


def test_normalize_eqp_conv_plan_clamps_past_start():
    rows = [
        {"EQP_ID": "EQP001", "FROM_LOT_CD": "LCA", "FROM_TEMP": "T650",
         "TO_LOT_CD": "LCB", "TO_TEMP": "T700", "START_TM": "20260712060000"},  # 1시간 전 → 0
        {"EQP_ID": "EQP002", "FROM_LOT_CD": "LCA", "FROM_TEMP": "T650",
         "TO_LOT_CD": "LCB", "TO_TEMP": "T700", "START_TM": "20260712080000"},  # 1시간 후 → 60
    ]
    out = _normalize_eqp_conv_plan(rows, BASE_TIME)
    by_eqp = {r["eqp_id"]: r for r in out}
    assert by_eqp["EQP001"]["start_min"] == 0
    assert by_eqp["EQP002"]["start_min"] == 60


def test_normalize_eqp_down_unlimited_and_past_end_filtered():
    rows = [
        {"EQP_ID": "EQP001", "DOWN_START_TM": "20260712060000", "DOWN_END_TM": None},
        {"EQP_ID": "EQP002", "DOWN_START_TM": "20260712080000", "DOWN_END_TM": "20260712100000"},
        {"EQP_ID": "EQP003", "DOWN_START_TM": "20260711000000", "DOWN_END_TM": "20260711120000"},  # 이미 종료
    ]
    out = _normalize_eqp_down(rows, BASE_TIME)
    by_eqp = {r["eqp_id"]: r for r in out}
    assert by_eqp["EQP001"]["down_start_min"] == 0
    assert by_eqp["EQP001"]["down_end_min"] is None
    assert by_eqp["EQP002"]["down_start_min"] == 60
    assert by_eqp["EQP002"]["down_end_min"] == 180
    assert "EQP003" not in by_eqp


def test_forced_conversion_triggers_immediately_when_start_already_due(env_data_factory):
    env_data = env_data_factory(eqp_conv_plan=[
        {"EQP_ID": "EQP001", "FROM_LOT_CD": "LC01", "FROM_TEMP": "T650",
         "TO_LOT_CD": "LC99", "TO_TEMP": "T999", "START_TM": "20260712065000"},  # 10분 전 → 즉시
    ])
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    assert sim.eqps["EQP001"].status == "converting"
    assert sim.eqps["EQP001"].prev_lot_cd == "LC99"
    assert sim.eqps["EQP001"].prev_temp == "T999"
    assert len(sim.conversion_plans) == 1
    plan = sim.conversion_plans[0]
    assert plan["eqp_id"] == "EQP001"
    assert plan["conv_start_min"] == 0
    assert plan["to_lot_cd"] == "LC99"


def test_forced_conversion_scheduled_in_future_does_not_block_current_work(env_data_factory):
    env_data = env_data_factory(eqp_conv_plan=[
        {"EQP_ID": "EQP001", "FROM_LOT_CD": "LC01", "FROM_TEMP": "T650",
         "TO_LOT_CD": "LC99", "TO_TEMP": "T999", "START_TM": "20260713070000"},  # 24시간 뒤
    ])
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    # 미래 예정 전환은 즉시 걸리지 않고, EQP는 정상적으로 idle 배정을 받을 수 있어야 한다.
    assert sim.eqps["EQP001"].status != "converting"


def test_unlimited_downtime_permanently_excludes_eqp(env_data_factory):
    env_data = env_data_factory(eqp_down=[
        {"EQP_ID": "EQP003", "DOWN_START_TM": "20260712065000", "DOWN_END_TM": None},
    ])
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    assert sim.eqps["EQP003"].status == "down"
    assert len(sim.down_windows) == 1
    assert sim.down_windows[0]["down_end_min"] is None

    _run_to_completion(sim)

    # 시뮬 종료까지 EQP003은 계속 down 상태이며 어떤 스케줄도 배정되지 않는다.
    assert sim.eqps["EQP003"].status == "down"
    assert all(rec["EQP_ID"] != "EQP003" for rec in sim.schedule)


def test_downtime_fully_elapsed_while_busy_is_skipped_without_time_travel(env_data_factory):
    """비선점 정책으로 다운 적용이 가공 종료까지 미뤄지는 사이, 다운 구간
    전체(start~end)가 이미 지나버리면 그 구간은 건너뛰어야 한다. 과거 시각으로
    DOWN_END 이벤트를 예약해 시뮬 시계가 역행하는 회귀를 방지한다."""
    env_data = env_data_factory(eqp_down=[
        # EQP001의 첫 가공(LOT001, ST=120분)이 끝나기 훨씬 전에 시작·종료되는 다운 구간
        {"EQP_ID": "EQP001", "DOWN_START_TM": "20260712073000", "DOWN_END_TM": "20260712080000"},  # 30~60분
    ])
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    assert sim.eqps["EQP001"].status != "down"  # 아직 가공 중이라 즉시 적용되지 않음

    _run_to_completion(sim)

    # 시뮬 시계가 역행하지 않았다면 모든 스케줄 행의 END_TM >= START_TM
    assert all(r["END_TM"] >= r["START_TM"] for r in sim.schedule)
    # 다운 구간 전체가 가공 중에 지나갔으므로 실제로는 적용되지 않았어야 한다.
    assert not any(w["eqp_id"] == "EQP001" for w in sim.down_windows)


def test_bounded_downtime_recovers_after_end(env_data_factory):
    env_data = env_data_factory(eqp_down=[
        {"EQP_ID": "EQP002", "DOWN_START_TM": "20260712065000", "DOWN_END_TM": "20260712071000"},  # 0~10분
    ])
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    assert sim.eqps["EQP002"].status == "down"

    _run_to_completion(sim)

    assert sim.down_windows[0]["down_end_min"] == 10
    # 다운 종료 후에는 다시 idle로 복귀해 정상 배정 대상이 될 수 있어야 한다(영구 배제 아님).
    assert sim.eqps["EQP002"].status != "down"
